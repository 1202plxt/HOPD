"""Selected-head LoRA and native Qwen3 causal/video utilities."""
import json
import math
import random
import re
from pathlib import Path


OFFICIAL_USER = ("Please find the visual event described by the sentence '{query}', "
                 "determining its starting and ending times. The format should be: "
                 "'The event happens in <start time> - <end time> seconds'.")


def load_samples(path, video_dir, limit, seed):
    """TimeLens JSONL: video_path + events[{query, span:[start,end]}]."""
    source = Path(path).read_text(encoding="utf-8-sig").strip()
    try:
        raw = json.loads(source)
    except json.JSONDecodeError:
        raw = [json.loads(line) for line in source.splitlines() if line.strip()]
    if isinstance(raw, dict) and "video_path" in raw:
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("要求 TimeLens 训练 JSONL/JSON list：video_path + events")
    rows, invalid, missing = [], 0, 0
    seen = set()
    for record in raw:
        rel = record.get("video_path", "")
        if not rel:
            invalid += 1
            continue
        vp = Path(rel)
        if not vp.suffix:
            vp = vp.with_suffix(".mp4")
        if not vp.is_absolute():
            vp = Path(video_dir) / vp
        if not vp.is_file():
            missing += 1
            continue
        for event in record.get("events", []):
            query, span = event.get("query", ""), event.get("span", [])
            if span and isinstance(span[0], (list, tuple)):
                if len(span) != 1:
                    invalid += 1
                    continue  # A single-interval output cannot represent multiple GT spans.
                span = span[0]
            if not isinstance(query, str) or not query.strip() or len(span) != 2:
                invalid += 1
                continue
            start, end = map(float, span)
            if not all(map(math.isfinite, (start, end))) or not 0 <= start < end:
                invalid += 1
                continue
            key = (str(vp.resolve()), query.strip(), start, end)
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(video_path=key[0], query=key[1], gt_start=start, gt_end=end))
    random.Random(seed).shuffle(rows)
    if len(rows) < limit:
        raise ValueError(f"有效训练样本 {len(rows)} < 请求 {limit}；missing videos={missing}, invalid={invalid}")
    return (rows[:limit] if limit else rows), dict(available=len(rows), missing_video_records=missing, invalid_events=invalid)


def verify_video_timestamps(processor, inputs, metadatas):
    expected = []
    patch = int(processor.video_processor.temporal_patch_size)
    for metadata in metadatas:
        indices = metadata["frames_indices"]
        if hasattr(indices, "tolist"):
            indices = indices.tolist()
        indices = [float(index) for index in indices]
        fps = float(metadata["fps"])
        if (not indices or not math.isfinite(fps) or fps <= 0 or patch < 1
                or any(not math.isfinite(index) for index in indices)
                or any(b < a for a, b in zip(indices, indices[1:]))):
            raise ValueError("视频时间戳 metadata 无效")
        indices += [indices[-1]] * ((-len(indices)) % patch)
        calculate = getattr(processor, '_calculate_timestamps', None)
        if callable(calculate):
            times = list(map(float, calculate(indices, fps, patch)))
        else:
            times = [(indices[i]/fps + indices[i+patch-1]/fps)/2
                     for i in range(0, len(indices), patch)]
        if len(times) != len(indices)//patch or not all(map(math.isfinite, times)):
            raise ValueError('Invalid native processor timestamp output')
        expected.extend(f'{time:.1f}' for time in times)
    decoded = processor.tokenizer.decode(inputs["input_ids"][0].tolist(),
                                          skip_special_tokens=False)
    actual = re.findall(r"<([0-9]+(?:\.[0-9]+)?) seconds>\s*<\|vision_start\|>", decoded)
    if actual != expected:
        raise RuntimeError(f"processor 视频时间戳不一致: expected={expected[:4]}, actual={actual[:4]}")


def attach_head_lora(model, entries, rank, alpha, targets):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class HeadLoRA(nn.Module):
        """Compact matrices structurally restrict delta-W to selected rows/columns."""
        def __init__(self, base, indices, target):
            super().__init__()
            self.base, self.target, self.enabled = base, target, True
            self.scaling = alpha / rank
            self.register_buffer("indices", torch.tensor(indices, device=base.weight.device))
            ni = len(indices) if target == "o_proj" else base.in_features
            no = len(indices) if target in ("q_proj", "v_proj") else base.out_features
            self.A = nn.Parameter(torch.empty(rank, ni, device=base.weight.device, dtype=torch.float32))
            self.B = nn.Parameter(torch.zeros(no, rank, device=base.weight.device, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        @property
        def weight(self):
            return self.base.weight

        def forward(self, x):
            result = self.base(x)
            if not self.enabled:
                return result
            source = x.index_select(-1, self.indices) if self.target == "o_proj" else x
            delta = F.linear(F.linear(source, self.A.to(x.dtype)), self.B.to(x.dtype)) * self.scaling
            if self.target in ("q_proj", "v_proj"):
                return result.index_add(-1, self.indices, delta.to(result.dtype))
            return result + delta.to(result.dtype)

        @torch.no_grad()
        def merge(self):
            delta = (self.B @ self.A * self.scaling).to(self.base.weight.dtype)
            self.base.weight.index_add_(
                0 if self.target in ("q_proj", "v_proj") else 1, self.indices, delta)

    for p in model.parameters():
        p.requires_grad_(False)
    cfg = getattr(model.config, 'text_config', model.config)
    dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))
    if cfg.num_attention_heads % num_kv_heads:
        raise ValueError("query heads 不能整除 KV heads")
    kv_groups = cfg.num_attention_heads // num_kv_heads
    layers = model.model.language_model.layers
    selected = {}
    for entry in entries:
        l, h = int(entry["layer"]), int(entry["head"])
        if not (0 <= l < len(layers) and 0 <= h < cfg.num_attention_heads):
            raise ValueError(f"无效 head L{l}H{h}")
        selected.setdefault(l, []).append(h)
    adapters = {}
    for l, heads in selected.items():
        for target in targets:
            target_heads = (sorted({h // kv_groups for h in heads})
                            if target == "v_proj" else sorted(heads))
            indices = [i for h in target_heads for i in range(h * dim, (h + 1) * dim)]
            attn = layers[l].self_attn
            base = getattr(attn, target)
            width = base.in_features if target == "o_proj" else base.out_features
            expected_heads = num_kv_heads if target == "v_proj" else cfg.num_attention_heads
            if width != expected_heads * dim:
                raise ValueError(f"L{l}.{target} 不是预期 head 布局")
            wrapper = HeadLoRA(base, indices, target)
            setattr(attn, target, wrapper)
            adapters[f"{l}.{target}"] = wrapper
    return adapters


def scoring_inputs(prompt, continuation):
    """Position P-1 predicts y[0]; position P+T-2 predicts y[T-1]."""
    import torch
    inputs = dict(prompt)
    prefix = continuation[:, :-1]
    p = prompt["input_ids"].shape[1]
    inputs["input_ids"] = torch.cat((prompt["input_ids"], prefix), dim=1)
    inputs["attention_mask"] = torch.cat((prompt["attention_mask"], torch.ones_like(prefix)), dim=1)
    for key in ("mm_token_type_ids", "token_type_ids"):
        if key in prompt:
            inputs[key] = torch.cat((prompt[key], torch.zeros_like(prefix)), dim=1)
    # These prediction positions are exactly the final T positions of the
    # scoring sequence. Pass T as an integer so Qwen uses a plain slice rather
    # than an index tensor that Accelerate may move to the wrong GPU.
    logits_to_keep = int(continuation.shape[1])
    return inputs, logits_to_keep


def reset_rope(model):
    # Independent prompts must never reuse the other branch's cached mRoPE offset.
    if hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None


def save_adapter(path, adapters, config):
    import torch
    path.mkdir(parents=True, exist_ok=False)
    torch.save({name: {"A": a.A.detach().cpu(), "B": a.B.detach().cpu(),
                       "indices": a.indices.cpu()} for name, a in adapters.items()}, path / "adapter.pt")
    (path / "adapter_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def messages_for(sample, video, privileged=False, prompt_mode="official"):
    if privileged or prompt_mode != "official":
        raise ValueError("Only normal video/query prompts are supported")
    return [{"role": "user", "content": [video, {"type": "text", "text": OFFICIAL_USER.format(query=sample["query"])}]}]
