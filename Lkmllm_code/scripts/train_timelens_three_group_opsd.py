r"""GT-privileged OPSD for selected TimeLens/Qwen3-VL attention heads.

从 Lkmllm_code 根目录执行；路径沿用当前项目 TimeLens 训练目录：
CUDA_VISIBLE_DEVICES=0 python scripts/train_timelens_three_group_opsd.py \
  --head-json '../Lkmllm_data/outputs/head_knockout_timelens8b_three_groups.json' \
  --anno-json ../Lkmllm_data/datasets/Train/timelens-100k/timelens-100k.jsonl \
  --video-dir ../Lkmllm_data/datasets/Train/timelens-100k \
  --model-path ../shared_models/TimeLens-8B \
  --output-dir ../Lkmllm_data/checkpoints/timelens_three_group_opsd_500_e2 \
  --head-groups gt_positive gt_negative \
  --gt-positive-k 3 --gt-negative-k 2 --prompt-mode official \
  --max-samples 500 --epochs 2 --grad-accum 4 \
  --lr 2e-6 --rank 8 --alpha 16 --max-new-tokens 80 \
  --fps 2 --max-frames 0 --min-tokens 64 --total-tokens 14336

单进程，micro-batch=1（模型可分片到多卡）；可训练 GT 正向、GT 负向、off 正向 head 的
q_proj/v_proj 行和 o_proj 列。每个 head 保留自己的参数切片，不要求不同 head 的方向一致。
每条先由当前学生采样，双方重评分同一学生前缀。旧默认教师关闭 adapter + 精确 GT；
--self-teacher-policy current 保持当前 adapter 并对教师分支停止梯度。
--teacher-mode evidence_frames --selection-dir DIR 改用冻结教师看到的额外 GT 区间
原采样帧，不写入 GT 文本；只允许 gt_negative 组的 Q/O 或 Q/V/O，读取已完成筛选的
config/results/selected/summary，并严格复现每条筛选记录的证据帧时间。
损失=完整回答的 full-vocabulary reverse KL：KL(student || privileged teacher)；
不是 GT teacher-forcing/SFT。默认不截断逐 token RKL，使差异最大的时间 token
保留梯度；训练稳定性由 --max-grad-norm 控制。--kl-clip 仅保留用于消融。
不使用 head hidden loss、Fisher/SCOPE 辅助项或外部本地脚本。
每轮保存自定义 adapter.pt；最终完整模型直接保存到 --output-dir，可用于现有 eval。
先用 --max-samples 2 --epochs 1 --grad-accum 1 --skip-merge 冒烟测试。
依赖 torch/transformers/accelerate/qwen-vl-utils，视频解码环境同现有探测脚本。
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys

# Direct invocation adds scripts/ rather than the repository root to sys.path.
# Keep both package imports (scripts.qwen_decode_worker) and historical sibling
# imports available, independent of the caller's working directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _import_path in (_PROJECT_ROOT, _SCRIPT_DIR):
    if str(_import_path) not in sys.path:
        sys.path.insert(0, str(_import_path))


TRAINING_SYSTEM = ("You are a video time analysis assistant. Given a video and a query, "
                   "you must locate the exact time segment where the described event occurs.")
TRAINING_USER = ("You are given a video with multiple frames. The numbers before each video "
        "frame indicate its sampling timestamp (in seconds). Please find the visual "
        "event described by the sentence '{query}', determining its starting and "
        "ending times. The format should be: "
        "'The event happens in <start time> - <end time> seconds'.")
OFFICIAL_USER = ("Please find the visual event described by the sentence '{query}', "
                 "determining its starting and ending times. The format should be: "
                 "'The event happens in <start time> - <end time> seconds'.")


def load_head_groups(path, gt_positive_k, gt_negative_k, off_positive_k,
                     enabled_groups, min_effect=0.0, require_ci=False):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "head_results" not in data:
        raise ValueError("head-json 必须是 knockout 结果，不能用 attention probe JSON")
    specs = [
        ("gt_positive", data.get("gt_positive_heads", data.get("video_only_top_heads", [])),
         gt_positive_k, 1),
        ("gt_negative", data.get("gt_negative_heads", []), gt_negative_k, -1),
        ("off_positive", data.get("off_positive_heads", []), off_positive_k, 1),
    ]
    groups, chosen = {}, []
    for group, entries, count, sign in specs:
        if group not in enabled_groups:
            continue
        eligible = []
        for raw in entries:
            entry = dict(raw)
            delta = float(entry["delta_miou"])
            ci = entry.get("delta_miou_ci95")
            if not math.isfinite(delta):
                raise ValueError(f"{group} 包含非有限 knockout delta")
            effect_ok = sign * delta > min_effect
            ci_ok = (not require_ci or (ci is not None and
                     (ci[0] > min_effect if sign > 0 else ci[1] < -min_effect)))
            if effect_ok and ci_ok:
                entry["opsd_group"] = group
                eligible.append(entry)
        eligible.sort(key=lambda x: (-sign * x["delta_miou"], x["layer"], x["head"]))
        if len(eligible) < count:
            raise ValueError(f"通过条件的 {group} head 仅 {len(eligible)} 个，要求 {count}")
        groups[group] = eligible[:count]
        chosen.extend(groups[group])
    pairs = [(int(x["layer"]), int(x["head"])) for x in chosen]
    if len(set(pairs)) != len(pairs):
        raise ValueError("三组之间存在重复 head")
    return groups, chosen


def parse_explicit_heads(values, expected_count=0):
    """Accept head coordinates without inventing knockout effects for a control run."""
    if not values:
        raise ValueError("--explicit-heads 至少需要一个 L层H头 坐标")
    if expected_count < 0 or (expected_count and len(values) != expected_count):
        raise ValueError(f"显式 head 数量 {len(values)} 与 --expected-head-count={expected_count} 不符")
    entries, seen = [], set()
    for value in values:
        match = re.fullmatch(r"[Ll]?(\d+)[Hh:](\d+)", value.strip())
        if match is None:
            raise ValueError(f"无效 head 坐标 {value!r}；使用 L16H21 或 16:21")
        pair = int(match.group(1)), int(match.group(2))
        if pair in seen:
            raise ValueError(f"显式 head 重复：L{pair[0]}H{pair[1]}")
        seen.add(pair)
        entries.append({"layer": pair[0], "head": pair[1], "opsd_group": "explicit"})
    return {"explicit": entries}, entries


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


def load_evidence_selection(directory, args):
    """Use only completed, auditable evidence-frame screening results."""
    root = Path(directory)
    config_path = root / "config.json"
    summary_path = root / "summary.json"
    results_path = root / "results.jsonl"
    selected_path = root / "selected.jsonl"
    for path in (config_path, summary_path, results_path, selected_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少筛选文件: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    screened = config.get("arguments", {})
    expected = dict(model_path=args.model_path, fps=args.fps,
                    max_frames=args.max_frames, min_tokens=args.min_tokens,
                    total_tokens=args.total_tokens)
    for key, value in expected.items():
        old = screened.get(key)
        if key == "model_path":
            equal = old is not None and Path(old).resolve() == Path(value).resolve()
        else:
            equal = old == value
        if not equal:
            raise ValueError(f"训练与筛选配置不一致: {key}: {old!r} != {value!r}")
    if (Path(screened.get("video_dir", "")).resolve() != Path(args.video_dir).resolve()
            or args.prompt_mode != "official"):
        raise ValueError("证据帧训练必须沿用筛选时的视频目录与 official prompt")
    if "train" not in Path(screened.get("anno_json", "")).parts and "Train" not in Path(
            screened.get("anno_json", "")).parts:
        raise ValueError("筛选源必须是训练集，不能从评测集选择训练样本")
    if config.get("teacher") != (
            "frozen TimeLens, full original video + original-sampled GT frames, absolute timestamps"):
        raise ValueError("筛选目录不是绝对时间的 GT 原采样帧教师结果")
    max_evidence = int(screened.get("evidence_max_frames", 0))
    if max_evidence < 2 or max_evidence % 2:
        raise ValueError("筛选配置中的 evidence_max_frames 无效")
    results = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if len(results) != summary.get("attempted"):
        raise ValueError("筛选 results.jsonl 与 summary.json 的尝试数不一致")
    chosen = [row for row in results if row.get("decision") == "selected"]
    written = [json.loads(line) for line in selected_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if len(chosen) != summary.get("selected") or len(written) != len(chosen):
        raise ValueError("筛选入选数与 selected.jsonl 不一致")
    samples, keys, video_counts = [], set(), {}
    for row, item in zip(chosen, written):
        events = item.get("events", [])
        if len(events) != 1 or events[0].get("query") != row.get("query"):
            raise ValueError("筛选记录与 selected.jsonl 的 query 不一致")
        span = events[0].get("span")
        if (not isinstance(span, list) or len(span) != 2
                or list(map(float, span)) != [float(row["gt_start"]), float(row["gt_end"])]
                or Path(item["video_path"]).resolve() != Path(row["video_path"]).resolve()):
            raise ValueError("筛选记录与 selected.jsonl 的视频或 GT 不一致")
        student, teacher = row.get("student", {}), row.get("teacher", {})
        gain = float(row.get("iou_gain", float("nan")))
        if (student.get("iou", 2) > screened["student_max_iou"]
                or teacher.get("parse_failure", True)
                or teacher.get("iou", -1) < screened["teacher_min_iou"]
                or not math.isfinite(gain) or gain <= 0
                or gain + 1e-9 < screened["iou_margin"]
                or abs(gain - (teacher["iou"] - student["iou"])) > 1e-6):
            raise ValueError("筛选记录不满足学生难例/教师增益条件")
        padding = row.get("evidence_padding")
        times = row.get("evidence_frame_times")
        if (not isinstance(padding, list) or len(padding) != 2
                or any(not math.isfinite(float(x)) or float(x) < 0 for x in padding)
                or not isinstance(times, list) or not times
                or any(not math.isfinite(float(x)) for x in times)):
            raise ValueError("筛选记录缺少可复现的证据帧时间")
        video_path = str(Path(row["video_path"]).resolve())
        if not Path(video_path).is_file():
            raise FileNotFoundError(video_path)
        key = (video_path, row["query"], float(row["gt_start"]), float(row["gt_end"]))
        if key in keys:
            raise ValueError("筛选结果存在重复事件")
        keys.add(key)
        video_counts[video_path] = video_counts.get(video_path, 0) + 1
        samples.append(dict(video_path=video_path, query=key[1], gt_start=key[2],
                            gt_end=key[3], evidence_padding=[float(x) for x in padding],
                            evidence_frame_times=[float(x) for x in times],
                            screening_index=row["index"], screening_student_iou=student["iou"],
                            screening_teacher_iou=teacher["iou"], screening_iou_gain=gain))
    if any(count > int(screened["max_per_video"]) for count in video_counts.values()):
        raise ValueError("同一视频的入选数量超过筛选配置")
    random.Random(args.seed).shuffle(samples)
    if args.max_samples:
        samples = samples[:args.max_samples]
    if not samples:
        raise ValueError("筛选目录中没有可训练样本")
    stats = dict(available=len(chosen), selected_for_training=len(samples),
                 unique_videos=len(video_counts), screening_attempted=summary["attempted"],
                 screening_selected=summary["selected"], screening_config=str(config_path),
                 screening_summary=str(summary_path), evidence_max_frames=max_evidence)
    return samples, stats


def messages_for(sample, video, privileged=False, prompt_mode="official"):
    if prompt_mode == "official":
        text = OFFICIAL_USER.format(query=sample["query"])
        messages = []
    elif prompt_mode == "taro":
        from taro_protocol import user_prompt
        text = user_prompt(sample["query"])
        messages = []
    elif prompt_mode == "taro_official":
        from taro_eval_protocol import user_prompt
        text = user_prompt(sample["query"], "official")
        messages = []
    elif prompt_mode == "training":
        text = TRAINING_USER.format(query=sample["query"])
        messages = [{"role": "system", "content": [
            {"type": "text", "text": TRAINING_SYSTEM}]}]
    else:
        raise ValueError(f"未知 prompt_mode: {prompt_mode}")
    if privileged:
        text += ("\nVerified training-only reference: the annotated event starts at "
                 f"{sample['gt_start']} seconds and ends at {sample['gt_end']} seconds. "
                 "Use this reference together with the video to answer in the required format.")
    messages.append({"role": "user", "content": [
        video, {"type": "text", "text": text}]})
    return messages


def evidence_teacher_messages(sample, video):
    messages = messages_for(sample, video, privileged=False, prompt_mode="official")
    content = messages[0]["content"]
    content.insert(1, {"type": "text", "text": (
        "Additional close-up video from the same source. Its timestamps use "
        "the first, complete video's timeline. Answer on that timeline.")})
    content.insert(2, {"type": "video", "video": sample["video_path"]})
    return messages


def copy_screened_evidence_frames(tensor, metadata, sample, max_frames, temporal_patch_size):
    indices = metadata["frames_indices"]
    if hasattr(indices, "tolist"):
        indices = indices.tolist()
    indices = [int(index) for index in indices]
    fps = float(metadata["fps"])
    total = int(metadata["total_num_frames"])
    patch = int(temporal_patch_size)
    if (not indices or len(indices) != len(tensor) or not math.isfinite(fps) or fps <= 0
            or total < 2 or patch < 1 or max_frames < patch or max_frames % patch
            or indices[0] < 0 or indices[-1] >= total
            or any(b < a for a, b in zip(indices, indices[1:]))):
        raise ValueError("筛选证据帧的原视频 metadata 无效")
    duration = total / fps
    left_margin, right_margin = sample["evidence_padding"]
    if sample["gt_start"] >= duration or sample["gt_end"] <= 0:
        raise ValueError("GT 区间不在视频时轴内")
    clip_start = max(0.0, sample["gt_start"] - left_margin)
    clip_end = min(duration, sample["gt_end"] + right_margin)
    if clip_end - clip_start < 2.0 / fps:
        clip_start = max(0.0, min(clip_start, duration - 2.0 / fps))
        clip_end = min(duration, max(clip_end, clip_start + 2.0 / fps))
    if clip_end <= clip_start:
        raise ValueError("GT 证据帧窗口为空")
    times = [index / fps for index in indices]
    groups = [list(range(start, min(start + patch, len(indices))))
              for start in range(0, len(indices), patch)]
    wanted = [group for group in groups if any(
        clip_start <= times[position] <= clip_end for position in group)]
    if not wanted:
        wanted = [min(groups, key=lambda group: min(
            max(sample["gt_start"] - times[position],
                times[position] - sample["gt_end"], 0.0) for position in group))]
    budget = max_frames // patch
    if len(wanted) > budget:
        if budget == 1:
            center = (sample["gt_start"] + sample["gt_end"]) / 2
            wanted = [min(wanted, key=lambda group: abs(
                sum(times[position] for position in group) / len(group) - center))]
        else:
            wanted = [wanted[round(i * (len(wanted) - 1) / (budget - 1))]
                      for i in range(budget)]
    positions = [position for group in wanted for position in group]
    selected_times = [times[position] for position in positions]
    expected = sample["evidence_frame_times"]
    if (len(selected_times) != len(expected)
            or any(abs(actual - saved) > 1e-4
                   for actual, saved in zip(selected_times, expected))):
        raise RuntimeError("训练证据帧与筛选时的帧时间不一致；检查视频解码环境")
    clip_metadata = dict(metadata)
    clip_metadata["frames_indices"] = [indices[position] for position in positions]
    return tensor[positions], clip_metadata


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


def prepare_pair(processor, sample, args):
    strict_budget = bool(getattr(args, 'strict_video_budget', False))
    if strict_budget and (getattr(args, 'model_family', 'qwen3vl') != 'qwen3vl'
                          or args.teacher_mode != 'gt_text' or args.video_processor_mode != 'legacy'):
        raise ValueError('Strict native video budget currently requires Qwen3/TimeLens-8B GT-text legacy input')
    if args.teacher_mode == 'frame_hint':
        from qwen_frame_hints import prepare_pair as frame_pair
        return frame_pair(processor, sample, args)
    if getattr(args, 'model_family', 'qwen3vl') == 'qwen25vl':
        return prepare_pair_qwen25(processor, sample, args)
    import copy
    import torch
    if getattr(args, '_head_opsd_protocol', None) or strict_budget:
        from scripts.qwen_decode_worker import process_vision_info
    else:
        from qwen_vl_utils import process_vision_info
    patch = int(processor.image_processor.patch_size)
    video = dict(type="video", video=sample["video_path"], fps=args.fps,
                 min_pixels=args.min_tokens * (patch * 2) ** 2,
                 total_pixels=args.total_tokens * (patch * 2) ** 2)
    if args.max_frames:
        video["max_frames"] = args.max_frames
    for key in ("video_start", "video_end"):
        if sample.get(key) is not None:
            video[key] = float(sample[key])
    messages = messages_for(sample, video, prompt_mode=args.prompt_mode)
    if strict_budget:
        from timelens8b_strict_video_budget import strict_video_budget
        budget_context = strict_video_budget(processor, args.total_tokens)
    else:
        budget_context = contextlib.nullcontext()
    with budget_context as budget_audit:
        if strict_budget:
            from scripts import qwen_decode_worker as strict_decoder
            vision_process = strict_decoder.process_vision_info
        else:
            vision_process = process_vision_info
        images, videos, kwargs = vision_process(
            messages, image_patch_size=patch, return_video_kwargs=True,
            return_video_metadata=True)
    if images or not videos or len(videos) != 1:
        raise ValueError("每条训练样本必须且只能包含一个视频")
    tensors, metadata = map(list, zip(*videos))
    metadata = copy.deepcopy(metadata)
    if args.video_processor_mode == "eval":
        for item in metadata:
            indices = [int(index) for index in item["frames_indices"]]
            if not indices:
                raise ValueError("视频没有采样帧")
            item["frames_indices"] = [index - indices[0] for index in indices]
        options = dict(kwargs)
        options.setdefault("do_sample_frames", False)
    else:
        options = dict(kwargs, do_resize=False, do_sample_frames=False)
    student_text = processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
    student = dict(processor(text=[student_text], videos=tensors,
                             video_metadata=copy.deepcopy(metadata),
                             return_tensors="pt", padding=True, **options))
    if getattr(args, '_head_opsd_protocol', None) or strict_budget:
        from qwen_head_opsd import json_value
        verify_video_timestamps(processor, student, metadata)
        args._last_video_audit = dict(
            pipeline='native_qwen3_shared_decode_gt_text_v1',
            video_grid_thw=student['video_grid_thw'].tolist(),
            metadata=json_value(metadata), do_resize=False, do_sample_frames=False)
        if strict_budget:
            actual = int((student['input_ids'] == processor.tokenizer.convert_tokens_to_ids('<|video_pad|>')).sum())
            if not 0 < actual <= args.total_tokens or actual != budget_audit['estimated_visual_tokens']:
                raise RuntimeError('Actual video token count violates strict budget')
            args._last_video_audit.update(actual_visual_tokens=actual, strict_budget=budget_audit)
            from qwen_head_opsd import append_json
            append_json(Path(args.output_dir) / 'visual_budget.jsonl', dict(video=sample['video_path'], **args._last_video_audit))
    if args.teacher_mode == "gt_text":
        teacher_text = processor.apply_chat_template(
            messages_for(sample, video, privileged=True, prompt_mode=args.prompt_mode),
            tokenize=False, add_generation_prompt=True)
        teacher = dict(processor(text=[teacher_text], videos=tensors,
                                 video_metadata=copy.deepcopy(metadata),
                                 return_tensors="pt", padding=True, **options))
        if getattr(args, '_head_opsd_protocol', None):
            verify_video_timestamps(processor, teacher, metadata)
        for key in ("pixel_values_videos", "video_grid_thw"):
            if not torch.equal(student[key], teacher[key]):
                raise RuntimeError(f"教师/学生视频处理不一致: {key}")
            teacher[key] = student[key]  # Same video; only text differs.
    elif args.teacher_mode == "evidence_frames":
        verify_video_timestamps(processor, student, metadata)
        clip_tensor, clip_metadata = copy_screened_evidence_frames(
            tensors[0], metadata[0], sample, args.evidence_max_frames,
            processor.video_processor.temporal_patch_size)
        teacher_text = processor.apply_chat_template(
            evidence_teacher_messages(sample, video), tokenize=False,
            add_generation_prompt=True)
        teacher = dict(processor(text=[teacher_text],
                                 videos=[tensors[0], clip_tensor],
                                 video_metadata=[copy.deepcopy(metadata[0]), clip_metadata],
                                 return_tensors="pt", padding=True, **options))
        student_grid, teacher_grid = student["video_grid_thw"], teacher["video_grid_thw"]
        if (student_grid.ndim != 2 or student_grid.shape[0] != 1
                or teacher_grid.ndim != 2 or teacher_grid.shape[0] != 2
                or not torch.equal(student_grid[0], teacher_grid[0])):
            raise RuntimeError("教师的完整原视频与学生视频不一致")
        verify_video_timestamps(processor, teacher, [metadata[0], clip_metadata])
    else:
        raise ValueError(f"未知教师模式: {args.teacher_mode}")
    return student, teacher


def validate_qwen25_training_args(args):
    if getattr(args, 'model_family', 'qwen3vl') != 'qwen25vl':
        return
    if (args.teacher_mode not in ('gt_text', 'frame_hint') or args.prompt_mode != 'official'
            or args.video_processor_mode != 'legacy'
            or getattr(args, 'opsd_token_scope', 'all') != 'all'
            or args.time_ce_weight != 0):
        raise ValueError('Qwen2.5 OPSD requires native legacy video, official prompt, gt_text, full-answer KL and no CE')


def validate_qwen25_screen(path, args):
    """Reject other weights, budgets, partial results and replacement rankings."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    meta = data.get('_meta', {})
    cfg = meta.get('intervention_config', {})
    if not data.get('selection_ready') or not meta.get('all_requested_work_complete'):
        raise ValueError('Knockout screening is not complete/selection-ready')
    if cfg.get('model_family') != 'qwen25vl' or cfg.get('native_input_protocol') != 'qwen25_native_file_v1':
        raise ValueError('Requires a native Qwen2.5 knockout result')
    if cfg.get('mode') != 'zero' or cfg.get('scope') != 'all':
        raise ValueError('Requires zero/all knockout, not replacement or response-only intervention')
    if Path(cfg.get('model_path', '')).resolve() != Path(args.model_path).resolve():
        raise ValueError('Knockout and training base model paths differ')
    for key in ('fps', 'min_tokens', 'total_tokens', 'max_frames'):
        if cfg.get(key) != getattr(args, key):
            raise ValueError(f'Knockout/training input budget differs: {key}')
    if cfg.get('video_processor_mode') != 'legacy':
        raise ValueError('Requires native Qwen2.5 video protocol')
    if data.get('n_complete_samples', 0) <= 0:
        raise ValueError('No complete paired knockout samples')
    return data


def prepare_pair_qwen25(processor, sample, args):
    """Decode once, retain native temporal metadata for both independent prompts."""
    import torch
    import qwen25_head_inputs as native
    native.validate_processor(processor)
    video = native.video_message(sample, args.fps, args.min_tokens,
                                 args.total_tokens, args.max_frames)[-1]['content'][0]
    for key in ('video_start', 'video_end'):
        if sample.get(key) is not None:
            video[key] = float(sample[key])
    student_messages = messages_for(sample, video, prompt_mode='official')
    teacher_messages = messages_for(sample, video, privileged=True, prompt_mode='official')
    decoded = native.decode_video(student_messages)
    student = native.process_messages(processor, student_messages, decoded)
    teacher = native.process_messages(processor, teacher_messages, decoded)
    for key in ('pixel_values_videos', 'video_grid_thw'):
        if not torch.equal(student[key], teacher[key]):
            raise RuntimeError(f'Native Qwen2.5 teacher/student visual mismatch: {key}')
        teacher[key] = student[key]
    if student['second_per_grid_ts'] != teacher['second_per_grid_ts']:
        raise RuntimeError('Native Qwen2.5 teacher/student temporal spacing differs')
    args._last_video_audit = native.audit_inputs(processor, student, decoded)
    return student, teacher


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


@contextlib.contextmanager
def teacher_mode(adapters):
    states = [a.enabled for a in adapters.values()]
    try:
        for a in adapters.values():
            a.enabled = False
        yield
    finally:
        for a, state in zip(adapters.values(), states):
            a.enabled = state


def self_teacher_context(args, adapters):
    """Teacher logits are detached in either case; current keeps trained adapters."""
    if getattr(args, 'self_teacher_policy', 'frozen_base') == 'current':
        return contextlib.nullcontext()
    return teacher_mode(adapters)


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


def timestamp_answer_tokens(tokenizer, sample):
    """Canonical answer IDs and separate start/end token masks for GT CE."""
    start, end = str(sample["gt_start"]), str(sample["gt_end"])
    prefix, middle = "The event happens in ", " - "
    answer = prefix + start + middle + end + " seconds."
    start_span = (len(prefix), len(prefix) + len(start))
    end_begin = start_span[1] + len(middle)
    end_span = (end_begin, end_begin + len(end))
    encoded = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    if not ids or len(ids) != len(offsets):
        raise RuntimeError("Tokenizer did not provide answer token offsets")
    start_mask = [left < start_span[1] and right > start_span[0]
                  for left, right in offsets]
    end_mask = [left < end_span[1] and right > end_span[0]
                for left, right in offsets]
    if not any(start_mask) or not any(end_mask):
        raise RuntimeError("Tokenizer did not expose both GT timestamp spans")
    if any(a and b for a, b in zip(start_mask, end_mask)):
        raise RuntimeError("A token spans both GT timestamps; CE target is ambiguous")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS")
    ids.append(tokenizer.eos_token_id)
    start_mask.append(False)
    end_mask.append(False)
    return ids, start_mask, end_mask


def reset_rope(model):
    # Independent prompts must never reuse the other branch's cached mRoPE offset.
    if hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None


def reverse_kl_per_token(student_logits, teacher_logits):
    """Exact full-vocabulary KL(student || stopgrad(teacher)) per token."""
    from torch.nn import functional as F
    log_p = F.log_softmax(student_logits.float(), dim=-1)
    log_q = F.log_softmax(teacher_logits.float(), dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1)


def rollout_metrics(text, sample, prompt_mode="official"):
    """Diagnostic only: never used as a reward or as a training target."""
    if prompt_mode in ("taro", "taro_official"):
        from taro_protocol import parse_final_answer
        parsed = parse_final_answer(text)
        if parsed is None:
            return dict(rollout_pred=None, rollout_iou=0.0, parse_failure=True)
        start, end = parsed[:2]
        intersection = max(0.0, min(end, sample["gt_end"]) - max(start, sample["gt_start"]))
        union = max(end, sample["gt_end"]) - min(start, sample["gt_start"])
        return dict(rollout_pred=[start, end], rollout_iou=intersection / union,
                    parse_failure=False)
    number = r"([0-9]+(?:\.[0-9]+)?)"
    match = re.search(r"event\s+(?:happens|occurs)\s+in\s+" + number
                      + r"\s*(?:[-–—]|to)\s*" + number + r"\s*seconds?", text, re.I)
    if match is None:
        return dict(rollout_pred=None, rollout_iou=0.0, parse_failure=True)
    start, end = map(float, match.groups())
    if not 0 <= start < end:
        return dict(rollout_pred=None, rollout_iou=0.0, parse_failure=True)
    intersection = max(0.0, min(end, sample["gt_end"]) - max(start, sample["gt_start"]))
    union = max(end, sample["gt_end"]) - min(start, sample["gt_start"])
    return dict(rollout_pred=[start, end], rollout_iou=intersection / union, parse_failure=False)


def save_adapter(path, adapters, config):
    import torch
    path.mkdir(parents=True, exist_ok=False)
    torch.save({name: {"A": a.A.detach().cpu(), "B": a.B.detach().cpu(),
                       "indices": a.indices.cpu()} for name, a in adapters.items()}, path / "adapter.pt")
    (path / "adapter_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def inspect_warm_start(args, samples, heads, protocol):
    """Validate a weights-only branch, without loading torch or changing its parent."""
    snapshot = Path(args.warm_start_snapshot).resolve()
    root = snapshot.parent
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    saved = read(snapshot / 'adapter_config.json')
    summary = read(snapshot / 'summary.json')
    if saved != read(root / 'training_metadata.json'):
        raise ValueError('Snapshot configuration differs from parent training metadata')
    previous = saved['arguments']
    old_protocol = saved.get('head_opsd_protocol', {})
    method = 'qwen25vl_gt_damage_positive_upto10_qo_gt_text_opsd_14336_v4'
    if (not protocol or protocol.get('method') != method or old_protocol.get('method') != method
            or not args.iou_gate or args.epochs != 1):
        raise ValueError('Weights-only continuation is restricted to gated Qwen2.5 GT-text 14336')
    output = Path(args.output_dir).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError('Weights-only continuation requires a separate new output directory')
    for key in ('model_path', 'anno_json', 'video_dir'):
        if Path(previous[key]).resolve() != Path(getattr(args, key)).resolve():
            raise ValueError('Warm-start input path differs: ' + key)
    for key in ('model_family', 'teacher_mode', 'self_teacher_policy', 'prompt_mode',
                'video_processor_mode', 'fps', 'min_tokens', 'total_tokens', 'max_frames',
                'rank', 'alpha', 'targets', 'lr', 'seed', 'epochs', 'grad_accum',
                'max_samples', 'max_new_tokens', 'temperature', 'top_p', 'max_grad_norm',
                'kl_clip', 'time_ce_weight', 'opsd_token_scope', 'iou_gate',
                'teacher_min_iou', 'iou_margin', 'max_attempts'):
        if previous.get(key) != getattr(args, key):
            raise ValueError('Warm-start recipe differs: ' + key)
    for key in ('annotation_sha256', 'selected_events_sha256', 'checkpoint_config_sha256',
                'head_selection', 'gate', 'layout', 'teacher_policy'):
        if old_protocol.get(key) != protocol.get(key):
            raise ValueError('Warm-start protocol differs: ' + key)
    coordinates = lambda rows: [(row['layer'], row['head']) for row in rows]
    if coordinates(saved['selected_heads']) != coordinates(heads):
        raise ValueError('Warm-start selected heads differ')
    if read(root / 'candidate_samples.json') != samples:
        raise ValueError('Warm-start candidate pool/order differs')
    accepted, cursor, steps = (summary[key] for key in ('global_sample', 'sample', 'optimizer_steps'))
    if (any(type(n) is not int for n in (accepted, cursor, steps))
            or summary.get('epoch') != 1 or not 0 < accepted < args.max_samples
            or accepted % args.grad_accum or steps != accepted // args.grad_accum
            or not accepted <= cursor <= len(samples)
            or (args.max_attempts and cursor >= args.max_attempts)):
        raise ValueError('Snapshot is not an unfinished optimizer-boundary checkpoint')
    events = read(root / 'train_samples.json')[:accepted]
    if len(events) != accepted:
        raise ValueError('Parent accepted-event journal is shorter than its snapshot')
    last_attempt = 0
    for index, event in enumerate(events, 1):
        attempt = event.get('attempt')
        if (event.get('accepted_index') != index or type(attempt) is not int
                or not last_attempt < attempt <= cursor
                or {k: event[k] for k in samples[attempt - 1]} != samples[attempt - 1]):
            raise ValueError('Parent accepted events do not match the candidate prefix')
        last_attempt = attempt
    if last_attempt != cursor:
        raise ValueError('Snapshot cursor does not end at its last accepted event')
    adapter_path = snapshot / 'adapter.pt'
    digest = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    return dict(snapshot=str(snapshot), adapter_sha256=digest,
                inherited_accepted=accepted, candidate_cursor=cursor,
                inherited_optimizer_steps=steps, accepted_events=events,
                optimizer_reset=True, rng_reset=True,
                note='Weights-only continuation in a new directory; AdamW and RNG restart; '
                     'post-snapshot candidates are replayed; not an exact training resume')


def load_warm_start_adapters(adapters, recovery):
    import torch
    path = Path(recovery['snapshot']) / 'adapter.pt'
    if hashlib.sha256(path.read_bytes()).hexdigest() != recovery['adapter_sha256']:
        raise ValueError('Warm-start adapter changed after preflight')
    states = torch.load(path, map_location='cpu', weights_only=True)
    if set(states) != set(adapters):
        raise ValueError('Warm-start adapter module names differ')
    # Validate everything before copying any weights.
    for name, adapter in adapters.items():
        state = states[name]
        if set(state) != {'A', 'B', 'indices'} or not torch.equal(state['indices'], adapter.indices.cpu()):
            raise ValueError('Warm-start adapter indices differ: ' + name)
        for key in ('A', 'B'):
            value = state[key]
            parameter = getattr(adapter, key)
            if value.shape != parameter.shape or value.dtype != parameter.dtype or not torch.isfinite(value).all():
                raise ValueError('Invalid warm-start adapter tensor: ' + name + '.' + key)
    with torch.no_grad():
        for name, adapter in adapters.items():
            adapter.A.copy_(states[name]['A'])
            adapter.B.copy_(states[name]['B'])


def scoring_memory_context(args):
    if not getattr(args, 'offload_saved_tensors', False):
        return contextlib.nullcontext()
    import torch
    return torch.autograd.graph.save_on_cpu(pin_memory=True)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model-family', choices=['qwen3vl', 'qwen25vl'], default='qwen3vl')
    p.add_argument('--strict-video-budget', action='store_true',
                   help='For Qwen3 GT-text: uniformly thin decoded frames to enforce actual token ceiling')
    for key in ("video-dir", "model-path", "output-dir"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--head-json", help="knockout 结果；与 --explicit-heads 二选一")
    p.add_argument("--explicit-heads", nargs="+", metavar="L16H21",
                   help="直接指定 head 坐标，不读取/伪造 knockout 分数；用于随机对照")
    p.add_argument("--expected-head-count", type=int, default=0,
                   help="显式名单数量校验；例如随机十头设为 10")
    p.add_argument("--anno-json", help="普通 GT 文本 OPSD 的训练标注；证据帧模式改用 --selection-dir")
    p.add_argument("--selection-dir", help="已完成的 GT 证据帧筛选目录，读取 selected/results/config/summary")
    p.add_argument("--teacher-mode", choices=["gt_text", "frame_hint", "evidence_frames"], default="gt_text")
    p.add_argument('--self-teacher-policy', choices=['frozen_base', 'current'],
                   default='frozen_base',
                   help='current: same current adapters, stop-gradient teacher; frozen_base: disable adapters (legacy)')
    for key, default in (("gt-positive-k", 3), ("gt-negative-k", 2), ("off-positive-k", 1),
                         ("max-samples", 500), ("epochs", 2), ("grad-accum", 4),
                         ("rank", 8), ("max-new-tokens", 80), ("seed", 42), ("max-frames", 0),
                         ("min-tokens", 64), ("total-tokens", 14336),
                         ("save-every", 0)):
        p.add_argument("--" + key, type=int, default=default)
    for key, default in (("lr", 1e-5), ("alpha", 16), ("temperature", 1.0), ("top-p", 0.9),
                         ("fps", 2.0), ("min-effect", 0.0), ("max-grad-norm", 1.0),
                         ("kl-clip", 0.0), ("time-ce-weight", 0.0)):
        p.add_argument("--" + key, type=float, default=default)
    p.add_argument("--targets", nargs="+", choices=["q_proj", "v_proj", "o_proj"],
                   default=["q_proj", "v_proj", "o_proj"])
    p.add_argument("--head-groups", nargs="+",
                   choices=["gt_positive", "gt_negative", "off_positive"],
                   default=None,
                   help="本次参与 OPSD 的 head 类别，可传一个或多个")
    p.add_argument("--prompt-mode", choices=["official", "training", "taro", "taro_official"], default="official",
                   help="taro 为旧短提示词；taro_official 与 taro_eval_vllm 默认官方提示词一致")
    p.add_argument("--video-processor-mode", choices=["legacy", "eval"], default="legacy",
                   help="eval 对齐 taro_eval_vllm 的帧索引重基准及 processor kwargs；旧实验默认 legacy")
    p.add_argument("--attn-implementation", choices=["sdpa", "flash_attention_2"], default="sdpa")
    p.add_argument("--device-map", choices=["single", "auto", "balanced", "balanced_low_0"],
                   default="single", help="多卡模型分片使用 balanced；不要配合 torchrun")
    p.add_argument("--gpu-mem-gib", type=float, default=21.0)
    p.add_argument("--gpu0-mem-gib", type=float, default=0.0,
                   help="可选：单独限制 GPU0 的权重放置预算，为视频反向激活留空间")
    p.add_argument("--require-ci", action="store_true")
    p.add_argument("--opsd-token-scope", choices=["all", "taro_timestamps"], default="all",
                   help="all保持完整回答RKL；taro_timestamps仅最终answer时间戳预测位置的全词表RKL")
    p.add_argument("--timestamp-invalid-policy", choices=["skip", "error"], default="skip",
                   help="时间戳模式遇到无有效最终答案/对齐失败：记录并跳过，或立即报错；不回退完整回答loss")
    p.add_argument("--skip-merge", action="store_true")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument('--offload-saved-tensors', action='store_true',
                   help='Store autograd saved tensors on CPU during student scoring/backward; trades RAM/transfers for VRAM')
    p.add_argument('--warm-start-snapshot',
                   help='Qwen2.5 GT-text 14336 only: inherit adapter and candidate cursor into a NEW directory; resets AdamW/RNG')
    return p


def main(args=None):
    args = build_parser().parse_args() if args is None else args
    online_gate = bool(getattr(args, '_head_opsd_protocol', None) and getattr(args, 'iou_gate', False))
    if args.teacher_mode == 'frame_hint' and not getattr(args, '_head_opsd_protocol', None):
        raise ValueError('Use the validated train_qwen25_head_opsd.py or train_qwen3_head_opsd.py frame-hint entry')
    args.opsd_token_scope = getattr(args, "opsd_token_scope", "all")
    args.timestamp_invalid_policy = getattr(args, "timestamp_invalid_policy", "skip")
    validate_qwen25_training_args(args)
    timestamp_only = args.opsd_token_scope == "taro_timestamps"
    if timestamp_only and args.prompt_mode not in ("taro", "taro_official"):
        raise ValueError("taro_timestamps仅支持TaRO最终answer时间戳")
    if timestamp_only:
        try:
            from .taro_timestamp_opsd import timestamp_selection, select_prediction_rows, accumulation_scale
        except ImportError:
            from taro_timestamp_opsd import timestamp_selection, select_prediction_rows, accumulation_scale
    explicit_mode = bool(args.explicit_heads)
    if explicit_mode == bool(args.head_json):
        raise ValueError("必须且只能提供 --head-json 或 --explicit-heads 之一")
    if explicit_mode:
        if (args.teacher_mode != "gt_text" or args.head_groups is not None
                or args.require_ci or args.min_effect > 0):
            raise ValueError("显式 head 对照仅支持 GT 文本教师，且不使用 head 分组、CI 或筛选效应阈值")
    elif args.head_groups is None:
        args.head_groups = ["gt_positive", "gt_negative"]
    if args.expected_head_count < 0 or (not explicit_mode and args.expected_head_count):
        raise ValueError("--expected-head-count 仅用于显式 head 名单，且不能为负数")
    if args.prompt_mode in ("taro", "taro_official") and (args.teacher_mode != "gt_text" or args.time_ce_weight > 0):
        raise ValueError("TaRO 当前仅支持 GT 文本教师与纯 OPSD；时间 CE 目标仍是 TimeLens 答案格式")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("此版本为单进程训练；多卡模型分片使用 --device-map balanced，不要用 torchrun")
    if min(args.epochs, args.grad_accum, args.rank,
           args.max_new_tokens, args.min_tokens, args.total_tokens) < 1:
        raise ValueError("计数参数必须为正")
    if args.save_every < 0 or (args.save_every and args.save_every % args.grad_accum):
        raise ValueError("--save-every 必须为非负数，且能被 --grad-accum 整除")
    if args.teacher_mode in ("gt_text", 'frame_hint'):
        if not args.anno_json or args.selection_dir or args.max_samples < 1:
            raise ValueError("GT 文本模式需要 --anno-json 和正的 --max-samples，不能传 --selection-dir")
    else:
        if (not args.selection_dir or args.anno_json or args.max_samples < 0
                or args.head_groups != ["gt_negative"]
                or set(args.targets) not in ({"q_proj", "o_proj"},
                                             {"q_proj", "v_proj", "o_proj"})):
            raise ValueError("证据帧模式需要 --selection-dir、仅 gt_negative 组、Q/O 或 Q/V/O；不传 --anno-json")
    if args.head_groups is not None and len(args.head_groups) != len(set(args.head_groups)):
        raise ValueError("head-groups 重复")
    group_counts = {"gt_positive": args.gt_positive_k,
                    "gt_negative": args.gt_negative_k,
                    "off_positive": args.off_positive_k}
    if not explicit_mode and any(group_counts[group] < 1 for group in args.head_groups):
        raise ValueError("启用类别对应的 head 数量必须为正")
    if (args.max_frames < 0 or args.max_frames % 2 or args.lr <= 0 or args.alpha <= 0
            or args.fps <= 0 or args.temperature <= 0 or not 0 < args.top_p <= 1
            or args.min_effect < 0 or args.max_grad_norm <= 0 or args.gpu_mem_gib <= 0
            or args.gpu0_mem_gib < 0 or args.kl_clip < 0
            or not math.isfinite(args.time_ce_weight) or args.time_ce_weight < 0):
        raise ValueError("无效超参数；max-frames 为 0 或正偶数")
    if len(args.targets) != len(set(args.targets)):
        raise ValueError("targets 重复")
    if explicit_mode:
        head_groups, heads = parse_explicit_heads(
            args.explicit_heads, args.expected_head_count)
    else:
        if getattr(args, 'model_family', 'qwen3vl') == 'qwen25vl':
            validate_qwen25_screen(args.head_json, args)
        head_groups, heads = load_head_groups(
            args.head_json, args.gt_positive_k, args.gt_negative_k,
            args.off_positive_k, set(args.head_groups), args.min_effect, args.require_ci)
    def ci_supports(entry):
        ci = entry.get("delta_miou_ci95")
        if not ci:
            return False
        return ci[1] < 0 if entry["opsd_group"] == "gt_negative" else ci[0] > 0
    if not explicit_mode and any(not ci_supports(h) for h in heads):
        print("[pilot] 部分选中 head 的 CI 跨零；本轮作为探索实验，head 列表写入训练元数据。", flush=True)
    if args.teacher_mode == "evidence_frames":
        samples, stats = load_evidence_selection(args.selection_dir, args)
        args.evidence_max_frames = stats["evidence_max_frames"]
    else:
        samples, stats = load_samples(args.anno_json, args.video_dir, 0 if online_gate else args.max_samples, args.seed)
        if online_gate and len(samples) < args.max_samples:
            raise ValueError('Candidate pool is smaller than the requested accepted target')
        if (not online_gate and getattr(args, 'model_family', 'qwen3vl') == 'qwen25vl'
                and len(samples) != args.max_samples):
            raise ValueError(f'Qwen2.5 training pool only supplies {len(samples)}/{args.max_samples} samples')
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("output-dir 已有文件，请使用新目录")
    recovery = None
    if getattr(args, 'warm_start_snapshot', None):
        recovery = inspect_warm_start(args, samples, heads, getattr(args, '_head_opsd_protocol', None))
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    n_gpus = torch.cuda.device_count()
    if args.device_map == "single" or n_gpus == 1:
        load_device_map, max_memory = {"": 0}, None
    else:
        max_memory = {i: f"{args.gpu_mem_gib:g}GiB" for i in range(n_gpus)}
        if args.gpu0_mem_gib > 0:
            max_memory[0] = f"{args.gpu0_mem_gib:g}GiB"
        max_memory["cpu"] = 0
        load_device_map = args.device_map
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map=load_device_map,
        max_memory=max_memory, attn_implementation=args.attn_implementation,
        local_files_only=True)
    if (model.config.model_type == 'qwen2_5_vl'
            and getattr(args, 'model_family', 'qwen3vl') != 'qwen25vl'):
        raise ValueError('Native Qwen2.5 training requires --model-family qwen25vl')
    if args.device_map != "single" and n_gpus > 1:
        used = {parameter.device.index for parameter in model.parameters()
                if parameter.device.type == "cuda" and parameter.device.index is not None}
        for device in getattr(model, "hf_device_map", {}).values():
            if isinstance(device, int):
                used.add(device)
            else:
                name = str(device)
                if name.isdigit():
                    used.add(int(name))
                elif name.startswith("cuda:"):
                    used.add(int(name.split(":", 1)[1]))
        if used != set(range(n_gpus)):
            raise RuntimeError(f"模型分片未使用全部可见 GPU：used={sorted(used)}")
    if "logits_to_keep" not in inspect.signature(model.forward).parameters:
        raise RuntimeError("当前 Transformers 模型不支持 logits_to_keep，请使用项目 Qwen3-VL 环境")
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, padding_side="left")
    if getattr(args, 'model_family', 'qwen3vl') == 'qwen25vl':
        if model.config.model_type != 'qwen2_5_vl':
            raise ValueError('Expected native Qwen2.5-VL checkpoint')
        from qwen25_head_inputs import validate_processor
        validate_processor(processor)
        model.generation_config.num_beams = 1
        model.generation_config.num_return_sequences = 1
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    adapters = attach_head_lora(model, heads, args.rank, args.alpha, args.targets)
    if recovery:
        load_warm_start_adapters(adapters, recovery)
    params = [p for p in model.parameters() if p.requires_grad]
    if {id(p) for p in params} != {id(p) for a in adapters.values() for p in (a.A, a.B)}:
        raise RuntimeError("发现非选中 head 的可训练参数")
    head_protocol = getattr(args, '_head_opsd_protocol', None)
    if head_protocol and sum(p.numel() for p in params) != head_protocol['expected_trainable_parameters']:
        raise RuntimeError('Actual trainable parameters differ from the compact Q/O preflight count')
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    # Disable dropout for the scoring pass too; sampling provides rollout diversity.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if hasattr(module, "attention_dropout"):
            module.attention_dropout = 0.0
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    input_device = model.get_input_embeddings().weight.device
    if args.prompt_mode == "official":
        prompt_spec = {"mode": "official", "system": None, "user_template": OFFICIAL_USER}
    elif args.prompt_mode == "taro":
        from taro_protocol import USER_TEMPLATE
        prompt_spec = {"mode": "taro", "system": None, "user_template": USER_TEMPLATE,
                       "final_answer_format": "<answer>start time to end time</answer>"}
    elif args.prompt_mode == "taro_official":
        from taro_eval_protocol import OFFICIAL_PROMPT
        prompt_spec = {"mode": "taro_official", "system": None, "user_template": OFFICIAL_PROMPT,
                       "final_answer_format": "<answer>start time to end time</answer>"}
    else:
        prompt_spec = {"mode": "training", "system": TRAINING_SYSTEM,
                       "user_template": TRAINING_USER}
    config = dict(arguments=dict(vars(args)), selected_heads=heads, selected_head_groups=head_groups,
                  head_selection_source=("explicit_cli_control" if explicit_mode else "knockout_json"),
                  data_stats=stats,
                  trainable_parameters=sum(p.numel() for p in params),
                  method=("GT text privileged full-vocabulary reverse-KL on student rollouts"
                          if args.teacher_mode == "gt_text" else
                          "GT interval evidence frames privileged full-vocabulary reverse-KL on student rollouts"),
                  video_pipeline="qwen_vl_utils_file_v1", torch_version=torch.__version__,
                  input_protocol=("taro_official_input_v1" if args.prompt_mode == "taro_official"
                                  and args.video_processor_mode == "eval" else "legacy_or_custom"),
                  transformers_version=transformers.__version__,
                  adapter_format="compact_head_lora_v1", seed=args.seed,
                  student_prompt=prompt_spec,
                  teacher_privilege=("exact GT timestamps appended after student user text"
                                     if args.teacher_mode == "gt_text" else
                                     "complete original video plus screened GT-near original sampled frames "
                                     "with absolute timestamps; no GT text"),
                  loss=("mean_tokens clamp(KL(student || stopgrad(teacher)), max=kl_clip), "
                        "full vocabulary; kl_clip=0 disables clipping"),
                  update_scope=("selected groups; q_proj query-head output rows, v_proj mapped "
                                "KV-head output rows, and o_proj query-head input columns"),
                  group_objective=("all groups independently parameterized and jointly optimized toward "
                                   "the same GT-privileged teacher distribution"))
    protocol = getattr(args, '_head_opsd_protocol', None)
    if protocol:
        config.update(method=protocol['method'], head_opsd_protocol=protocol,
                      teacher_policy=protocol['teacher_policy'],
                      update_scope='selected query-head q_proj output rows and o_proj input columns only')
        if args.teacher_mode == 'frame_hint':
            config.update(teacher_privilege='teacher-only sampled temporal-group IDs and green GT-evidence frame borders; no exact GT text',
                          video_pipeline='same native sampling/grid/timestamps; teacher pixels carry training-only marks')
    if recovery:
        config['weights_only_continuation'] = {k: v for k, v in recovery.items() if k != 'accepted_events'}
    if getattr(args, 'model_family', 'qwen3vl') == 'qwen25vl':
        prompt_note = ('official user-only for student/evaluation; selected by 4096 GT-region diagnostic; training budget recorded in head_opsd_protocol'
                       if protocol and protocol.get('method') == 'qwen25vl_gt_damage_positive_upto10_qo_gt_text_opsd_14336_v4'
                       else 'official user-only for training/evaluation; knockout uses its saved native prompt; budgets aligned')
        config.update(model_family='qwen25vl', video_pipeline='qwen25_native_file_v1',
                      input_protocol='qwen25_native_file_v1',
                      prompt_protocol_note=prompt_note)
        if not explicit_mode:
            config['head_selection_provenance'] = dict(
                path=str(Path(args.head_json).resolve()),
                sha256=hashlib.sha256(Path(args.head_json).read_bytes()).hexdigest(),
                ranking=('gt_positive; descending baseline-minus-knockout delta; ties by layer/head'
                         if args.head_groups == ['gt_positive'] else
                         'selected signed groups; baseline-minus-knockout delta; ties by layer/head'),
                min_effect=args.min_effect, require_ci=args.require_ci)
    if args.time_ce_weight > 0:
        config["method"] += " plus teacher-forced GT timestamp CE"
        config["loss"] += (" + time_ce_weight * 0.5 * (CE_GT_start + CE_GT_end); "
                           "only GT timestamp sub-tokens are supervised by CE")
    if timestamp_only:
        config["method"] += " restricted to student-generated final-answer timestamps"
        config["loss"] = ("mean over generated start/end timestamp prediction rows of full-vocabulary "
                          "KL(student || stopgrad(GT-privileged teacher)); same student prefix; "
                          "optional per-row kl_clip; no GT teacher-forcing or CE")
        config["timestamp_opsd"] = dict(
            scope="one strict final <answer> start to end </answer>",
            alignment="actual generated IDs; exact re-encoding offsets or stable decoded prefixes",
            punctuation="decimal points and clock colons included; tags/think/units/EOS not independent targets",
            fused_tokens="any token overlapping timestamp characters is indivisible; count logged",
            invalid_policy=args.timestamp_invalid_policy,
            accumulation="mean over valid samples per original group; empty groups have no optimizer step")
    out.mkdir(parents=True, exist_ok=True)
    (out / "training_metadata.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / ("candidate_samples.json" if online_gate else "train_samples.json")).write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    step = recovery['inherited_optimizer_steps'] if recovery else 0
    initial_step = step
    global_sample = recovery['inherited_accepted'] if recovery else 0
    attempted = recovery['candidate_cursor'] if recovery else 0
    accepted_events = recovery['accepted_events'] if recovery else []
    if recovery:
        from qwen_head_opsd import atomic_json
        atomic_json(out / 'train_samples.json', accepted_events)
        print(f"[weights-only continuation] accepted={global_sample} cursor={attempted} "
              f"historical_steps={step}; AdamW/RNG RESET; new output={out}", flush=True)
    with (out / "train.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(args.epochs):
            order = list(samples)
            if not online_gate:
                random.Random(args.seed + epoch).shuffle(order)
            optimizer.zero_grad(set_to_none=True)
            epoch_rkl, epoch_raw_rkl, epoch_iou, epoch_parse_failures = 0.0, 0.0, 0.0, 0
            epoch_time_ce, epoch_total_loss = 0.0, 0.0
            valid_in_group, epoch_timestamp_skips, epoch_supervised_samples = 0, 0, 0
            start_cursor = recovery['candidate_cursor'] if recovery else 0
            for i in range(start_cursor, len(order)):
                sample = order[i]
                if online_gate and (global_sample >= args.max_samples or
                        (args.max_attempts and attempted >= args.max_attempts)):
                    break
                if online_gate:
                    attempted += 1
                for device in range(n_gpus):
                    torch.cuda.reset_peak_memory_stats(device)
                try:
                    pair = prepare_pair(processor, sample, args)
                except Exception as error:
                    if not online_gate:
                        raise
                    from qwen_frame_hints import NoEvidenceFrame
                    if not isinstance(error, NoEvidenceFrame):
                        raise
                    from qwen_head_opsd import append_json
                    append_json(out / 'gate.jsonl', dict(attempt=attempted, **sample,
                                accepted=False, reason='unusable_frame_hint', error=str(error)))
                    continue
                # Text teacher shares the original video; evidence teacher has an extra clip.
                student = {k: v.to(input_device) if hasattr(v, "to") else v for k, v in pair[0].items()}
                teacher = {k: (student[k] if args.teacher_mode == "gt_text" and
                            k in ("pixel_values_videos", "video_grid_thw") else
                            v.to(input_device) if hasattr(v, "to") else v) for k, v in pair[1].items()}
                del pair
                if protocol:
                    from qwen_head_opsd import validate_context
                    validate_context(model, student, args)
                    validate_context(model, teacher, args)
                if online_gate:
                    from qwen_head_opsd import evaluate_gate, append_json
                    gate_record = evaluate_gate(model, processor, student, teacher, sample, args, adapters)
                    gate_record.update(attempt=attempted, **sample,
                                       video_processing=args._last_video_audit)
                    append_json(out / 'gate.jsonl', gate_record)
                    print(f"[gate] attempt={attempted} trained={global_sample} "
                          f"student_IoU={gate_record['student']['rollout_iou']:.4f} "
                          f"teacher_IoU={gate_record['teacher']['rollout_iou']:.4f} "
                          f"accepted={gate_record['accepted']} reason={gate_record['reason']}", flush=True)
                    if not gate_record['accepted']:
                        del student, teacher
                        continue
                model.eval()
                reset_rope(model)
                with torch.no_grad():
                    generated = model.generate(**student, do_sample=True, temperature=args.temperature,
                        top_p=args.top_p, top_k=0, max_new_tokens=args.max_new_tokens, use_cache=True,
                        pad_token_id=processor.tokenizer.pad_token_id)
                y = generated[:, student["input_ids"].shape[1]:].detach()
                if y.shape[1] == 0:
                    raise RuntimeError("学生生成空 token 序列")
                slot = global_sample if online_gate else i
                slot_total = args.max_samples if online_gate else len(order)
                group_start = (slot // args.grad_accum) * args.grad_accum
                group_size = min(args.grad_accum, slot_total - group_start)
                record = dict(epoch=epoch + 1, sample=i + 1,
                              tokens=y.shape[1], video_path=sample["video_path"], query=sample["query"],
                              gt=[sample["gt_start"], sample["gt_end"]],
                              rollout=processor.tokenizer.decode(y[0].tolist(), skip_special_tokens=True))
                if online_gate:
                    record.update(attempt=attempted, accepted_index=global_sample + 1,
                                  gate=gate_record)
                if getattr(args, 'model_family', 'qwen3vl') == 'qwen25vl' or protocol:
                    record['video_processing'] = args._last_video_audit
                selected, skip_reason = None, None
                if timestamp_only:
                    try:
                        selected = timestamp_selection(processor.tokenizer, y[0].tolist())
                    except ValueError as error:
                        if args.timestamp_invalid_policy == "error":
                            raise ValueError(f"sample={i+1} timestamp OPSD: {error}") from error
                        skip_reason = str(error)
                    record.update(opsd_token_scope=args.opsd_token_scope,
                                  opsd_skipped=skip_reason is not None,
                                  timestamp_skip_reason=skip_reason,
                                  timestamp_selection=selected)
                if skip_reason is not None:
                    epoch_timestamp_skips += 1
                    record.update(loss=0.0, raw_rkl=0.0, max_token_rkl=0.0,
                                  clipped_token_fraction=0.0, opsd_tokens=0)
                else:
                    s_inputs, s_pos = scoring_inputs(student, y)
                    t_inputs, t_pos = scoring_inputs(teacher, y)
                    reset_rope(model)
                    with torch.no_grad(), self_teacher_context(args, adapters):
                        t_logits = model(**t_inputs, logits_to_keep=t_pos, use_cache=False).logits.detach()
                    with scoring_memory_context(args):
                        model.train()
                        reset_rope(model)
                        s_logits = model(**s_inputs, logits_to_keep=s_pos, use_cache=False).logits
                        if s_logits.shape != t_logits.shape or s_logits.shape[1] != y.shape[1]:
                            raise RuntimeError("师生 logits 未按同一学生前缀对齐")
                        # Index the matching causal prediction rows, never the sampled
                        # token IDs as vocabulary columns. Preserve plain-slice scoring.
                        if timestamp_only:
                            s_logits = select_prediction_rows(s_logits, selected["indices"])
                            t_logits = select_prediction_rows(t_logits, selected["indices"])
                        token_rkl = reverse_kl_per_token(s_logits, t_logits)
                        raw_rkl = token_rkl.mean()
                        clipped_rkl = (token_rkl.clamp(max=args.kl_clip)
                                       if args.kl_clip > 0 else token_rkl)
                        loss = clipped_rkl.mean()
                        if not torch.isfinite(loss):
                            raise RuntimeError("非有限 OPSD loss")
                        record.update(loss=float(loss.detach()), raw_rkl=float(raw_rkl.detach()),
                                      max_token_rkl=float(token_rkl.max().detach()),
                                      clipped_token_fraction=(float((token_rkl > args.kl_clip).float().mean())
                                                              if args.kl_clip > 0 else 0.0))
                        if timestamp_only:
                            record["opsd_tokens"] = len(selected["indices"])
                        (loss / group_size).backward()
                    valid_in_group += 1
                    epoch_supervised_samples += 1
                    # Release the OPSD graph before the additional teacher-forced CE pass.
                    del s_logits, t_logits, token_rkl, raw_rkl, clipped_rkl, loss
                    del s_inputs, t_inputs
                if args.time_ce_weight > 0:
                    from torch.nn import functional as F
                    ids, start_mask, end_mask = timestamp_answer_tokens(
                        processor.tokenizer, sample)
                    target = torch.tensor([ids], dtype=torch.long, device=input_device)
                    ce_inputs, ce_pos = scoring_inputs(student, target)
                    with scoring_memory_context(args):
                        reset_rope(model)
                        ce_logits = model(**ce_inputs, logits_to_keep=ce_pos,
                                          use_cache=False).logits
                        if ce_logits.shape[:2] != target.shape:
                            raise RuntimeError("GT answer logits and token IDs are not aligned")
                        target_on_logits = target.to(ce_logits.device)
                        start_mask_tensor = torch.tensor(start_mask, dtype=torch.bool,
                                                         device=ce_logits.device)
                        end_mask_tensor = torch.tensor(end_mask, dtype=torch.bool,
                                                       device=ce_logits.device)
                        start_ce = F.cross_entropy(
                            ce_logits[:, start_mask_tensor, :].float().reshape(-1, ce_logits.shape[-1]),
                            target_on_logits[:, start_mask_tensor].reshape(-1))
                        end_ce = F.cross_entropy(
                            ce_logits[:, end_mask_tensor, :].float().reshape(-1, ce_logits.shape[-1]),
                            target_on_logits[:, end_mask_tensor].reshape(-1))
                        time_ce = 0.5 * (start_ce + end_ce)
                        if not torch.isfinite(time_ce):
                            raise RuntimeError("非有限 GT timestamp CE loss")
                        weighted_time_ce = args.time_ce_weight * time_ce
                        (weighted_time_ce / group_size).backward()
                    record.update(rkl_loss=record["loss"],
                                  time_ce=float(time_ce.detach()),
                                  time_ce_start=float(start_ce.detach()),
                                  time_ce_end=float(end_ce.detach()),
                                  weighted_time_ce=float(weighted_time_ce.detach()),
                                  gt_time_tokens=int(sum(start_mask) + sum(end_mask)))
                    record["loss"] += record["weighted_time_ce"]
                    del ce_logits, ce_inputs, target, target_on_logits
                    del start_ce, end_ce, time_ce, weighted_time_ce
                record.update(rollout_metrics(record["rollout"], sample, args.prompt_mode))
                epoch_rkl += record.get("rkl_loss", record["loss"])
                epoch_raw_rkl += record["raw_rkl"]
                epoch_time_ce += record.get("time_ce", 0.0)
                epoch_total_loss += record["loss"]
                epoch_iou += record["rollout_iou"]
                epoch_parse_failures += int(record["parse_failure"])
                if (slot + 1) % args.grad_accum == 0 or slot + 1 == slot_total:
                    scale = accumulation_scale(group_size, valid_in_group) if timestamp_only else 1.0
                    if scale is not None:
                        if scale != 1.0:
                            for parameter in params:
                                if parameter.grad is not None:
                                    parameter.grad.mul_(scale)
                        norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm, error_if_nonfinite=True)
                        if protocol and (not torch.isfinite(norm) or float(norm) <= 0):
                            raise RuntimeError('OPSD adapter gradient is zero or non-finite; training stopped')
                        optimizer.step()
                        step += 1
                        record.update(optimizer_step=step, grad_norm=float(norm))
                    else:
                        record["optimizer_group_skipped"] = True
                    optimizer.zero_grad(set_to_none=True)
                    valid_in_group = 0
                global_sample += 1
                if online_gate:
                    from qwen_head_opsd import atomic_json
                    accepted_events.append(dict(sample, accepted_index=global_sample, attempt=attempted))
                    atomic_json(out / 'train_samples.json', accepted_events)
                if args.save_every and global_sample % args.save_every == 0:
                    checkpoint = out / f"sample-{global_sample:06d}"
                    save_adapter(checkpoint, adapters, config)
                    (checkpoint / "summary.json").write_text(json.dumps(dict(
                        epoch=epoch + 1, sample=i + 1, global_sample=global_sample,
                        optimizer_steps=step), indent=2), encoding="utf-8")
                    print(f"[checkpoint] {checkpoint}", flush=True)
                record["peak_gpu_gib"] = max(
                    torch.cuda.max_memory_allocated(device) for device in range(n_gpus)) / 1024**3
                record['peak_gpu_gib_by_device'] = [
                    torch.cuda.max_memory_allocated(device) / 1024**3 for device in range(n_gpus)]
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                ce_status = (f"time_CE={record['time_ce']:.6f} total={record['loss']:.6f} "
                             if args.time_ce_weight > 0 else "")
                if timestamp_only:
                    ce_status += (f"timestamp_tokens={record['opsd_tokens']} "
                                  f"skipped={record['opsd_skipped']} ")
                print(f"epoch={epoch+1} sample={i+1}/{len(order)} step={step} "
                      f"RKL={record.get('rkl_loss', record['loss']):.6f} "
                      f"raw_RKL={record['raw_rkl']:.6f} {ce_status}"
                      f"rollout_IoU={record['rollout_iou']:.4f} tokens={y.shape[1]} "
                      f"peak={record['peak_gpu_gib']:.2f}GiB", flush=True)
                del student, teacher, generated, y
            if online_gate and global_sample != args.max_samples:
                from qwen_head_opsd import atomic_json
                save_adapter(out / 'incomplete-adapter', adapters, config)
                atomic_json(out / 'summary.json', dict(training_complete=False, target_reached=False,
                    merged=False, attempted=attempted, accepted=global_sample,
                    optimizer_steps=step, accumulation_pending=global_sample % args.grad_accum))
                raise RuntimeError(f'OPSD gate exhausted pool/attempt cap: trained {global_sample}/{args.max_samples}; no merged model exported')
            save_adapter(out / f"epoch-{epoch+1}", adapters, config)
            denominator = epoch_supervised_samples if online_gate else len(order)
            summary = dict(epoch=epoch+1, mean_rkl=epoch_rkl / denominator,
                           mean_raw_rkl=epoch_raw_rkl / denominator,
                           train_rollout_miou=epoch_iou / denominator,
                           parse_failures=epoch_parse_failures, optimizer_steps=step,
                           note="online sampled training rollouts, not held-out evaluation")
            if args.time_ce_weight > 0:
                summary.update(mean_time_ce=epoch_time_ce / len(order),
                               mean_total_loss=epoch_total_loss / len(order))
            if timestamp_only:
                summary.update(timestamp_skipped_samples=epoch_timestamp_skips,
                               supervised_samples=epoch_supervised_samples,
                               mean_rkl_over_supervised_samples=(epoch_rkl / epoch_supervised_samples
                                                                if epoch_supervised_samples else None),
                               mean_rkl_denominator="all selected samples; skipped samples contribute zero",
                               timestamp_skip_rate=epoch_timestamp_skips / len(order))
            (out / f"epoch-{epoch+1}" / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary), flush=True)
            if timestamp_only and epoch_supervised_samples == 0:
                raise RuntimeError("本轮没有可监督时间戳，不导出可评测合并模型；检查train.jsonl跳过原因")
    if not args.skip_merge:
        for name, adapter in adapters.items():
            adapter.merge()
            layer, target = name.split(".")
            setattr(model.model.language_model.layers[int(layer)].self_attn, target, adapter.base)
        model.config.use_cache = True
        # Save the complete merged Hugging Face checkpoint directly at
        # --output-dir, matching the existing checkpoints layout. Training
        # metadata and epoch adapter snapshots can coexist with HF model files.
        model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
        processor.save_pretrained(out)
    if protocol:
        from qwen_head_opsd import atomic_json
        atomic_json(out / 'summary.json', dict(
            method=protocol['method'], training_complete=True,
            target_reached=global_sample == args.max_samples * args.epochs,
            processed_samples=global_sample, unique_selected_events=len(accepted_events) if online_gate else len(samples),
            attempted=attempted if online_gate else global_sample, accepted=global_sample,
            optimizer_steps=step, merged=not args.skip_merge,
            optimizer_steps_this_session=step - initial_step,
            weights_only_continuation=config.get('weights_only_continuation'),
            teacher_policy=protocol['teacher_policy'],
            note='Training evidence only; held-out evaluation is separate'))
    print(f"完成 {step} 次优化，输出 {out}", flush=True)


