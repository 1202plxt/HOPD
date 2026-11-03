"""Two-K100 synchronized TimeLens-8B head OPSD; exact GT text only in teacher."""
import argparse
import datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_smoke_heads(protocol_path, model_path):
    path = Path(protocol_path)
    protocol = json.loads(path.read_text(encoding='utf-8'))
    a = protocol['arguments']
    if (protocol['method'] != 'timelens8b_gt_only_answer_iou_rank_v1'
            or Path(a['model_path']).resolve() != Path(model_path).resolve()
            or a['total_tokens'] != 14336 or a['min_tokens'] != 64
            or a['fps'] != 2 or a['max_frames'] != 0):
        raise ValueError('冒烟候选 protocol 的模型或预算不匹配')
    rows = protocol['heads']
    if len(rows) != 30:
        raise ValueError('候选名单不是 30 个 head')
    heads = [{'layer': int(h['layer']), 'head': int(h['head'])} for h in rows[:2]]
    if len({(h['layer'], h['head']) for h in heads}) != 2:
        raise ValueError('候选 head 重复')
    return heads


def read_heads(summary_path, model_path):
    path = Path(summary_path)
    data = json.loads(path.read_text(encoding='utf-8'))
    provenance = json.loads((path.parent / 'merge_protocol.json').read_text(encoding='utf-8'))
    workers = provenance['worker_protocols']
    if (data.get('complete') is not True or data.get('target') != 300
            or data.get('n_paired') != 300 or len(workers) not in (2, 4)):
        raise ValueError('需要完整的 300 样本、2 或 4 worker GT mask 合并结果')
    for w in workers:
        a = w['arguments']
        if (w['method'] != 'timelens8b_gt_only_answer_iou_rank_v1'
                or Path(a['model_path']).resolve() != Path(model_path).resolve()
                or a['total_tokens'] != 14336 or a['min_tokens'] != 64
                or a['fps'] != 2 or a['max_frames'] != 0 or len(w['heads']) != 30
                or not w.get('budget_policy', '').startswith('strict actual visual token ceiling')):
            raise ValueError('分析模型、预算、方法或候选 head 不匹配')
    if {w['arguments']['shard_index'] for w in workers} != set(range(len(workers))):
        raise ValueError('分析 worker 分片不完整')
    ranking = data['ranking']
    if len(ranking) != 30:
        raise ValueError('分析排名必须包含 30 个 head')
    base = float(data['plain']['mIoU'])
    found, seen = [], set()
    for row in ranking:
        head = row['head']
        key = (int(head['layer']), int(head['head']))
        score = float(row['selected_gt_damage_pp'])
        actual = 100 * (base - float(row['metrics']['selected_gt']['mIoU']))
        if key in seen or not math.isfinite(score) or abs(score - actual) > 1e-7:
            raise ValueError('head 重复或 GT mask 损害分数不一致')
        seen.add(key)
        if score > 0:
            found.append({'layer': key[0], 'head': key[1], 'diagnostic_score_pp': score})
    found.sort(key=lambda h: (-h['diagnostic_score_pp'], h['layer'], h['head']))
    if not found:
        raise ValueError('没有移除 GT 后 mIoU 下降的 head；停止训练')
    return found[:10]


def digest_params(params):
    digest = hashlib.sha256()
    for name, p in params:
        digest.update(name.encode())
        digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('model-path', 'anno-json', 'video-dir', 'output-dir'):
        p.add_argument('--' + arg, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--analysis-summary')
    group.add_argument('--smoke-protocol')
    p.add_argument('--rank-id', type=int, required=True, choices=(0, 1))
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--max-samples', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--alpha', type=float, default=16.)
    p.add_argument('--max-new-tokens', type=int, default=80)
    p.add_argument('--skip-merge', action='store_true')
    args = p.parse_args()
    if args.max_samples < 4 or args.max_samples % 4:
        p.error('--max-samples 必须为 4 的正整数倍')
    if args.smoke_protocol and (not args.skip_merge or args.max_samples != 4):
        p.error('--smoke-protocol 仅支持 4 样本 --skip-merge 冒烟')
    heads = read_smoke_heads(args.smoke_protocol, args.model_path) if args.smoke_protocol else read_heads(args.analysis_summary, args.model_path)
    import torch
    import torch.distributed as dist
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import train_timelens_three_group_opsd as old
    from timelens8b_strict_video_budget import strict_video_budget

    if torch.version.hip is None or torch.cuda.device_count() != 1:
        raise RuntimeError('每个 worker 必须通过 HIP_VISIBLE_DEVICES 看到一张 K100')
    torch.cuda.set_device(0)
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{args.port}',
                            rank=args.rank_id, world_size=2, timeout=datetime.timedelta(minutes=20))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        samples, stats = old.load_samples(args.anno_json, args.video_dir, args.max_samples, args.seed)
        random.Random(args.seed).shuffle(samples)
        sample_hash = hashlib.sha256(json.dumps(samples, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        setup = (heads, sample_hash, sha(args.smoke_protocol or args.analysis_summary),
                 None if args.smoke_protocol else sha(Path(args.analysis_summary).parent / 'merge_protocol.json'))
        gathered = [None, None]
        dist.all_gather_object(gathered, setup)
        if gathered[0] != gathered[1]:
            raise RuntimeError('两个 worker 的 head 或训练数据不一致')
        torch.manual_seed(args.seed)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path, local_files_only=True, trust_remote_code=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 0}).eval()
        if 'logits_to_keep' not in inspect.signature(model.forward).parameters:
            raise RuntimeError('当前 TimeLens 模型 forward 不支持 logits_to_keep；不要直接启动正式训练')
        processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True,
            trust_remote_code=True, padding_side='left', do_resize=False)
        if model.config.model_type != 'qwen3_vl':
            raise RuntimeError('TimeLens-8B 需要 Qwen3-VL 架构')
        processor.tokenizer.padding_side = 'left'
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        torch.manual_seed(args.seed)
        adapters = old.attach_head_lora(model, heads, args.rank, args.alpha, ['q_proj', 'o_proj'])
        params = sorted((name, value) for name, value in model.named_parameters() if value.requires_grad)
        if {id(x) for _, x in params} != {id(x) for a in adapters.values() for x in (a.A, a.B)}:
            raise RuntimeError('存在非所选 head 的可训练参数')
        hashes = [None, None]
        dist.all_gather_object(hashes, digest_params(params))
        if hashes[0] != hashes[1]:
            raise RuntimeError('两卡初始 adapter 不一致')
        initial_adapter_sha256 = hashes[0]
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.config.use_cache = False
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout): module.p = 0.0
            if hasattr(module, 'attention_dropout'): module.attention_dropout = 0.0
        optimizer = torch.optim.AdamW([x for _, x in params], lr=args.lr, weight_decay=0.0)
        config = dict(method='timelens8b_gtmask_positive_upto10_gttext_opsd_k100_dp2',
            arguments=vars(args), selected_heads=heads, samples_sha256=sample_hash, data_stats=stats,
            analysis_sha256=setup[2], analysis_merge_protocol_sha256=setup[3],
            source_sha256={str(q.relative_to(ROOT)): sha(q) for q in [Path(__file__).resolve(),
                ROOT/'scripts/train_timelens_three_group_opsd.py', ROOT/'scripts/qwen_decode_worker.py',
                ROOT/'scripts/qwen_head_opsd.py', ROOT/'scripts/timelens8b_strict_video_budget.py']},
            teacher='current model with current head adapter; stop gradient; GT timestamp text appended',
            student='current selected-head Q/O adapter, video and query only',
            loss='full-vocabulary KL(student || current stop-gradient GT-privileged teacher) on student sampled rollout',
            gate='none', head_selection=('smoke candidates only' if args.smoke_protocol else 'completed GT mask ranking'), total_samples=args.max_samples, local_samples=args.max_samples//2,
            local_accumulation=2, global_batch=4, trainable_parameters=sum(x.numel() for _, x in params),
            gradient_sync='CPU Gloo all-reduce mean on every optimizer step',
            video_budget=dict(fps=2,min_tokens=64,total_tokens=14336,max_frames=0,
                policy='strict cap via uniform thinning after decode with original timestamps'),
            runtime=dict(torch=torch.__version__,transformers=transformers.__version__,hip=torch.version.hip))
        if args.rank_id == 0:
            if (out/'training_metadata.json').exists(): raise FileExistsError('请使用新输出目录，不支持断点恢复')
            (out/'training_metadata.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
            (out/'train_samples.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
        dist.barrier()
        torch.manual_seed(args.seed + args.rank_id + 1)
        local = samples[args.rank_id::2]
        video_args = argparse.Namespace(prompt_mode='official',fps=2.,min_tokens=64,total_tokens=14336,max_frames=0, teacher_mode='gt_text',model_family='qwen3vl',video_processor_mode='legacy',_head_opsd_protocol={'smoke_or_train':True})
        optimizer.zero_grad(set_to_none=True)
        step = 0
        with (out/f'train_rank{args.rank_id}.jsonl').open('w', encoding='utf-8') as log:
            for i, sample in enumerate(local):
                started = time.monotonic()
                with strict_video_budget(processor, 14336) as budget_audit:
                    pair = old.prepare_pair(processor, sample, video_args)
                video_id = processor.tokenizer.convert_tokens_to_ids('<|video_pad|>')
                actual_visual = int((pair[0]['input_ids'] == video_id).sum())
                if actual_visual != budget_audit['estimated_visual_tokens'] or actual_visual > 14336:
                    raise RuntimeError('actual visual token count differs from strict budget calculation')
                student = {k: v.to('cuda:0') if hasattr(v, 'to') else v for k,v in pair[0].items()}
                teacher = {k: (student[k] if k in ('pixel_values_videos','video_grid_thw') else
                           v.to('cuda:0') if hasattr(v,'to') else v) for k,v in pair[1].items()}
                del pair
                model.eval(); old.reset_rope(model)
                with torch.no_grad():
                    generated = model.generate(**student, do_sample=True, temperature=1., top_p=.9, top_k=0,
                        max_new_tokens=args.max_new_tokens, use_cache=True,
                        pad_token_id=processor.tokenizer.pad_token_id)
                y = generated[:, student['input_ids'].shape[1]:].detach()
                if y.shape[1] == 0: raise RuntimeError('学生生成空回答')
                si, sp = old.scoring_inputs(student, y)
                ti, tp = old.scoring_inputs(teacher, y)
                old.reset_rope(model)
                with torch.no_grad():
                    teacher_logits = model(**ti, logits_to_keep=tp, use_cache=False).logits.detach()
                model.train(); old.reset_rope(model)
                student_logits = model(**si, logits_to_keep=sp, use_cache=False).logits
                if teacher_logits.shape != student_logits.shape or student_logits.shape[1] != y.shape[1]:
                    raise RuntimeError('师生 logits 未对齐学生回答')
                loss = old.reverse_kl_per_token(student_logits, teacher_logits).mean()
                if not torch.isfinite(loss): raise RuntimeError('loss 非有限')
                (loss / 2).backward()
                if (i + 1) % 2 == 0:
                    for name, param in params:
                        # All workers issue the same collectives even for missing gradients.
                        valid = torch.tensor([int(param.grad is not None and torch.isfinite(param.grad).all())], dtype=torch.int32)
                        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
                        if not valid.item(): raise RuntimeError(f'缺失或非有限梯度: {name}')
                        grad = param.grad.detach().cpu()
                        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                        param.grad.copy_(grad.to(param.device).div_(2))
                    norm = torch.nn.utils.clip_grad_norm_([x for _,x in params], 1., error_if_nonfinite=True)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
                rec = dict(rank=args.rank_id, local_sample=i+1, local_total=len(local), step=step,
                    loss=float(loss.detach()), tokens=int(y.shape[1]), seconds=time.monotonic()-started,
                    peak_gib=torch.cuda.max_memory_allocated()/1024**3,
                    actual_visual_tokens=actual_visual, strict_budget=budget_audit)
                log.write(json.dumps(rec)+'\n'); log.flush()
                print(json.dumps(rec), flush=True)
                del student, teacher, si, ti, generated, y, student_logits, teacher_logits, loss
        hashes = [None, None]
        dist.all_gather_object(hashes, digest_params(params))
        if hashes[0] != hashes[1]: raise RuntimeError('最终两卡 adapter 不一致，停止保存')
        if hashes[0] == initial_adapter_sha256:
            raise RuntimeError('优化后 adapter 没有变化；不能认定训练冒烟通过')
        dist.barrier()
        if args.rank_id == 0:
            if not args.skip_merge:
                for name, adapter in adapters.items():
                    adapter.merge()
                    layer, target = name.split('.')
                    setattr(model.model.language_model.layers[int(layer)].self_attn, target, adapter.base)
                model.config.use_cache = True
                model.save_pretrained(out, safe_serialization=True, max_shard_size='5GB')
                processor.save_pretrained(out)
            result = dict(training_complete=True, processed_samples=args.max_samples,
                optimizer_steps=step, merged=not args.skip_merge, selected_heads=heads,
                initial_adapter_sha256=initial_adapter_sha256, final_adapter_sha256=hashes[0],
                note='训练结果；测评需要单独执行')
            (out/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    finally:
        dist.destroy_process_group()


