"""One-pass frozen-32B distillation, one GPU group or synchronized groups."""
import argparse
from collections import Counter
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import train_timelens_three_group_opsd as modeling
import probe_qwen32b_privileged_k100 as video
from train_qwen3_boundary_selective_opd import boundary_decision, boundary_token_groups, reverse_kl, selected_indices

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('student-path', 'teacher-path', 'anno-json', 'video-dir', 'output-dir'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--heads-json', default=str(ROOT/'heads/qwen3_gtmask_top10.json'))
    p.add_argument('--boundary-selective', action='store_true',
                   help='Independent boundary-error gate and selected-number-token KL; otherwise whole-answer IoU-gated baseline')
    p.add_argument('--boundary-epsilon', type=float, default=0., help='Strict improvement margin in seconds')
    for name, default in (('gpus-per-worker', 2), ('max-samples', 1000), ('max-attempts', 10000),
                          ('grad-accum', 4), ('rank', 8), ('total-tokens', 14336), ('min-tokens', 64),
                          ('max-frames', 0), ('pool-factor', 2), ('max-pool-frames', 1536),
                          ('clip-total-tokens', 4096), ('clip-max-frames', 128),
                          ('max-new-tokens', 80), ('save-every-steps', 25), ('seed', 42)):
        p.add_argument('--'+name, type=int, default=default)
    for name, default in (('lr', 2e-5), ('alpha', 16.), ('fps', 2.), ('clip-fps', 4.),
                          ('clip-padding-seconds', 0.), ('student-reserve-gib', 12.),
                          ('teacher-reserve-gib', 4.), ('max-grad-norm', 1.), ('decode-timeout', 600.)):
        p.add_argument('--'+name, type=float, default=default)
    p.add_argument('--offload-saved-tensors', action='store_true')
    p.add_argument('--skip-merge', action='store_true')
    p.add_argument('--plan-only', action='store_true')
    return p


def group_devices(visible, rank, world, size):
    ids = [x.strip() for x in visible.split(',')]
    if size < 2 or not 0 <= rank < world or len(ids) != world*size or len(set(ids)) != len(ids) or not all(ids):
        raise ValueError('Visible GPU IDs must form disjoint groups of gpus-per-worker (at least two per group)')
    return ','.join(ids[rank*size:(rank+1)*size])


def wave_size(workers, batch, pending, target, accepted, remaining):
    return min(workers, batch-pending, target-accepted, remaining)


def prepare(a):
    for name in ('lr', 'alpha', 'fps', 'clip_fps', 'max_grad_norm', 'decode_timeout'):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
            raise ValueError('Require finite positive '+name)
    for name in ('boundary_epsilon', 'clip_padding_seconds', 'student_reserve_gib', 'teacher_reserve_gib'):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) < 0:
            raise ValueError('Require finite nonnegative '+name)
    if min(a.max_samples, a.grad_accum, a.rank, a.total_tokens, a.min_tokens, a.max_new_tokens) < 1:
        raise ValueError('Sample, token, rank and accumulation counts must be positive')
    if (a.max_attempts < 0 or a.save_every_steps < 0 or a.gpus_per_worker < 2 or
            a.max_frames < 0 or a.max_frames % 2 or a.min_tokens > a.total_tokens or
            a.pool_factor < 2 or a.max_pool_frames < 4 or a.max_pool_frames % 2 or
            not 0 < a.clip_total_tokens < a.total_tokens or a.clip_max_frames < 4 or a.clip_max_frames % 2):
        raise ValueError('Invalid GPU, visual budget, candidate or checkpoint settings')
    if 'train' not in {x.lower() for x in Path(a.anno_json).parts}:
        raise ValueError('Use annotations under a Train directory')
    out = Path(a.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new empty output directory')
    student, teacher = video.checkpoint_identity(a.student_path), video.checkpoint_identity(a.teacher_path)
    if student['path'] == teacher['path']:
        raise ValueError('Require an independent frozen 32B teacher, not self-distillation')
    for base in (Path(student['path']), Path(teacher['path'])):
        if out == base or base in out.parents:
            raise ValueError('Output overlaps a source model')
    sc, tc = student['config']['text_config'], teacher['config']['text_config']
    if (sc['num_hidden_layers'], sc['num_attention_heads'], sc['hidden_size']) != (36, 32, 4096):
        raise ValueError('Bundled heads require the Qwen3-VL-8B layout')
    if tc['hidden_size'] <= sc['hidden_size'] or tc['num_hidden_layers'] < sc['num_hidden_layers']:
        raise ValueError('Teacher must be the larger Qwen3-VL checkpoint')
    raw = video.read(a.heads_json)
    heads = raw['heads'] if isinstance(raw, dict) else raw
    if not isinstance(heads, list) or not heads:
        raise ValueError('Require nonempty preselected heads')
    coordinates = []
    for h in heads:
        if type(h.get('layer')) is not int or type(h.get('head')) is not int:
            raise ValueError('Head coordinates must be integers')
        pair = h['layer'], h['head']
        if not 0 <= pair[0] < 36 or not 0 <= pair[1] < 32 or pair in coordinates:
            raise ValueError('Duplicate or out-of-range head')
        coordinates.append(pair)
    heads = [dict(layer=l, head=h) for l, h in coordinates]
    counts = Counter(h['layer'] for h in heads)
    dim = sc.get('head_dim', sc['hidden_size']//sc['num_attention_heads'])
    samples, stats = modeling.load_samples(a.anno_json, a.video_dir, 0, a.seed)
    if a.max_attempts:
        samples = samples[:a.max_attempts]
    if len(samples) < a.max_samples:
        raise ValueError('Candidate pool smaller than accepted target')
    return dict(method='boundary_selective_opd' if a.boundary_selective else 'whole_answer_opd_baseline',
        arguments=vars(a), student_checkpoint=student, teacher_checkpoint=teacher,
        selected_heads=heads, heads_sha256=video.sha(a.heads_json), data_stats=stats,
        candidate_count=len(samples), annotation_sha256=video.sha(a.anno_json),
        candidate_samples_sha256=hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest(),
        expected_trainable_parameters=2*a.rank*sum(sc['hidden_size']+n*dim for n in counts.values()),
        boundary_policy=dict(epsilon_seconds=a.boundary_epsilon, overall_iou_gate=False,
            rule='student absolute error minus teacher absolute error > epsilon independently per boundary',
            invalid='Skip invalid prediction, unsupported format or unreliable token mapping'),
        objective=('Mean full-vocabulary KL(student || stopgrad teacher) over selected numeric tokens per example; average examples in effective batch'
            if a.boundary_selective else 'Mean full-vocabulary KL over the entire student continuation; average examples in effective batch'),
        active_gate='Independent boundary absolute errors, no IoU gate' if a.boundary_selective else 'Teacher valid and teacher IoU > student IoU',
        baseline='Without --boundary-selective: valid teacher IoU strictly greater than sampled student IoU (invalid student IoU=0); full continuation KL',
        teacher_policy='Independent pretrained 32B, permanently frozen; full-source context plus original-timeline GT clip; no GT answer text',
        student_policy='Normal full-source video/query; sampled continuation reused for gate and scoring',
        visual_budget='Global source context and reference clip share the total cap; normal student has equal actual visual-token count',
        source_sha256={str(p.relative_to(ROOT)): video.sha(p) for p in sorted(ROOT.rglob('*.py'))},
        resume='Weights-only snapshots; no optimizer/RNG resume'), samples


def candidate_gradient(a, student, teacher, processors, sample):
    import torch
    si, ti, sa, ta, duration = video.training_inputs(processors, sample, a)
    y, sg = video.generate(student, processors[0], si, sample, duration, a, sampled=True)
    _, tg = video.generate(teacher, processors[1], ti, sample, duration, a, sampled=False)
    decision = groups = None
    indices = []
    if a.boundary_selective:
        decision, reason = boundary_decision(sg, tg, [sample['gt_start'], sample['gt_end']], a.boundary_epsilon)
        active = bool(decision and (decision['start'] or decision['end']))
        if active:
            try:
                groups = boundary_token_groups(processors[0].tokenizer, y[0].tolist())
                indices = selected_indices(decision, groups)
            except ValueError as error:
                active, reason = False, 'timestamp_tokens_'+str(error)
    else:
        active = tg['valid'] and tg['iou'] > sg['iou']
        reason = 'teacher_improves_iou' if active else 'teacher_not_better_iou'
        if active:
            indices = list(range(y.shape[1]))
    record = dict(sample=sample, accepted=active, reason=reason, student=sg, teacher=tg,
        boundary=decision, boundary_token_groups=groups, selected_token_indices=indices,
        student_video=sa, teacher_video=ta)
    if active:
        with torch.no_grad():
            teacher_logits = video.scoring_logits(teacher, ti, y)
            log_q = teacher_logits.float().log_softmax(-1).cpu()
            del teacher_logits
        del ti
        student.train()
        context = torch.autograd.graph.save_on_cpu(pin_memory=True) if a.offload_saved_tensors else contextlib.nullcontext()
        with context:
            logits = video.scoring_logits(student, si, y)
            loss = reverse_kl(logits, log_q, indices)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite reverse KL')
        loss.backward()
        record['loss'] = float(loss.detach())
    return record


def main(argv=None):
    a = parser().parse_args(argv)
    rank, local = int(os.environ.get('RANK', '0')), int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if rank != local:
        raise ValueError('Single-node training only')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(world*a.gpus_per_worker))))
    os.environ['CUDA_VISIBLE_DEVICES'] = group_devices(visible, local, world, a.gpus_per_worker)
    protocol, samples = prepare(a)
    protocol['distributed'] = dict(workers=world, gpus_per_worker=a.gpus_per_worker,
        global_effective_batch=a.grad_accum, reduction='CPU float32 sum / global accepted count')
    if a.plan_only:
        if rank == 0:
            print(json.dumps(protocol, ensure_ascii=False, indent=2), flush=True)
        return 0
    import datetime
    import torch
    import torch.distributed as dist
    import qwen_decode_worker as decoder
    if torch.cuda.device_count() != a.gpus_per_worker:
        raise RuntimeError('Unexpected visible GPU count')
    if world > 1:
        dist.init_process_group('gloo', timeout=datetime.timedelta(minutes=60))
    def gather(value):
        if world == 1:
            return [value]
        values = [None]*world
        dist.all_gather_object(values, value)
        return values
    def barrier():
        if world > 1:
            dist.barrier()
    out = Path(a.output_dir)
    accepted = processed = steps = pending = 0
    try:
        if any(p != protocol for p in gather(protocol)):
            raise RuntimeError('Worker protocol/candidate mismatch')
        if rank == 0:
            out.mkdir(parents=True, exist_ok=True)
            video.write(out/'training_metadata.json', protocol)
            video.write(out/'candidate_samples.json', samples)
        barrier()
        student, teacher, processors = video.load_runtime(a)
        video.tokenizer_equivalence(processors[0].tokenizer, processors[1].tokenizer)
        if student.config.text_config.vocab_size != teacher.config.text_config.vocab_size:
            raise ValueError('Output vocabulary dimensions differ')
        adapters = modeling.attach_head_lora(student, protocol['selected_heads'], a.rank, a.alpha, ['q_proj', 'o_proj'])
        params = sorted((name, p) for name, p in student.named_parameters() if p.requires_grad)
        if ({id(p) for _, p in params} != {id(p) for m in adapters.values() for p in (m.A, m.B)}
                or sum(p.numel() for _, p in params) != protocol['expected_trainable_parameters']):
            raise RuntimeError('Unexpected trainable parameters')
        with torch.no_grad():
            for _, p in params:
                value = p.detach().cpu().clone()
                if world > 1:
                    dist.broadcast(value, src=0)
                p.copy_(value.to(p.device))
        torch.manual_seed(a.seed+rank)
        random.seed(a.seed+rank)
        for module in student.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
            if hasattr(module, 'attention_dropout'):
                module.attention_dropout = 0.
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        student.config.use_cache = False
        optimizer = torch.optim.AdamW([p for _, p in params], lr=a.lr, weight_decay=0.)
        optimizer.zero_grad(set_to_none=True)
        initial = adapter_digest(params)
        protocol['workers'] = gather(dict(rank=rank, physical_gpus=os.environ['CUDA_VISIBLE_DEVICES'],
            teacher_device_map=teacher.hf_device_map, runtime=video.runtime_identity()))
        if rank == 0:
            video.write(out/'training_metadata.json', protocol)
        reasons, selection_counts, trained = Counter(), Counter(), []
        while processed < len(samples) and accepted < a.max_samples:
            count = wave_size(world, a.grad_accum, pending, a.max_samples, accepted, len(samples)-processed)
            record = None
            if rank < count:
                started = time.monotonic()
                record = candidate_gradient(a, student, teacher, processors, samples[processed+rank])
                record.update(candidate=processed+rank+1, worker_rank=rank, seconds=time.monotonic()-started)
            for row in gather(record)[:count]:
                reasons[row['reason']] += 1
                if row['accepted']:
                    accepted += 1
                    pending += 1
                    if row['boundary']:
                        key = 'both' if row['boundary']['start'] and row['boundary']['end'] else 'start_only' if row['boundary']['start'] else 'end_only'
                        selection_counts[key] += 1
                    if rank == 0:
                        trained.append(row['sample'])
                        video.append(out/'accepted_samples.jsonl', dict(row, accepted_index=accepted))
                if rank == 0:
                    video.append(out/'train.jsonl', dict(row, accepted_total=accepted, optimizer_steps_before=steps))
            processed += count
            if pending and (pending == a.grad_accum or accepted == a.max_samples or processed == len(samples)):
                for name, p in params:
                    value = torch.zeros(p.shape, dtype=torch.float32) if p.grad is None else p.grad.detach().float().cpu()
                    if world > 1:
                        dist.all_reduce(value)
                    if not torch.isfinite(value).all():
                        raise RuntimeError('Nonfinite adapter gradient: '+name)
                    p.grad = (value/pending).to(device=p.device, dtype=p.dtype)
                norm = torch.nn.utils.clip_grad_norm_([p for _, p in params], a.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                pending = 0
                if rank == 0 and a.save_every_steps and steps % a.save_every_steps == 0:
                    folder = out/f'step-{steps:06d}'
                    modeling.save_adapter(folder, adapters, protocol)
                    video.write(folder/'summary.json', dict(optimizer_steps=steps, accepted_events=accepted,
                        candidate_events=processed, gradient_norm=float(norm)))
            if rank == 0:
                print(f'[OPD] candidates={processed} accepted={accepted}/{a.max_samples} steps={steps} pending={pending} selections={dict(selection_counts)}', flush=True)
        hashes = gather(adapter_digest(params))
        if len(set(hashes)) != 1:
            raise RuntimeError('Student replicas diverged')
        success = accepted == a.max_samples and hashes[0] != initial
        summary = dict(training_complete=success, accepted_events=accepted, candidate_events=processed,
            accepted_target=a.max_samples, optimizer_steps=steps, gate_counts=dict(reasons),
            boundary_selection_counts=dict(selection_counts), merged=False,
            initial_adapter_sha256=initial, final_adapter_sha256=hashes[0], worker_adapter_sha256=hashes)
        if rank == 0:
            folder = out/('final-adapter' if success else 'partial-adapter')
            modeling.save_adapter(folder, adapters, protocol)
            video.write(folder/'summary.json', summary)
            video.write(out/'summary.json', summary)
            video.write(out/'train_samples.json', trained)
        if not success:
            raise RuntimeError('Accepted target/update incomplete; partial snapshot saved, no merge')
        if any(p.requires_grad or p.grad is not None for p in teacher.parameters()):
            raise RuntimeError('Teacher unexpectedly receives gradients')
        del teacher
        for i in range(a.gpus_per_worker):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
        if rank == 0 and not a.skip_merge:
            for name, adapter in adapters.items():
                adapter.merge()
                layer, target = name.split('.')
                setattr(student.model.language_model.layers[int(layer)].self_attn, target, adapter.base)
            student.gradient_checkpointing_disable()
            student.config.use_cache = True
            student.save_pretrained(out, safe_serialization=True, max_shard_size='4GB')
            processors[0].save_pretrained(out)
            summary['merged'] = True
            video.write(out/'summary.json', summary)
        barrier()
        return 0
    except Exception as error:
        if out.exists():
            video.write(out/f'failure_rank{rank}.json', dict(error=repr(error),
                accepted_events=accepted, candidate_events=processed, optimizer_steps=steps))
        raise
    finally:
        if decoder._decoder is not None:
            decoder._decoder.close()
            decoder._decoder = None
        if world > 1:
            dist.destroy_process_group()


def adapter_digest(params):
    h = hashlib.sha256()
    for name, p in params:
        h.update(name.encode())
        h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
