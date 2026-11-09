"""One-pass frozen-32B distillation, one GPU group or synchronized groups."""
import argparse
from collections import Counter
import contextlib
import hashlib
import json
import math
import inspect
import multiprocessing as mp
import queue
import traceback
import opd_resume as recovery
from opd_four_bucket import CandidateWindow, physical_layout, direction_has_room
import os
from pathlib import Path
import random
import subprocess
import sys
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
    p.add_argument('--rounds', type=int, default=1, help='Each round accepts max-samples NEW events; preserve student and optimizer')
    p.add_argument('--duration-ratio', nargs='+', type=int, default=None, help='3 buckets: <=5,5-15,>15; 4 buckets: <=5,5-15,15-60,>60')
    p.add_argument('--resume', action='store_true', help='Restore atomic resume.pt in the same output directory, including pending gradients')
    p.add_argument('--distill-kl', choices=('reverse',), default='reverse', help='Full-vocabulary RKL; Top10 is only the gate')
    p.add_argument('--pipeline', action='store_true')
    p.add_argument('--gpu-memory-gib', type=float, default=48.)
    p.add_argument('--prefetch-candidates', type=int, default=10)
    p.add_argument('--worker-timeout', type=float, default=3600.)
    p.add_argument('--end-direction-ratio', nargs=2, type=int, default=None)
    p.add_argument('--student-processor-path', default=None)
    p.add_argument('--teacher-processor-path', default=None)
    p.add_argument('--quiet', action='store_true')
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--token-local-gate', action='store_true', help='One teacher score; Top10 probability-weighted independent boundary-error gate; full-vocabulary KL on ordinary and accepted boundary positions')
    p.add_argument('--boundary-selective', action='store_true',
                   help='Independent boundary-error gate and selected-number-token KL; otherwise whole-answer IoU-gated baseline')
    p.add_argument('--boundary-epsilon', type=float, default=0., help='Strict improvement margin in seconds')
    for name, default in (('gpus-per-worker', 2), ('max-samples', 2000), ('max-attempts', 10000),
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
    p.add_argument('--auto-launch', action='store_true',
                   help='Detect visible CUDA GPUs and launch one synchronized sample worker per GPU group')
    return p


def automatic_topology(devices, size=2):
    if size < 2 or not devices or len(set(devices)) != len(devices) or not all(devices):
        raise ValueError('Require distinct visible GPU IDs and at least two GPUs per group')
    workers = len(devices)//size
    if not workers:
        raise ValueError(f'This 8B student + 32B teacher recipe requires at least {size} visible GPUs')
    used = devices[:workers*size]
    return workers, used, devices[workers*size:]


def launch_auto(a, values):
    import torch
    if not torch.cuda.is_available() or not (torch.version.cuda or torch.version.hip):
        raise RuntimeError('Require available CUDA or HIP GPUs')
    visibility = 'HIP_VISIBLE_DEVICES' if torch.version.hip else 'CUDA_VISIBLE_DEVICES'
    visible = os.environ.get(visibility)
    count = torch.cuda.device_count()
    devices = [d.strip() for d in visible.split(',')] if visible is not None else list(map(str,range(count)))
    if len(devices) != count:
        raise ValueError(f'{visibility} differs from the visible device count')
    workers, used, idle = automatic_topology(devices, a.gpus_per_worker)
    print(f'[auto] visible GPUs={count}; groups={workers}; GPUs/group={a.gpus_per_worker}; '
          f'groups={[used[i:i+a.gpus_per_worker] for i in range(0,len(used),a.gpus_per_worker)]}; unused={idle}', flush=True)
    child_args = [v for v in values if v != '--auto-launch']
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
               f'--nproc_per_node={workers}', str(Path(__file__).resolve()), *child_args]
    child_env = device_environment(os.environ, visibility, ','.join(used))
    return subprocess.run(command, env=child_env, check=False).returncode


def device_environment(environment, visibility, devices):
    result = dict(environment)
    for key in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','HSA_VISIBLE_DEVICES'):
        result.pop(key, None)
    result[visibility] = devices
    return result


def group_devices(visible, rank, world, size):
    ids = [x.strip() for x in visible.split(',')]
    if size < 2 or not 0 <= rank < world or len(ids) != world*size or len(set(ids)) != len(ids) or not all(ids):
        raise ValueError('Visible GPU IDs must form disjoint groups of gpus-per-worker (at least two per group)')
    return ','.join(ids[rank*size:(rank+1)*size])


def wave_size(workers, batch, pending, target, accepted, remaining):
    return min(workers, batch-pending, target-accepted, remaining)


def prepare(a):
    if a.rounds < 1 or a.log_every < 1 or (a.rounds > 1 and not a.duration_ratio):
        raise ValueError('Require positive rounds/log-every; multiple rounds require duration-ratio')
    if a.token_local_gate and a.boundary_selective:
        raise ValueError('Choose --token-local-gate or legacy --boundary-selective, not both')
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
    if a.resume and not (out/'resume.pt').is_file():
        raise FileNotFoundError('Full-state resume requires output-dir/resume.pt; old adapter-only snapshots cannot restore AdamW/RNG')
    if not a.resume and out.exists() and any(out.iterdir()):
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
    if a.max_attempts and not a.duration_ratio:
        samples = samples[:a.max_attempts]
    if len(samples) < a.max_samples:
        raise ValueError('Candidate pool smaller than accepted target')
    from qwen_token_local_gate import TOP10_POLICY, SECONDS_INSTRUCTION
    return dict(method='top10_independent_expected_error_seconds_opd_v2' if a.token_local_gate else 'boundary_selective_opd' if a.boundary_selective else 'whole_answer_opd_baseline',
        arguments=vars(a), student_checkpoint=student, teacher_checkpoint=teacher,
        selected_heads=heads, heads_sha256=video.sha(a.heads_json), data_stats=stats,
        candidate_count=len(samples), annotation_sha256=video.sha(a.anno_json),
        candidate_samples_sha256=hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest(),
        expected_trainable_parameters=2*a.rank*sum(sc['hidden_size']+n*dim for n in counts.values()),
        boundary_policy=dict(epsilon_seconds=a.boundary_epsilon, overall_iou_gate=False,
            rule=(TOP10_POLICY['criterion'] if a.token_local_gate else 'student absolute error minus teacher absolute error > epsilon independently per boundary'),
            invalid='Skip invalid prediction, unsupported format or unreliable token mapping'),
        token_local_policy=(dict(TOP10_POLICY, epsilon_seconds=a.boundary_epsilon) if a.token_local_gate else None),
        prompt_suffix=SECONDS_INSTRUCTION if a.token_local_gate else '',
        sample_acceptance=(TOP10_POLICY['sample_acceptance'] if a.token_local_gate else 'Historical gate'),
        objective=('Mean full-vocabulary KL(student || stopgrad teacher) over ordinary tokens and improving numeric/termination positions; average examples in effective batch' if a.token_local_gate else 'Mean full-vocabulary KL(student || stopgrad teacher) over selected numeric tokens per example; average examples in effective batch'
            if a.boundary_selective else 'Mean full-vocabulary KL over the entire student continuation; average examples in effective batch'),
        active_gate='Teacher Top10 expected error strictly below both student expected error on the same valid candidates and original boundary error; at least one boundary position; no ordering constraint on hypothetical replacements' if a.token_local_gate else 'Independent boundary absolute errors, no IoU gate' if a.boundary_selective else 'Teacher valid and teacher IoU > student IoU',
        baseline='Without --boundary-selective: valid teacher greedy IoU strictly greater than greedy student IoU (invalid student IoU=0); full continuation KL',
        teacher_policy=('Independent frozen 32B; full-source context plus original-timeline GT clip and exact GT text; score student trajectory once' if a.token_local_gate else 'Independent pretrained 32B, permanently frozen; full-source context plus original-timeline GT clip; no GT answer text'),
        student_policy='Normal full-source video/query; current student greedy continuation reused for gate and scoring',
        rollout_generation=dict(student_do_sample=False, teacher_do_sample=None if a.token_local_gate else False, teacher_answer_generated=not a.token_local_gate, num_beams=1,
            temperature=None, top_p=None, top_k=None, repetition_penalty=1.,
            max_new_tokens=a.max_new_tokens, policy_version='greedy_v1_20261105'),
        visual_budget='Global source context and reference clip share the total cap; normal student has equal actual visual-token count',
        source_sha256={str(p.relative_to(ROOT)): video.sha(p) for p in sorted(ROOT.rglob('*.py'))},
        resume='Weights-only snapshots; no optimizer/RNG resume'), samples


def candidate_gradient(a, student, teacher, processors, sample):
    import torch
    local = bool(getattr(a, 'token_local_gate', False))
    si, ti, sa, ta, duration = video.training_inputs(processors, sample, a)
    y, sg = video.generate(student, processors[0], si, sample, duration, a, sampled=False)
    decision = groups = tg = None
    indices = []
    log_q = None
    if local:
        active, reason = bool(sg['valid']), 'token_local'
        if active:
            from qwen_token_local_gate import completion_positions, validate_seconds_answer
            try:
                validate_seconds_answer(processors[0].tokenizer, y[0].tolist())
                _, _, groups = completion_positions(processors[0].tokenizer, y[0].tolist())
            except ValueError as error:
                active, reason = False, 'timestamp_tokens_'+str(error)
    else:
        _, tg = video.generate(teacher, processors[1], ti, sample, duration, a, sampled=False)
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
            if local:
                from qwen_token_local_gate import topk_gate, gate_change
                ids = y[0].tolist()
                indices, kept, audit = topk_gate(processors[0].tokenizer, ids, groups,
                    log_q[0], logits[0], [sample['gt_start'],sample['gt_end']], a.boundary_epsilon)
                record.update(accepted=bool(kept),
                    reason='top10_expected_error_improved' if kept else 'no_top10_effective_position',
                    token_local_gate=audit, selected_token_indices=indices,
                    selected_boundary_indices=kept, ordinary_tokens=len(ids)-sum(map(len,groups)))
                if not getattr(a, 'quiet', False):
                    print(f"[answer] {sg.get('output', sg.get('prediction'))!r} GT={[sample['gt_start'],sample['gt_end']]}", flush=True)
                    for entry in audit:
                        print(f"[token-{entry['field']}] pos={entry['position']} {entry['student_piece']!r}->{entry['teacher_piece']!r} {gate_change(entry)} {'DISTILL' if entry['accepted'] else 'SKIP'}", flush=True)
                    print(f"[scope] ordinary={record['ordinary_tokens']} boundary={len(kept)} total={len(indices)}/{len(ids)} full-vocab RKL", flush=True)
                if not kept:
                    return record
            loss = reverse_kl(logits, log_q, indices)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite reverse KL')
        loss.backward()
        record['loss'] = float(loss.detach())
    return record


def synchronized_main(argv=None):
    values = sys.argv[1:] if argv is None else list(argv)
    a = parser().parse_args(values)
    if a.auto_launch and 'LOCAL_RANK' not in os.environ:
        return launch_auto(a, values)
    rank, local = int(os.environ.get('RANK', '0')), int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if rank != local:
        raise ValueError('Single-node training only')
    visibility = 'HIP_VISIBLE_DEVICES' if 'HIP_VISIBLE_DEVICES' in os.environ else 'CUDA_VISIBLE_DEVICES'
    visible = os.environ.get(visibility, ','.join(map(str, range(world*a.gpus_per_worker))))
    selected_devices = group_devices(visible, local, world, a.gpus_per_worker)
    environment = device_environment(os.environ, visibility, selected_devices)
    for key in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','HSA_VISIBLE_DEVICES'):
        os.environ.pop(key, None)
    os.environ[visibility] = environment[visibility]
    protocol, samples = prepare(a)
    sampler = None
    target = a.max_samples * a.rounds
    if a.duration_ratio:
        from opd_duration_sampler import DurationRounds
        from opd_four_bucket import FourDurationRounds
        sampler_class = FourDurationRounds if len(a.duration_ratio)==4 else DurationRounds
        sampler = sampler_class(samples, a.max_samples, rounds=a.rounds,
            ratio=a.duration_ratio, seed=a.seed, max_attempts=a.max_attempts)
        protocol['duration_sampling'] = sampler.describe()
    protocol['accepted_target_global'] = target
    protocol['round_policy'] = ('Continuous student/AdamW; frozen teacher; only >60s bucket may cycle scarce events'
        if a.duration_ratio and len(a.duration_ratio)==4 else 'Continuous student/AdamW; frozen teacher; no candidate reuse')
    protocol['resume_policy'] = 'Four-bucket runs atomically save resume.pt after each completed wave, including per-rank pending gradients and RNG'
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
    import opd_resume as recovery
    if a.resume and (not sampler or len(a.duration_ratio)!=4):
        raise ValueError('Full-state resume is supported for the four-bucket recipe')
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
    saved = None
    lock = None
    try:
        if any(p != protocol for p in gather(protocol)):
            raise RuntimeError('Worker protocol/candidate mismatch')
        if rank == 0:
            out.mkdir(parents=True,exist_ok=True)
            lock=recovery.acquire_lock(out)
        if a.resume:
            saved = torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
            if saved['contract'] != recovery.contract(protocol) or saved['world'] != world:
                raise ValueError('Resume contract changed: keep model, data, code, quotas, batch and worker count unchanged')
            if rank == 0:
                recovery.rollback_journals(out,saved['journal_offsets'])
                import uuid
                for folder in list(out.glob('step-*'))+[out/'final-adapter',out/'partial-adapter']:
                    if folder.is_dir() and (folder.name in ('final-adapter','partial-adapter') or int(folder.name[5:])>saved['steps']):
                        folder.rename(out/('uncommitted_'+folder.name+'_'+uuid.uuid4().hex[:8]))
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
        protocol['workers'] = gather(dict(rank=rank, physical_gpus=os.environ[visibility], visibility_variable=visibility,
            teacher_device_map=teacher.hf_device_map, runtime=video.runtime_identity()))
        if rank == 0:
            video.write(out/'training_metadata.json', protocol)
        reasons, selection_counts, trained = Counter(), Counter(), []
        if saved is not None:
            for name,p in params:
                with torch.no_grad():p.copy_(saved['parameters'][name].to(p.device))
            optimizer.load_state_dict(saved['optimizer'])
            accepted,processed,steps,pending=(saved[k] for k in ('accepted','processed','steps','pending'))
            initial=saved['initial'];sampler.load_state_dict(saved['sampler'])
            reasons.update(saved['reasons']);selection_counts.update(saved['selection_counts'])
            trained=saved['trained'] if rank==0 else []
            state=saved['rank_states'][rank]
            for name,p in params:
                g=state['gradients'][name]
                p.grad=None if g is None else g.to(device=p.device,dtype=p.dtype)
            recovery.restore_rng(state['rng'])
            if rank==0:print(f'[resume] accepted={accepted} steps={steps} pending={pending}; AdamW/sampler/RNG restored',flush=True)
            del saved
        def save_resume():
            if not sampler or len(a.duration_ratio)!=4:return
            states=gather(dict(rng=recovery.rng_state(),gradients={n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in params}))
            if rank==0:
                recovery.atomic_save(out/'resume.pt',dict(version=1,contract=recovery.contract(protocol),world=world,
                    parameters={n:p.detach().cpu().clone() for n,p in params},optimizer=recovery.cpu_tree(optimizer.state_dict()),
                    accepted=accepted,processed=processed,steps=steps,pending=pending,initial=initial,
                    sampler=sampler.state_dict(),rank_states=states,reasons=dict(reasons),
                    selection_counts=dict(selection_counts),trained=trained,journal_offsets=recovery.journal_offsets(out)))
            barrier()
        save_resume()
        progress_started = time.monotonic()
        while processed < len(samples) and accepted < target:
            if sampler and sampler.accepted == sampler.quotas:
                if not sampler.advance_round():
                    break
            count = wave_size(world, a.grad_accum, pending, target, accepted, len(samples)-processed)
            wave = sampler.take_wave(count) if sampler else samples[processed:processed+count]
            count = len(wave)
            if not count:
                break
            record = None
            if rank < count:
                started = time.monotonic()
                record = candidate_gradient(a, student, teacher, processors, wave[rank])
                record.update(candidate=processed+rank+1, worker_rank=rank, seconds=time.monotonic()-started)
                if sampler:
                    record.update(round=sampler.round_index, duration_bucket=sampler.current)
            records = gather(record)[:count]
            if sampler:
                sampler.finish_wave([bool(row['accepted']) for row in records])
            for row in records:
                reasons[row['reason']] += 1
                if row['accepted']:
                    for token in row.get('token_local_gate', []):
                        selection_counts[token['field'] + ('_distill' if token['accepted'] else '_skip')] += 1
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
            if pending and (pending == a.grad_accum or accepted == target or processed == len(samples)):
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
                elapsed=time.monotonic()-progress_started
                eta=elapsed/accepted*max(0,target-accepted) if accepted else None
                video.append(out/'progress.jsonl',dict(candidates=processed,accepted=accepted,target=target,elapsed_seconds=elapsed,estimated_remaining_seconds=eta))
                eta_text=f'{eta/3600:.2f}h' if eta is not None else 'pending'
                if not a.quiet or processed % a.log_every < count or accepted == target:
                    print(f'[OPD] candidates={processed} accepted={accepted}/{target} steps={steps} '
                          f'round={sampler.round_index if sampler else 1}/{a.rounds} elapsed={elapsed/3600:.2f}h ETA~{eta_text}', flush=True)
            save_resume()
        if sampler and sampler.accepted == sampler.quotas and sampler.round_index == a.rounds:
            if len(getattr(sampler,'completed',[]))<a.rounds:sampler.advance_round()
        hashes = gather(adapter_digest(params))
        if len(set(hashes)) != 1:
            raise RuntimeError('Student replicas diverged')
        success = accepted == target and hashes[0] != initial
        summary = dict(training_complete=success, accepted_events=accepted, candidate_events=processed,
            accepted_target=target, accepted_per_round=a.max_samples, rounds=a.rounds,
            duration_sampling=sampler.describe() if sampler else None,
            kl_direction='reverse', optimizer_steps=steps, gate_counts=dict(reasons),
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
        if lock is not None:lock.close()
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



def end_direction(student_result, sample):
    if not student_result or not student_result.get('valid'):
        return None
    end = student_result['prediction'][1]
    gt = sample['gt_end']
    return 'early' if end < gt else 'late' if end > gt else 'equal'

def allocate_directions(directions, quotas, accepted):
    """Reserve no more than remaining accepted slots; rejections free slots next wave."""
    reserved = dict(accepted)
    allowed = []
    for direction in directions:
        ok = direction in quotas and reserved[direction] < quotas[direction]
        allowed.append(ok)
        if ok:
            reserved[direction] += 1
    return allowed

def pipeline_parser():
    p = parser()
    p.description = 'Four K100s: screen 0, teacher 1/2, online student 3; bounded candidate window'
    return p

def pipeline_isolate(devices):
    visibility = 'HIP_VISIBLE_DEVICES' if 'HIP_VISIBLE_DEVICES' in os.environ else 'CUDA_VISIBLE_DEVICES'
    env = device_environment(os.environ, visibility, devices)
    for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'HSA_VISIBLE_DEVICES'):
        os.environ.pop(key, None)
    os.environ[visibility] = env[visibility]

def pipeline_check_devices(count):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != count:
        raise RuntimeError(f'Pipeline expects CUDA/HIP torch and exactly {count} visible GPUs in this role')

def pipeline_processor(path):
    from transformers import AutoProcessor
    p = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=True, padding_side='left', do_resize=False)
    if p.tokenizer.pad_token_id is None:
        p.tokenizer.pad_token = p.tokenizer.eos_token
    vp = p.video_processor
    if tuple((int(getattr(vp, k)) for k in ('patch_size', 'merge_size', 'temporal_patch_size'))) != (16, 2, 2):
        raise ValueError('Require native Qwen3 patch/merge/temporal=16/2/2')
    return p

def pipeline_configure_decoder(a):
    import qwen_decode_worker as decoder
    decoder.configure_decoder(timeout=a.decode_timeout, attempts=3)

def pipeline_close_decoder():
    import qwen_decode_worker as decoder
    if decoder._decoder is not None:
        decoder._decoder.close()
        decoder._decoder = None

def pipeline_memory_limit(a):
    import torch
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory
        limit = min(total, int(a.gpu_memory_gib * 1024**3))
        torch.cuda.set_per_process_memory_fraction(limit / total, i)


def pipeline_load_student(a):
    import torch
    from transformers import AutoModelForImageTextToText
    from transformers import AutoConfig
    torch.manual_seed(a.seed)
    pipeline_memory_limit(a)
    checkpoint_bytes = sum(item['bytes'] for item in video.checkpoint_identity(a.student_path)['files'])
    available = min(torch.cuda.mem_get_info(0)[0], int(a.gpu_memory_gib * 1024**3))
    if checkpoint_bytes + a.student_reserve_gib * 1024**3 > available:
        raise RuntimeError('Student weights plus activation reserve do not fit the GPU memory budget')
    p = pipeline_processor(a.student_processor_path or a.student_path)
    saved = Path(a.student_path) / 'tokenizer.json'
    if saved.is_file():
        raw = json.loads(saved.read_text(encoding='utf-8'))
        vocab = dict(raw['model']['vocab'])
        vocab.update({x['content']: x['id'] for x in raw.get('added_tokens', [])})
        if vocab != p.tokenizer.get_vocab():
            raise ValueError('Student processor changed checkpoint token IDs')
    model = AutoModelForImageTextToText.from_pretrained(a.student_path, local_files_only=True, trust_remote_code=True, config=AutoConfig.from_pretrained(a.student_path, local_files_only=True, trust_remote_code=True), torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa')
    model.eval().requires_grad_(False)
    if 'logits_to_keep' not in inspect.signature(model.forward).parameters:
        raise RuntimeError('Installed model forward must support logits_to_keep')
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if hasattr(module, 'attention_dropout'):
            module.attention_dropout = 0.0
    return (model, p)

def pipeline_parameters(student, protocol, a):
    adapters = modeling.attach_head_lora(student, protocol['selected_heads'], a.rank, a.alpha, ['q_proj', 'o_proj'])
    params = sorted(((n, p) for n, p in student.named_parameters() if p.requires_grad))
    if sum((p.numel() for _, p in params)) != protocol['expected_trainable_parameters']:
        raise RuntimeError('Unexpected trainable parameters')
    return (adapters, params)

def pipeline_cpu_parameters(params):
    return {n: p.detach().cpu().clone() for n, p in params}

def pipeline_copy_parameters(params, state):
    import torch
    if set(state) != {n for n, _ in params}:
        raise ValueError('Adapter parameter names differ')
    with torch.no_grad():
        for n, p in params:
            if p.shape != state[n].shape:
                raise ValueError('Adapter parameter shape differs: ' + n)
            p.copy_(state[n].to(device=p.device, dtype=p.dtype))

def pipeline_screen_worker(a, protocol, inbox, outbox):
    import torch
    student, p = pipeline_load_student(a)
    _, params = pipeline_parameters(student, protocol, a)
    student.requires_grad_(False)
    pipeline_configure_decoder(a)
    weights = Path(a.output_dir) / 'screen_weights.pt'
    stamp, version = (None, None)
    outbox.put(dict(kind='ready', role='screen', runtime=video.runtime_identity()))
    while True:
        task = inbox.get()
        if task is None:
            return
        current = weights.stat().st_mtime_ns
        if current != stamp:
            state = torch.load(weights, map_location='cpu', weights_only=False)
            pipeline_copy_parameters(params, state['parameters'])
            version, stamp = (state['step'], current)
        started = time.monotonic()
        si, audit, pool, duration = video.student_input_for_direction(p, task['sample'], a)
        del pool
        y, result = video.generate(student, p, si, task['sample'], duration, a, sampled=False)
        direction = end_direction(result, task['sample'])
        outbox.put(dict(kind='screen', id=task['id'], policy_step=version, direction=direction, eligible=direction in task['directions'], student=result, rollout_token_ids=y[0].tolist(), seconds=time.monotonic() - started, actual_visual_tokens=audit['actual_visual_tokens']))
        del si, y

def pipeline_load_teacher(a):
    import torch
    from transformers import AutoModelForImageTextToText
    from accelerate import init_empty_weights, infer_auto_device_map
    from accelerate.utils import get_balanced_memory
    from transformers import AutoConfig
    pipeline_memory_limit(a)
    config = AutoConfig.from_pretrained(a.teacher_path, local_files_only=True, trust_remote_code=True)
    with init_empty_weights():
        empty = AutoModelForImageTextToText.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16)
    empty.tie_weights()
    budget = {i: int(min(torch.cuda.mem_get_info(i)[0], int(a.gpu_memory_gib * 1024 ** 3)) - max(6., a.teacher_reserve_gib) * 1024 ** 3) for i in range(2)}
    if min(budget.values()) <= 0:
        raise RuntimeError('Insufficient free teacher GPU memory')
    no_split = empty._no_split_modules or ['Qwen3VLDecoderLayer', 'Qwen3VLVisionBlock']
    balanced = get_balanced_memory(empty, max_memory=budget, no_split_module_classes=no_split, dtype=torch.bfloat16)
    device_map = infer_auto_device_map(empty, max_memory=balanced, no_split_module_classes=no_split, dtype=torch.bfloat16)
    if any((str(d) in ('cpu', 'disk') for d in device_map.values())):
        raise RuntimeError('32B teacher does not fit the two dedicated GPU budgets')
    if {str(d).replace('cuda:', '') for d in device_map.values()} != {'0', '1'}:
        raise RuntimeError('Expected teacher shards on both dedicated GPUs')
    del empty
    teacher = AutoModelForImageTextToText.from_pretrained(a.teacher_path, local_files_only=True, trust_remote_code=True, config=config, torch_dtype=torch.bfloat16, device_map=device_map, attn_implementation='sdpa')
    teacher.eval().requires_grad_(False)
    if 'logits_to_keep' not in inspect.signature(teacher.forward).parameters:
        raise RuntimeError('Installed teacher must support logits_to_keep')
    return teacher

def pipeline_teacher_worker(a, inbox, outbox):
    import torch
    teacher = pipeline_load_teacher(a)
    outbox.put(dict(kind='ready', role='teacher', runtime=video.runtime_identity(), device_map=teacher.hf_device_map, vocab_size=teacher.config.text_config.vocab_size))
    while True:
        request = inbox.get()
        if request is None:
            return
        started = time.monotonic()
        with torch.no_grad():
            logits = video.scoring_logits(teacher, request['inputs'], request['tokens'])
            log_q = logits.float().log_softmax(-1).cpu()
            del logits
        outbox.put(dict(kind='teacher', id=request['id'], log_q=log_q, seconds=time.monotonic() - started))

def pipeline_worker_entry(role, devices, a, protocol, inbox, outbox, errors):
    pipeline_isolate(devices)
    if os.name == 'posix':
        import signal

        def stop(signum, frame):
            raise SystemExit(0)
        signal.signal(signal.SIGTERM, stop)
    try:
        pipeline_check_devices(2 if role == 'teacher' else 1)
        if role == 'teacher':
            pipeline_teacher_worker(a, inbox, outbox)
        else:
            pipeline_screen_worker(a, protocol, inbox, outbox)
    except SystemExit:
        pass
    except BaseException:
        errors.put(dict(role=role, traceback=traceback.format_exc()))
        raise
    finally:
        pipeline_close_decoder()
        outbox.cancel_join_thread()

def pipeline_receive(outbox, workers, errors, timeout):
    deadline = time.monotonic() + timeout
    while True:
        try:
            error = errors.get_nowait()
        except queue.Empty:
            error = None
        if error:
            raise RuntimeError(error['role'] + ' worker failed:\n' + error['traceback'])
        if any((p.exitcode is not None for p in workers)):
            raise RuntimeError('Pipeline worker exited unexpectedly: ' + str([(p.name, p.exitcode) for p in workers]))
        if time.monotonic() >= deadline:
            raise TimeoutError('Pipeline worker response timed out')
        try:
            return outbox.get(timeout=0.2)
        except queue.Empty:
            pass

def pipeline_formal_gradient(a, student, teacher_processor, student_processor, task, quotas, counts, teacher_in, teacher_out, workers, errors):
    import torch
    from qwen_token_local_gate import completion_positions, validate_seconds_answer, topk_gate
    si, sa, pool, duration = video.student_input_for_direction(student_processor, task['sample'], a)
    y, sg = video.generate(student, student_processor, si, task['sample'], duration, a, sampled=False)
    direction = end_direction(sg, task['sample'])
    row = dict(sample=task['sample'], accepted=False, reason='formal_end_direction_filtered', student=sg, teacher=None, boundary=None, student_video=sa, teacher_video=None, end_direction=direction, selected_token_indices=[], rollout_token_ids=y[0].tolist())
    if not direction_has_room(direction, quotas, counts):
        return row
    try:
        ids = y[0].tolist()
        validate_seconds_answer(student_processor.tokenizer, ids)
        _, _, groups = completion_positions(student_processor.tokenizer, ids)
    except ValueError as error:
        row['reason'] = 'timestamp_tokens_' + str(error)
        return row
    ti, ta = video.teacher_input_for_direction(teacher_processor, task['sample'], a, pool, sa)
    del pool
    row['teacher_video'] = ta
    row['boundary_token_groups'] = groups
    teacher_in.put(dict(id=task['id'], inputs=ti, tokens=y))
    student.train()
    context = torch.autograd.graph.save_on_cpu(pin_memory=True) if a.offload_saved_tensors else contextlib.nullcontext()
    with context:
        logits = video.scoring_logits(student, si, y)
        response = pipeline_receive(teacher_out, workers, errors, a.worker_timeout)
        if response['kind'] != 'teacher' or response['id'] != task['id']:
            raise RuntimeError('Teacher response/trajectory mismatch')
        indices, kept, audit = topk_gate(student_processor.tokenizer, ids, groups, response['log_q'][0], logits[0], [task['sample']['gt_start'], task['sample']['gt_end']], a.boundary_epsilon)
        row.update(accepted=bool(kept), reason='top10_expected_error_improved' if kept else 'no_top10_effective_position', token_local_gate=audit, selected_token_indices=indices, selected_boundary_indices=kept, ordinary_tokens=len(ids) - sum(map(len, groups)), teacher_seconds=response['seconds'])
        if not kept:
            return row
        loss = reverse_kl(logits, response['log_q'], indices)
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite reverse KL')
    loss.backward()
    row['loss'] = float(loss.detach())
    return row

def pipeline_main(argv=None):
    a = pipeline_parser().parse_args(argv)
    if any((k in os.environ for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))):
        raise ValueError('Launch this pipeline with plain python, not torchrun')
    layout = physical_layout(os.environ.get('HIP_VISIBLE_DEVICES') or os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3'))
    if not a.token_local_gate or a.boundary_selective or a.rounds != 1 or (not a.end_direction_ratio) or (len(a.duration_ratio or []) != 4):
        raise ValueError('Pipeline requires token-local gate, end-direction ratio, four duration buckets and one round')
    if not math.isfinite(a.gpu_memory_gib) or a.gpu_memory_gib <= 0:
        raise ValueError('Require positive finite gpu-memory-gib')
    if any(v <= 0 for v in a.end_direction_ratio):
        raise ValueError('Require positive early/late ratio')
    if a.prefetch_candidates < 1 or a.worker_timeout <= 0:
        raise ValueError('Require positive prefetch-candidates and worker-timeout')
    protocol, samples = prepare(a)
    target = a.max_samples
    early = target * a.end_direction_ratio[0] // sum(a.end_direction_ratio)
    quotas = dict(early=early, late=target - early)
    scheduler = CandidateWindow(samples, target, a.duration_ratio, a.seed, a.max_attempts, a.prefetch_candidates)
    protocol.update(method='qwen3_end23_online_regeneration_pipeline_v1', accepted_target_global=target, end_direction_quotas=quotas, pipeline=dict(physical_gpus=layout, candidate_window=a.prefetch_candidates, screen_role='candidate metadata only; LoRA refreshed before each rollout when changed', train_role='regenerate with current weights; recheck direction; original token-local gate and full-vocab RKL', quota_accounting='final accepted formal rollouts; independent global duration and direction quotas', transport='bounded CPU queues; JSONL audit; tensor payloads in shared memory', resume='restore optimizer/RNG/pending gradients and candidate reservations; regenerate pending prescreens'), duration_sampling=scheduler.source.describe(), resume='Full pipeline state; incompatible with previous synchronized-worker checkpoints')
    for name in ('train_qwen3_32b_head_opd_dp4_mp2.py', 'opd_four_bucket.py'):
        protocol['source_sha256'][name] = video.sha(Path(__file__).parent / name)
    if a.plan_only:
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return 0
    return pipeline_run(a, protocol, samples, quotas, scheduler, layout)

def pipeline_run(a, protocol, samples, quotas, scheduler, layout):
    pipeline_isolate(layout['train'])
    import torch
    pipeline_check_devices(1)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    lock = recovery.acquire_lock(out)
    workers, channels = ([], [])
    accepted = processed = steps = pending = formal_rollouts = 0
    counts, reasons, selections, trained = (dict(early=0, late=0), Counter(), Counter(), [])
    try:
        saved = torch.load(out / 'resume.pt', map_location='cpu', weights_only=False) if a.resume else None
        if saved and saved['contract'] != recovery.contract(protocol):
            raise ValueError('Pipeline resume contract changed; keep code, data, models, topology and quotas unchanged')
        if saved:
            recovery.rollback_journals(out, saved['journal_offsets'])
            import uuid
            for folder in list(out.glob('step-*')) + [out / 'final-adapter', out / 'partial-adapter']:
                if folder.is_dir() and (folder.name in ('final-adapter', 'partial-adapter') or int(folder.name[5:]) > saved['steps']):
                    folder.rename(out / ('uncommitted_' + folder.name + '_' + uuid.uuid4().hex[:8]))
        student, sp = pipeline_load_student(a)
        tp = pipeline_processor(a.teacher_processor_path or a.teacher_path)
        video.tokenizer_equivalence(sp.tokenizer, tp.tokenizer)
        adapters, params = pipeline_parameters(student, protocol, a)
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        student.config.use_cache = False
        optimizer = torch.optim.AdamW([p for _, p in params], lr=a.lr, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(a.seed)
        random.seed(a.seed)
        initial = adapter_digest(params)
        if saved:
            pipeline_copy_parameters(params, saved['parameters'])
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            accepted, processed, steps, pending, formal_rollouts = (saved[k] for k in ('accepted', 'processed', 'steps', 'pending', 'formal_rollouts'))
            counts.update(saved['direction_counts'])
            reasons.update(saved['reasons'])
            selections.update(saved['selections'])
            trained, initial = (saved['trained'], saved['initial'])
            for n, p in params:
                g = saved['gradients'][n]
                p.grad = None if g is None else g.to(device=p.device, dtype=p.dtype)
            recovery.restore_rng(saved['rng'])
            del saved
        pipeline_configure_decoder(a)

        def publish():
            recovery.atomic_save(out / 'screen_weights.pt', dict(step=steps, parameters=pipeline_cpu_parameters(params)))
        publish()
        ctx = mp.get_context('spawn')
        screen_in, screen_out = (ctx.Queue(a.prefetch_candidates), ctx.Queue(a.prefetch_candidates))
        teacher_in, teacher_out, errors = (ctx.Queue(1), ctx.Queue(1), ctx.Queue())
        channels = [screen_in, screen_out, teacher_in, teacher_out, errors]
        for role, inbox, outbox in [('screen', screen_in, screen_out), ('teacher', teacher_in, teacher_out)]:
            p = ctx.Process(target=pipeline_worker_entry, args=(role, layout[role], a, protocol, inbox, outbox, errors), name=role)
            p.start()
            workers.append(p)
        runtime = [pipeline_receive(screen_out, workers, errors, a.worker_timeout), pipeline_receive(teacher_out, workers, errors, a.worker_timeout)]
        if any((r.get('kind') != 'ready' for r in runtime)):
            raise RuntimeError('Worker handshake failed')
        if runtime[1]['vocab_size'] != student.config.text_config.vocab_size:
            raise ValueError('Teacher/student output vocabulary dimensions differ')
        metadata = dict(protocol, workers=runtime, learner_runtime=video.runtime_identity())
        video.write(out / 'training_metadata.json', metadata)
        video.write(out / 'candidate_samples.json', samples)
        print(f"[pipeline] screen={layout['screen']} teacher={layout['teacher']} train={layout['train']} window={a.prefetch_candidates}; duration={scheduler.source.quotas}; end={quotas}", flush=True)
        for task in scheduler.pending.values():
            screen_in.put(task)

        def refill():
            directions = [d for d in quotas if direction_has_room(d, quotas, counts)]
            while True:
                task = scheduler.take(directions)
                if task is None:
                    break
                screen_in.put(task)

        def save_state():
            offsets = recovery.journal_offsets(out)
            offsets['screen.jsonl'] = (out / 'screen.jsonl').stat().st_size if (out / 'screen.jsonl').exists() else 0
            recovery.atomic_save(out / 'resume.pt', dict(version=1, contract=recovery.contract(protocol), parameters=pipeline_cpu_parameters(params), optimizer=recovery.cpu_tree(optimizer.state_dict()), gradients={n: None if p.grad is None else p.grad.detach().cpu().clone() for n, p in params}, rng=recovery.rng_state(), scheduler=scheduler.state_dict(), initial=initial, accepted=accepted, processed=processed, steps=steps, pending=pending, formal_rollouts=formal_rollouts, direction_counts=counts, reasons=dict(reasons), selections=dict(selections), trained=trained, journal_offsets=offsets))
        refill()
        save_state()
        started, initial_accepted = (time.monotonic(), accepted)
        while accepted < a.max_samples and scheduler.pending:
            screen = pipeline_receive(screen_out, workers, errors, a.worker_timeout)
            task = scheduler.pending[screen['id']]
            video.append(out / 'screen.jsonl', dict(screen, sample=task['sample'], duration_bucket=task['bucket']))
            row = dict(sample=task['sample'], accepted=False, reason='prescreen_filtered', end_direction=screen['direction'], student=screen['student'])
            sample_started = time.monotonic()
            if screen['eligible'] and direction_has_room(screen['direction'], quotas, counts):
                formal_rollouts += 1
                row = pipeline_formal_gradient(a, student, tp, sp, task, quotas, counts, teacher_in, teacher_out, workers, errors)
            row.update(candidate=processed + 1, task_id=task['id'], duration_bucket=task['bucket'], formal_policy_step=steps, screen_policy_step=screen['policy_step'], screen_end_direction=screen['direction'], seconds=time.monotonic() - sample_started)
            reasons[row['reason']] += 1
            if row['accepted']:
                direction = row['end_direction']
                if not direction_has_room(direction, quotas, counts):
                    raise RuntimeError('Direction quota exceeded')
                counts[direction] += 1
                accepted += 1
                pending += 1
                trained.append(task['sample'])
                for token in row['token_local_gate']:
                    selections[token['field'] + ('_distill' if token['accepted'] else '_skip')] += 1
                video.append(out / 'accepted_samples.jsonl', dict(row, accepted_index=accepted))
            scheduler.finish(task['id'], row['accepted'])
            processed += 1
            video.append(out / 'train.jsonl', dict(row, accepted_total=accepted, optimizer_steps_before=steps))
            if pending and (pending == a.grad_accum or accepted == a.max_samples):
                with torch.no_grad():
                    for _, p in params:
                        if p.grad is not None:
                            p.grad.div_(pending)
                torch.nn.utils.clip_grad_norm_([p for _, p in params], a.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                pending = 0
                publish()
                if a.save_every_steps and steps % a.save_every_steps == 0:
                    folder = out / f'step-{steps:06d}'
                    modeling.save_adapter(folder, adapters, protocol)
                    video.write(folder / 'summary.json', dict(optimizer_steps=steps, accepted_events=accepted))
            refill()
            elapsed, new = (time.monotonic() - started, accepted - initial_accepted)
            eta = elapsed / new * (a.max_samples - accepted) if new else None
            progress = dict(candidates=processed, formal_rollouts=formal_rollouts, accepted=accepted, target=a.max_samples, steps=steps, queued=len(scheduler.pending), end_direction_counts=counts, duration_counts=scheduler.source.accepted, elapsed_seconds=elapsed, estimated_remaining_seconds=eta)
            video.append(out / 'progress.jsonl', progress)
            if not a.quiet or processed % a.log_every == 0 or accepted == a.max_samples:
                eta_text = f'{eta / 3600:.2f}h' if eta is not None else 'pending'
                print(f'[pipeline] candidates={processed} formal={formal_rollouts} accepted={accepted}/{a.max_samples} steps={steps} queued={len(scheduler.pending)}/{a.prefetch_candidates} end={counts} duration={scheduler.source.accepted} elapsed={elapsed / 3600:.2f}h ETA~{eta_text}', flush=True)
            save_state()
        success = accepted == a.max_samples and counts == quotas and (scheduler.source.accepted == scheduler.source.quotas)
        if success:
            scheduler.source.advance_round()
        summary = dict(training_complete=success, accepted_events=accepted, accepted_target=a.max_samples, candidate_events=processed, formal_rollouts=formal_rollouts, optimizer_steps=steps, end_direction_quotas=quotas, end_direction_counts=counts, duration_sampling=scheduler.source.describe(), gate_counts=dict(reasons), boundary_selection_counts=dict(selections), merged=False, initial_adapter_sha256=initial, final_adapter_sha256=adapter_digest(params))
        success = success and summary['initial_adapter_sha256'] != summary['final_adapter_sha256']
        summary['training_complete'] = success
        modeling.save_adapter(out / ('final-adapter' if success else 'partial-adapter'), adapters, protocol)
        video.write(out / 'summary.json', summary)
        video.write(out / 'train_samples.json', trained)
        if not success:
            raise RuntimeError('Candidate pool/attempt limit exhausted before all quotas; partial adapter and resume saved')
        if not a.skip_merge:
            for name, adapter in adapters.items():
                adapter.merge()
                layer, target = name.split('.')
                setattr(student.model.language_model.layers[int(layer)].self_attn, target, adapter.base)
            student.gradient_checkpointing_disable()
            student.config.use_cache = True
            student.save_pretrained(out, safe_serialization=True, max_shard_size='4GB')
            sp.save_pretrained(out)
            summary['merged'] = True
            video.write(out / 'summary.json', summary)
        return 0
    except BaseException as error:
        video.write(out / 'failure_pipeline.json', dict(error=repr(error), traceback=traceback.format_exc(), accepted_events=accepted, candidate_events=processed, optimizer_steps=steps))
        raise
    finally:
        for p in workers:
            if p.is_alive():
                p.terminate()
        for p in workers:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join(timeout=5)
        for channel in channels:
            channel.cancel_join_thread()
            channel.close()
        pipeline_close_decoder()
        lock.close()

def main(argv=None):
    values = sys.argv[1:] if argv is None else list(argv)
    if "--pipeline" in values:
        return pipeline_main(values)
    return synchronized_main(values)

if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

