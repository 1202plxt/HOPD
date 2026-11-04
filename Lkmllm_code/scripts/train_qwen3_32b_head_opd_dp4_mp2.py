"""Four synchronized sample workers, two visible NVIDIA GPUs per worker.

Launch with torchrun --standalone --nproc_per_node=4 on one eight-GPU node.
Only compact adapter gradients are reduced, on CPU through Gloo.
"""
import os


def gpu_pair(visible, rank, world):
    devices = visible.split(',')
    if world != 4 or len(devices) != 8 or len(set(devices)) != 8:
        raise ValueError('Require four workers and eight distinct visible GPU IDs')
    if not 0 <= rank < world:
        raise ValueError('Invalid local rank')
    return ','.join(devices[rank * 2:rank * 2 + 2])


def wave_size(world, batch, pending, target, accepted, remaining):
    # Never speculate beyond the current effective batch or accepted target.
    return min(world, batch - pending, target - accepted, remaining)


def round_candidates(samples, seed, round_index, policy, attempted_keys, limit):
    import random
    order = list(samples)
    random.Random(seed + round_index - 1).shuffle(order)
    if policy == 'disjoint':
        order = [s for s in order if sample_identity(s) not in attempted_keys]
    if limit:
        order = order[:limit]
    return order


def sample_identity(sample):
    return (sample['video_path'], sample['query'], sample['gt_start'], sample['gt_end'])


def candidate_gradient(a, recipe, student, teacher, processors, sample):
    import contextlib
    import torch
    import probe_qwen32b_privileged_k100 as common
    import train_qwen32b_qwen3_gtclip_opd_k100 as base
    student.eval()
    si, ti, sa, ta, pool, duration = recipe.training_inputs(processors, sample, a)
    si = common.move(si, student)
    y, sg = recipe.generate(student, processors[0], si, sample, duration, a, sampled=True)
    ti = common.move(ti, teacher)
    _, tg = recipe.generate(teacher, processors[1], ti, sample, duration, a)
    active, reason = base.decide_gate(sg, tg, a)
    record = dict(sample=sample, accepted=active, gate_reason=reason,
                  student_gate=sg, teacher_gate=tg, student_video=sa, teacher_video=ta)
    if active:
        with torch.no_grad():
            logits = recipe.select_logits(common.scoring_logits(teacher, ti, recipe.teacher_targets(y)), 'teacher')
            log_q = logits.float().log_softmax(-1).cpu()
            del logits
        del ti, pool
        student.train()
        context = torch.autograd.graph.save_on_cpu(pin_memory=True) if a.offload_saved_tensors else contextlib.nullcontext()
        with context:
            logits = recipe.select_logits(common.scoring_logits(student, si, y), 'student')
            log_p = logits.float().log_softmax(-1)
            log_q = log_q.to(log_p.device)
            if log_p.shape != log_q.shape:
                raise RuntimeError('Student/teacher vocabulary mismatch')
            loss = (log_p.exp() * (log_p - log_q)).sum(-1).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        record['loss'] = float(loss.detach())
    return record


def main(entry_override=None):
    # Restrict visibility BEFORE importing anything that might initialize CUDA.
    rank = int(os.environ.get('RANK', '-1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '-1'))
    world = int(os.environ.get('WORLD_SIZE', '0'))
    if rank != local_rank:
        raise ValueError('This launcher supports one node only')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_pair(visible, local_rank, world)
    import datetime
    import json
    import random
    import time
    from pathlib import Path
    import torch
    import torch.distributed as dist
    import train_qwen3_32b_direct_head_opd as entry
    import train_timelens_three_group_opsd as core
    import scripts.qwen_decode_worker as decoder
    common = entry.common
    original = entry
    if entry_override is not None:
        entry = entry_override
    a = entry.parser().parse_args()
    a._group_data_parallel = True
    if a.student_device != 0:
        raise ValueError('Each two-GPU group requires --student-device 0')
    dist.init_process_group('gloo', timeout=datetime.timedelta(minutes=60))
    out = Path(a.output_dir)
    accepted = processed = steps = pending = 0
    try:
        protocol, samples, _ = entry.prepare(a)
        protocol.update(method=('qwen3_32b_direct_heads_dp4_mp2_v1' if entry_override is None else protocol['method']),
            placement='Four disjoint two-GPU model groups; synchronized student adapters',
            distributed=dict(workers=world, gpus_per_worker=2, global_effective_batch=a.grad_accum,
                gradient_reduction='CPU float32 sum divided by global accepted count',
                scheduling='Disjoint candidate waves capped by remaining effective batch slots',
                rng='Same initial adapters broadcast from rank0; rank-offset rollout RNG'),
            student_rollout_policy='Same sampled continuation for gate and OPD',
            arguments=dict(vars(a)))
        protocol['source_sha256'][Path(__file__).name] = common.sha(Path(__file__))
        identities = [None] * world
        dist.all_gather_object(identities, protocol)
        if any(x != identities[0] for x in identities):
            raise RuntimeError('Worker configuration/sample identity mismatch')
        if rank == 0:
            print(json.dumps(protocol, ensure_ascii=False, indent=2), flush=True)
        if a.plan_only:
            return 0
        if torch.cuda.device_count() != 2:
            raise RuntimeError('Each worker must see exactly two GPUs')
        if rank == 0:
            out.mkdir(parents=True, exist_ok=True)
            common.write(out/'training_metadata.json', protocol)
            common.write(out/'candidate_samples.json', samples)
        dist.barrier()
        student, teacher, processors = (common.load_runtime(a) if entry_override is None else entry.load_runtime(a))
        recipe = entry.Recipe()
        protocol['token_alignment'] = recipe.prepare_alignment(student, teacher, processors)
        adapters = core.attach_head_lora(student, protocol['selected_heads'], a.rank, a.alpha, ['q_proj', 'o_proj'])
        params = sorted((n, p) for n, p in student.named_parameters() if p.requires_grad)
        if ({id(p) for _, p in params} != {id(p) for x in adapters.values() for p in (x.A, x.B)}
                or sum(p.numel() for _, p in params) != protocol['expected_trainable_parameters']):
            raise RuntimeError('Unexpected trainable parameters')
        with torch.no_grad():
            for _, p in params:
                value = p.detach().cpu().clone()
                dist.broadcast(value, src=0)
                p.copy_(value.to(p.device))
        torch.manual_seed(a.seed + rank)
        random.seed(a.seed + rank)
        for module in student.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
            if hasattr(module, 'attention_dropout'):
                module.attention_dropout = 0.
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        student.config.use_cache = False
        optimizer = torch.optim.AdamW([p for _, p in params], lr=a.lr, weight_decay=0.)
        optimizer.zero_grad(set_to_none=True)
        initial = original.base.digest(params)
        def adapter_hash(items):
            return original.base.digest(sorted((name+'.'+key, getattr(adapter, key))
                for name, adapter in items.items() for key in ('A', 'B')))
        initial_canonical = adapter_hash(adapters)
        teacher_adapters = None
        if entry_override is not None:
            teacher_adapters = core.attach_head_lora(teacher, protocol['selected_heads'], a.rank, a.alpha, ['q_proj', 'o_proj'])
            entry.refresh_teacher(adapters, teacher_adapters)
            teacher.requires_grad_(False).eval()
            for module in teacher.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.
                if hasattr(module, 'attention_dropout'):
                    module.attention_dropout = 0.
        runtimes = [None] * world
        dist.all_gather_object(runtimes, dict(rank=rank, physical_gpus=os.environ['CUDA_VISIBLE_DEVICES'],
            teacher_device_map=teacher.hf_device_map, runtime=common.runtime_identity()))
        protocol['worker_runtimes'] = runtimes
        if rank == 0:
            common.write(out/'training_metadata.json', protocol)
        total_rounds = getattr(a, 'rounds', 1)
        attempted_keys = set()
        round_summaries = []
        all_trained = []
        all_reasons = common.Counter()
        for round_index in range(1, total_rounds + 1):
            round_out = out if total_rounds == 1 else out/f'round-{round_index:02d}'
            if rank == 0:
                round_out.mkdir(parents=True, exist_ok=True)
            accepted = processed = pending = 0
            round_start_steps = steps
            if teacher_adapters is not None:
                if round_index == 1 or a.teacher_refresh == 'per-round':
                    entry.refresh_teacher(adapters, teacher_adapters)
                teacher_params = sorted((name+'.'+key, getattr(adapter, key))
                    for name, adapter in teacher_adapters.items() for key in ('A', 'B'))
                teacher_hashes = [None] * world
                dist.all_gather_object(teacher_hashes, original.base.digest(teacher_params))
                if len(set(teacher_hashes)) != 1:
                    raise RuntimeError('Self-teacher replicas differ')
                expected_teacher = adapter_hash(adapters) if a.teacher_refresh == 'per-round' else initial_canonical
                if teacher_hashes[0] != expected_teacher:
                    raise RuntimeError('Teacher does not match its intended student version')
                if rank == 0:
                    common.write(round_out/'teacher_version.json', dict(round=round_index,
                        refresh=a.teacher_refresh, teacher_adapter_sha256=teacher_hashes[0],
                        source_round=(round_index-1 if a.teacher_refresh == 'per-round' else 0),
                        base_checkpoint=a.student_path))
                    core.save_adapter(round_out/'teacher-adapter', teacher_adapters, protocol)
            dist.barrier()
            order = round_candidates(samples, a.seed, round_index,
                getattr(a, 'candidate_policy', 'replay'), attempted_keys, a.max_attempts)
            trained = []
            reasons = common.Counter()
            while processed < len(order) and accepted < a.max_samples:
                count = wave_size(world, a.grad_accum, pending, a.max_samples, accepted, len(order)-processed)
                record = None
                if rank < count:
                    started = time.monotonic()
                    record = candidate_gradient(a, recipe, student, teacher, processors, order[processed+rank])
                    record.update(candidate=processed+rank+1, worker_rank=rank, epoch=1, round=round_index,
                                  seconds=time.monotonic()-started)
                records = [None] * world
                dist.all_gather_object(records, record)
                for row in records[:count]:
                    attempted_keys.add(sample_identity(row['sample']))
                    reasons[row['gate_reason']] += 1
                    if row['accepted']:
                        accepted += 1
                        pending += 1
                        if rank == 0:
                            trained.append(row['sample'])
                            common.append(round_out/'accepted_samples.jsonl', dict(row, accepted_index=accepted))
                    if rank == 0:
                        common.append(round_out/'train.jsonl', dict(row, accepted_total=accepted, optimizer_steps=steps))
                processed += count
                if pending and (pending == a.grad_accum or accepted == a.max_samples or processed == len(order)):
                    for name, p in params:
                        value = torch.zeros(p.shape, dtype=torch.float32) if p.grad is None else p.grad.detach().float().cpu()
                        dist.all_reduce(value)
                        if not torch.isfinite(value).all():
                            raise RuntimeError('Nonfinite global gradient: '+name)
                        p.grad = (value / pending).to(device=p.device, dtype=p.dtype)
                    norm = torch.nn.utils.clip_grad_norm_([p for _, p in params], a.max_grad_norm, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    steps += 1
                    pending = 0
                    if rank == 0 and a.save_every_steps and steps % a.save_every_steps == 0:
                        folder = round_out/f'step-{steps:06d}'
                        core.save_adapter(folder, adapters, protocol)
                        common.write(folder/'summary.json', dict(steps=steps, round=round_index, accepted=accepted, processed=processed,
                            gradient_norm=float(norm), note='Weights-only snapshot, not optimizer resume'))
                if rank == 0:
                    print(f'[dp4/mp2 round={round_index}] candidates={processed} accepted={accepted}/{a.max_samples} steps={steps} pending={pending}', flush=True)
            round_hashes = [None] * world
            dist.all_gather_object(round_hashes, original.base.digest(params))
            if len(set(round_hashes)) != 1:
                raise RuntimeError('Student adapters diverged at round boundary')
            if teacher_adapters is not None:
                if original.base.digest(teacher_params) != teacher_hashes[0]:
                    raise RuntimeError('Frozen self-teacher changed within the round')
                if any(p.grad is not None or p.requires_grad for _, p in teacher_params):
                    raise RuntimeError('Teacher unexpectedly has trainable parameters or gradients')
            reached = accepted == a.max_samples
            round_summary = dict(round=round_index, target_reached=reached,
                accepted_events=accepted, candidate_events=processed,
                optimizer_steps=steps-round_start_steps, cumulative_optimizer_steps=steps,
                gate_counts=dict(reasons), worker_adapter_sha256=round_hashes,
                student_adapter_sha256=adapter_hash(adapters))
            round_summaries.append(round_summary)
            all_reasons.update(reasons)
            if rank == 0:
                all_trained.extend(trained)
            if rank == 0:
                core.save_adapter(round_out/('student-adapter' if reached else 'partial-adapter'), adapters, protocol)
                common.write(round_out/'round_summary.json', round_summary)
                common.write(round_out/('student-adapter' if reached else 'partial-adapter')/'summary.json', round_summary)
                common.write(round_out/'round_train_samples.json', trained)
            if not reached:
                raise RuntimeError(f'Round {round_index}: accepted target not reached; stopping before refresh')
            dist.barrier()
        hashes = [None] * world
        dist.all_gather_object(hashes, original.base.digest(params))
        if len(set(hashes)) != 1:
            raise RuntimeError('Student adapters diverged across workers')
        success = accepted == a.max_samples and hashes[0] != initial
        if rank == 0:
            core.save_adapter(out/('final-adapter' if success else 'partial-adapter'), adapters, protocol)
            common.write(out/'train_samples.json', all_trained)
            summary = dict(training_complete=success, target_reached=accepted == a.max_samples,
                candidate_events=sum(r['candidate_events'] for r in round_summaries),
                accepted_events=sum(r['accepted_events'] for r in round_summaries),
                accepted_target=a.max_samples * total_rounds, rounds=round_summaries,
                optimizer_steps=steps, gate_counts=dict(all_reasons), merged=False,
                initial_adapter_sha256=initial, final_adapter_sha256=hashes[0], worker_adapter_sha256=hashes)
            common.write(out/'summary.json', summary)
        if not success:
            raise RuntimeError('Accepted target not reached or adapters unchanged; partial output saved')
        del teacher
        for device in range(2):
            with torch.cuda.device(device):
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
            common.write(out/'summary.json', summary)
        dist.barrier()
        return 0
    except Exception as error:
        if out.exists():
            common.write(out/f'failure_rank{rank}.json', dict(error=repr(error), accepted=accepted,
                processed=processed, optimizer_steps=steps))
        raise
    finally:
        if decoder._decoder is not None:
            decoder._decoder.close()
            decoder._decoder = None
        dist.destroy_process_group()


if __name__ == '__main__':
    raise SystemExit(main())
