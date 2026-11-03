"""32B GT-clip teacher -> Qwen3 own GT-mask Top10, two K100s, 14336."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
import probe_qwen32b_privileged_k100 as common
import train_qwen32b_timelens8b_opd_k100 as engine


def parser():
    p = engine.parser()
    p.description = __doc__
    p.set_defaults(teacher_mode='gt_clip',gate='teacher-better',total_tokens=14336,min_tokens=64,
                   student_total_tokens=14336,student_min_tokens=64,
                   teacher_total_tokens=14336,teacher_min_tokens=64,max_frames=0,
                   lr=2e-5,grad_accum=4,save_every_steps=25,_probe_pool_input=True)
    return p


def prepare(a):
    if (a.teacher_mode != 'gt_clip' or a.gate != 'teacher-better' or a.selection_json
            or a.epochs != 1 or a.privilege_mix != 1):
        raise ValueError('Qwen3 entry requires GT-clip teacher, strict teacher-better gate and one pass')
    if ((a.student_total_tokens,a.teacher_total_tokens,a.total_tokens) != (14336,14336,14336)
            or (a.student_min_tokens,a.teacher_min_tokens,a.min_tokens) != (64,64,64)
            or a.fps != 2. or a.max_frames != 0):
        raise ValueError('Require student/teacher visual caps 14336, min64, fps2, max-frames0')
    if (a.student_target_iou,a.teacher_min_iou,a.iou_margin) != (1.,0.,0.):
        raise ValueError('teacher-better has no additional teacher IoU floor or gain margin')
    if (min(a.rank,a.grad_accum,a.max_samples) < 1 or a.max_attempts < 0 or a.save_every_steps < 0
            or not all(math.isfinite(v) and v > 0 for v in (a.lr,a.alpha,a.max_grad_norm))):
        raise ValueError('Invalid training parameters')
    protocol,samples = common.validate_common(a,sample_limit=0)
    if a.max_attempts:
        samples = samples[:a.max_attempts]
    if len(samples) < a.max_samples:
        raise ValueError('Candidate pool smaller than the accepted sample target')
    from train_timelens8b_qwen3_gtclip_opd_single40g import read_heads
    cfg = protocol['student_checkpoint']['config']['text_config']
    selected = read_heads(a.analysis_summary,a.student_path,cfg)
    if len(selected) != 10:
        raise ValueError('Require Qwen3 own verified GT-mask Top10, never TimeLens coordinates')
    heads = selected
    if a.head_selection == 'random-matched':
        rng = random.Random(a.seed)
        occupied = {(h['layer'],h['head']) for h in selected}
        counts = common.Counter(h['layer'] for h in selected)
        heads = [dict(layer=l,head=h) for l,n in sorted(counts.items())
                 for h in rng.sample([i for i in range(cfg['num_attention_heads']) if (l,i) not in occupied],n)]
    dim = cfg.get('head_dim',cfg['hidden_size']//cfg['num_attention_heads'])
    counts = common.Counter(h['layer'] for h in heads)
    protocol.update(method='qwen32b_gt_clip_to_qwen3_gtmask10_14336_strict_better_opd_v1',
        selected_heads=heads,original_gtmask_heads=selected,
        analysis_sha256=common.sha(a.analysis_summary),
        analysis_protocol_sha256=common.sha(Path(a.analysis_summary).parent/'protocol.json'),
        expected_trainable_parameters=2*a.rank*sum(cfg['hidden_size']+n*dim for n in counts.values()),
        candidate_count=len(samples),accepted_target=a.max_samples,
        candidate_samples_sha256=hashlib.sha256(json.dumps(samples,sort_keys=True).encode()).hexdigest(),
        sample_count_semantics='max-samples counts unique accepted updates; max-attempts caps candidate events',
        objective='Full-vocabulary KL(student || stop-gradient frozen teacher), averaged over sampled student continuation',
        gate=dict(mode='teacher-better',decision_rule='Teacher valid AND teacher greedy IoU > current sampled student IoU; invalid student=0; ties rejected',
                  failure_policy='Rejected events logged and skipped until accepted target or candidate exhaustion'),
        placement='Qwen3 student cuda:0; frozen 32B teacher sharded over cuda:0/1',
        video_protocol='Common 14336 dense pool; student normal view; teacher full-source context plus separately decoded GT clip under same total visual cap',
        privilege='GT selects only teacher reference clip and training IoU gate; student receives normal video/query',
        teacher_prompt_policy=dict(student='official normal video/query',teacher=common.GT_CLIP_GUIDANCE,
                                   teacher_mode='gt_clip',gt_answer_text=False,original_source_timestamps=True),
        clip_policy=dict(maximum_visual_tokens=a.clip_total_tokens,included_in_teacher_cap=True,
                         fps=a.clip_fps,max_frames=a.clip_max_frames,padding_seconds=a.clip_padding_seconds),
        head_selection_transfer='Qwen3 own GT-mask Top10 selected under 4096/min4; reused at 14336, not re-ranked at 14336',
        shortcut_note='GT crop plus original timestamps reveals approximate target position; teacher-gate pass is not student test improvement',
        inference='Exported student uses only normal video/query, no teacher or GT',
        resume='Weights-only snapshots; optimizer/RNG resume not implemented')
    for name in (Path(__file__).name,'train_qwen32b_timelens8b_opd_k100.py',
                 'train_timelens8b_qwen3_gtclip_opd_single40g.py'):
        protocol['source_sha256'][name] = common.sha(ROOT/'scripts'/name)
    return protocol,samples,{}


class Recipe:
    prepare = staticmethod(prepare)
    training_inputs = staticmethod(engine.training_inputs)
    generate = staticmethod(common.generate)
    teacher_targets = staticmethod(lambda y:y)
    select_logits = staticmethod(lambda logits,role:logits)

    @staticmethod
    def prepare_alignment(student,teacher,processors):
        common.tokenizer_equivalence(processors[0].tokenizer,processors[1].tokenizer)
        if student.config.text_config.vocab_size != teacher.config.text_config.vocab_size:
            raise ValueError('Teacher/student output vocabulary dimensions differ')
        return dict(mode='exact_full_vocabulary',note='Same sampled student token prefixes in both models')


digest = engine.digest
decide_gate = engine.decide_gate


def run(a, recipe=None):
    protocol, samples, roles = (recipe.prepare(a) if recipe else prepare(a))
    protocol['student_rollout_policy'] = ('One sampled student continuation is reused for IoU gating and OPD; '
        'teacher gate prediction remains greedy; sampled gate differs from earlier greedy-gate experiments')
    if isinstance(protocol.get('gate'), dict):
        protocol['gate']['decision_rule'] = ('Teacher valid and greedy teacher IoU > sampled student IoU; '
            'invalid student is assigned IoU 0; ties rejected') if a.gate == 'teacher-better' else (
            'Student gate uses the same sampled continuation as OPD; other configured gate rules unchanged')
    print(json.dumps(protocol, ensure_ascii=False, indent=2), flush=True)
    if a.plan_only:
        return 0
    import torch
    import train_timelens_three_group_opsd as core
    import scripts.qwen_decode_worker as decoder
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    common.write(out/'training_metadata.json', protocol)
    common.write(out/'candidate_samples.json', samples)
    trained_samples = []
    common.write(out/'train_samples.json', trained_samples)
    accepted = processed = steps = pending = 0
    reasons = common.Counter()
    try:
        student, teacher, processors = common.load_runtime(a)
        if recipe:
            protocol['token_alignment'] = recipe.prepare_alignment(student, teacher, processors)
        else:
            common.tokenizer_equivalence(processors[0].tokenizer, processors[1].tokenizer)
            if student.config.text_config.vocab_size != teacher.config.text_config.vocab_size:
                raise ValueError('Model output vocabulary dimensions differ')
        generate = recipe.generate if recipe else common.generate
        adapters = core.attach_head_lora(student, protocol['selected_heads'], a.rank, a.alpha, ['q_proj', 'o_proj'])
        params = sorted((name, p) for name, p in student.named_parameters() if p.requires_grad)
        if ({id(p) for _, p in params} != {id(p) for x in adapters.values() for p in (x.A, x.B)}
                or sum(p.numel() for _, p in params) != protocol['expected_trainable_parameters']):
            raise RuntimeError('Trainable parameters differ from the selected Q/O recipe')
        for module in student.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
            if hasattr(module, 'attention_dropout'):
                module.attention_dropout = 0.
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        student.config.use_cache = False
        optimizer = torch.optim.AdamW([p for _, p in params], lr=a.lr, weight_decay=0.)
        optimizer.zero_grad(set_to_none=True)
        initial = digest(params)
        protocol.update(teacher_device_map=teacher.hf_device_map,
                        trainable_parameters=sum(p.numel() for _, p in params),
                        runtime=common.runtime_identity())
        common.write(out/'training_metadata.json', protocol)

        def update():
            nonlocal steps, pending
            if not pending:
                return
            for name, p in params:
                if p.grad is None or not torch.isfinite(p.grad).all():
                    raise RuntimeError('Missing/nonfinite adapter gradient: '+name)
                p.grad.div_(pending)
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in params], a.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            pending = 0
            if a.save_every_steps and steps % a.save_every_steps == 0:
                folder = out/f'step-{steps:06d}'
                core.save_adapter(folder, adapters, protocol)
                common.write(folder/'summary.json', dict(steps=steps, accepted=accepted, processed=processed,
                    gradient_norm=float(norm), note='Weights-only snapshot; not optimizer resume or evaluation'))

        for epoch in range(a.epochs):
            order = list(samples)
            random.Random(a.seed+epoch).shuffle(order)
            for sample in order:
                if a.gate == 'teacher-better' and accepted >= a.max_samples:
                    break
                started = time.monotonic()
                for device in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(device)
                processed += 1
                student_gate = teacher_gate = None
                student_input, teacher_input, student_audit, teacher_audit, pool, duration = (
                    recipe.training_inputs(processors, sample, a) if recipe else training_inputs(processors, sample, a))
                student_input = common.move(student_input, student)
                # Generate once under the current normal-input student policy.
                # Reuse these exact tokens for both the gate and the OPD target prefix.
                y, rollout = generate(student, processors[0], student_input, sample, duration, a, sampled=True)
                if a.gate in ('dynamic', 'teacher-better'):
                    student_gate = rollout
                    if a.gate == 'dynamic' and student_gate['valid'] and student_gate['iou'] >= a.student_target_iou:
                        active, reason = False, 'student_already_sufficient'
                    else:
                        if teacher_input is None:
                            teacher_input, teacher_audit = common.build_input(processors[1], sample, pool, a.teacher_mode, a)
                        teacher_input = common.move(teacher_input, teacher)
                        _, teacher_gate = generate(teacher, processors[1], teacher_input, sample, duration, a)
                        active, reason = decide_gate(student_gate, teacher_gate, a)
                else:
                    active, reason = True, 'ungated'
                key = (sample['video_path'], sample['query'], sample['gt_start'], sample['gt_end'])
                record = dict(epoch=epoch+1, candidate=processed, sample=sample, role=roles.get(key, 'random'),
                    accepted=active, gate_reason=reason, student_gate=student_gate, teacher_gate=teacher_gate,
                    student_video=student_audit, teacher_video=teacher_audit,
                    matching_source_frames=student_audit.get('source_frame_indices') == teacher_audit.get('source_frame_indices')
                        if teacher_audit else None)
                reasons[reason] += 1
                if active:
                    # y is the same sampled continuation used for student_gate.
                    if teacher_input is None:
                        if a.gate == 'teacher-better':
                            raise RuntimeError('Missing independently prepared teacher input')
                        teacher_input, teacher_audit = common.build_input(processors[1], sample, pool, a.teacher_mode, a)
                    with torch.no_grad():
                        teacher_targets = recipe.teacher_targets(y) if recipe else y
                        teacher_logits = common.scoring_logits(teacher, teacher_input, teacher_targets)
                        if recipe:
                            teacher_logits = recipe.select_logits(teacher_logits, 'teacher')
                        log_q = teacher_logits.float().log_softmax(-1).cpu()
                        del teacher_logits
                        if a.privilege_mix < 1:
                            normal_input, _ = common.build_input(processors[1], sample, pool, 'normal', a)
                            normal_logits = common.scoring_logits(teacher, normal_input, y)
                            log_q0 = normal_logits.float().log_softmax(-1).cpu()
                            del normal_logits, normal_input
                            if a.privilege_mix == 0:
                                log_q = log_q0
                            else:
                                log_q = torch.logaddexp(log_q + math.log(a.privilege_mix),
                                                       log_q0 + math.log1p(-a.privilege_mix))
                            del log_q0
                    del teacher_input
                    teacher_input = None
                    student.train()
                    context = torch.autograd.graph.save_on_cpu(pin_memory=True) if a.offload_saved_tensors else contextlib.nullcontext()
                    with context:
                        logits = common.scoring_logits(student, student_input, y)
                        loss_logits = recipe.select_logits(logits, 'student') if recipe else logits
                        log_p = loss_logits.float().log_softmax(-1)
                        if log_p.shape != log_q.shape:
                            raise RuntimeError('Student/teacher output shape mismatch')
                        log_q = log_q.to(log_p.device)
                        loss = (log_p.exp()*(log_p-log_q)).sum(-1).mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite OPD reverse KL')
                    loss.backward()
                    accepted += 1
                    pending += 1
                    trained_samples.append(sample)
                    common.append(out/'accepted_samples.jsonl', dict(accepted_index=accepted,
                        candidate=processed, sample=sample, student_gate=student_gate, teacher_gate=teacher_gate))
                    record.update(loss=float(loss.detach()), rollout=rollout, teacher_video=teacher_audit)
                    del logits, loss_logits, teacher_targets, log_p, log_q, loss, y
                    if pending == a.grad_accum:
                        update()
                record.update(optimizer_steps=steps, accepted_total=accepted,
                              seconds=time.monotonic()-started, memory=common.memory())
                common.append(out/'train.jsonl', record)
                print(json.dumps({k:record[k] for k in ('epoch', 'candidate', 'accepted', 'gate_reason',
                    'optimizer_steps', 'accepted_total', 'seconds', 'memory')} | dict(loss=record.get('loss'),
                    accepted_target=a.max_samples if a.gate == 'teacher-better' else None,
                    student_iou=student_gate['iou'] if student_gate else None,
                    teacher_iou=teacher_gate['iou'] if teacher_gate else None,
                    student_tokens=student_audit['actual_visual_tokens'],
                    teacher_tokens=teacher_audit['actual_visual_tokens'] if teacher_audit else None,
                    teacher_clip_tokens=teacher_audit.get('clip_visual_tokens',0) if teacher_audit else None,
                    teacher_clip_timestamp_range=teacher_audit.get('clip',{}).get('sampled_timestamp_range') if teacher_audit else None,
                    matching_source_frames=record['matching_source_frames']),
                    ensure_ascii=False), flush=True)
                del student_input, teacher_input, pool
            # Average any final partial effective batch by its actual accepted count.
            update()
        final = digest(params)
        target_reached = accepted == a.max_samples if a.gate == 'teacher-better' else True
        core.save_adapter(out/('final-adapter' if target_reached else 'partial-adapter'), adapters, protocol)
        common.write(out/'train_samples.json', trained_samples)
        successful = accepted > 0 and final != initial and target_reached
        summary = dict(training_complete=successful, candidate_events=processed, accepted_events=accepted,
            target_reached=target_reached,
            accepted_target=a.max_samples if a.gate == 'teacher-better' else None,
            candidate_pool_exhausted=a.gate == 'teacher-better' and not target_reached,
            independent_budgets=protocol.get('independent_budgets'),
            optimizer_steps=steps, gate_counts=dict(reasons), merged=False, selected_heads=protocol['selected_heads'],
            initial_adapter_sha256=initial, final_adapter_sha256=final,
            note='Training completion/gate statistics only; independent normal-input student evaluation required')
        common.write(out/'summary.json', summary)
        if not successful:
            raise RuntimeError(f'Training target/update incomplete: accepted={accepted}, target={a.max_samples}, '
                               f'target_reached={target_reached}; partial adapter/diagnostics saved, no merge')
        if not a.skip_merge:
            # Release all teacher shards before merge/export to reduce peak memory.
            del teacher
            for device in range(torch.cuda.device_count()):
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
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
        return 0
    except Exception as error:
        common.write(out/'train_samples.json', trained_samples)
        common.write(out/'failure.json', dict(training_complete=False, processed=processed, accepted=accepted,
                                              optimizer_steps=steps, error=repr(error)))
        raise
    finally:
        if decoder._decoder is not None:
            decoder._decoder.close()
            decoder._decoder = None


def main(argv=None):
    return run(parser().parse_args(argv),recipe=Recipe())


