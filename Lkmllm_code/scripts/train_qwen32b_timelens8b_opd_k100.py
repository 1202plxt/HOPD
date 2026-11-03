"""Frozen sharded Qwen3-VL-32B teacher -> TimeLens-8B GT-mask10 Q/O OPD.

Single process / two K100s, teacher inference on both, student training on GPU0.
Student generates its own rollout; frozen teacher scores EXACTLY that prefix.
Exact vocabulary reverse KL; no GT-answer CE or hidden-state matching.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_qwen32b_privileged_k100 as common


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.epilog = ('Weights-only continuation: --resume-snapshot STEP_DIR --output-dir NEW_DIR '
                '[--total-accepted-target 1000] [--plan-only]; optimizer/RNG restart.')
    common.add_common(p)
    for action in p._actions:
        if action.dest == 'max_samples':
            action.help = 'teacher-better: unique ACCEPTED event target; legacy gates: candidate count'
    p.add_argument('--analysis-summary', required=True, help='STUDENT completed 300-pair GT-mask summary')
    p.add_argument('--selection-json', help='Completed probe selection.json; otherwise fixed random candidates')
    p.add_argument('--teacher-mode', choices=common.MODES, default='normal')
    p.add_argument('--head-selection', choices=('selected', 'random-matched'), default='selected')
    p.add_argument('--gate', choices=('teacher-better', 'dynamic', 'none'), default='teacher-better')
    p.add_argument('--max-attempts', type=int, default=0,
                   help='teacher-better: candidate cap; 0 scans the entire pool, max-samples is ACCEPTED target')
    p.add_argument('--student-total-tokens', type=int, default=4096)
    p.add_argument('--student-min-tokens', type=int, default=4)
    p.add_argument('--teacher-total-tokens', type=int, default=14336)
    p.add_argument('--teacher-min-tokens', type=int, default=64)
    p.add_argument('--max-frames', type=int, default=0,
                   help='teacher-better direct native pipeline: common frame cap, 0 uses decoder default')
    p.add_argument('--student-target-iou', type=float, default=1., help='Legacy dynamic gate only')
    p.add_argument('--teacher-min-iou', type=float, default=0., help='Legacy dynamic gate only')
    p.add_argument('--iou-margin', type=float, default=0., help='Legacy dynamic gate only')
    p.add_argument('--privilege-mix', type=float, default=1., help='q=(1-lambda)q_normal+lambda q_privileged')
    p.add_argument('--lr', type=float, default=1e-6)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--alpha', type=float, default=16.)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--max-grad-norm', type=float, default=1.)
    p.add_argument('--save-every-steps', type=int, default=25)
    p.add_argument('--offload-saved-tensors', action='store_true', help='Offload student saved activations to CPU RAM')
    p.add_argument('--skip-merge', action='store_true')
    return p


def decide_gate(student, teacher, a):
    if a.gate == 'none':
        return True, 'ungated'
    if a.gate == 'teacher-better':
        if not teacher['valid']:
            return False, 'teacher_parse_failure'
        s, t = (student['iou'] if student['valid'] else 0.), teacher['iou']
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in (s, t)):
            raise ValueError('Invalid IoU passed to strict teacher-better gate')
        # Deliberately no minimum teacher IoU, minimum gain, or student ceiling.
        return (True, 'teacher_strictly_better') if t > s else (False, 'teacher_not_better')
    if student['valid'] and student['iou'] >= a.student_target_iou:
        return False, 'student_already_sufficient'
    if not teacher['valid']:
        return False, 'teacher_parse_failure'
    if teacher['iou'] < a.teacher_min_iou:
        return False, 'teacher_below_minimum'
    if teacher['iou'] <= student['iou'] or teacher['iou']-student['iou']+1e-12 < a.iou_margin:
        return False, 'teacher_gain_insufficient'
    return True, 'accepted'


def prepare(a):
    better = a.gate == 'teacher-better'
    protocol, samples = common.validate_common(a, sample_limit=0 if better else None)
    if a.max_attempts < 0 or a.max_frames < 0 or a.max_frames % 2:
        raise ValueError('Invalid max-attempts or max-frames')
    for role in ('student', 'teacher'):
        cap, lower = getattr(a, role+'_total_tokens'), getattr(a, role+'_min_tokens')
        if not 0 < lower <= cap:
            raise ValueError('Invalid independent budget: '+role)
    if better:
        if a.teacher_mode not in ('normal','gt_clip','gt_clip_blank') or a.selection_json or a.epochs != 1 or a.privilege_mix != 1:
            raise ValueError('teacher-better supports normal or GT-clip teacher, online selection, one pass, and one frozen teacher')
        if (a.total_tokens, a.min_tokens) != (14336, 64):
            raise ValueError('teacher-better requires explicit --student/teacher-total-tokens and -min-tokens; shared budget flags are legacy')
        if (a.student_target_iou, a.teacher_min_iou, a.iou_margin) != (1., 0., 0.):
            raise ValueError('teacher-better has no IoU floor/margin/student ceiling; remove legacy threshold flags')
        if a.teacher_mode in ('gt_clip','gt_clip_blank') or getattr(a, '_probe_pool_input', False):
            if (a.student_total_tokens, a.teacher_total_tokens, a.total_tokens) != (14336,14336,14336):
                raise ValueError('GT-clip OPD currently requires matched student/teacher caps of 14336')
            if (a.student_min_tokens, a.teacher_min_tokens, a.min_tokens) != (64,64,64) or a.max_frames != 0:
                raise ValueError('GT-clip OPD preserves the probe input: min64, dense-pool sampling, max-frames=0; use max-pool-frames')
        if a.max_attempts:
            samples = samples[:a.max_attempts]
        if len(samples) < a.max_samples:
            raise ValueError('Candidate pool smaller than the requested accepted target')
    if (min(a.rank, a.grad_accum, a.epochs) < 1 or a.save_every_steps < 0
            or not all(math.isfinite(x) and x > 0 for x in (a.lr, a.alpha, a.max_grad_norm))):
        raise ValueError('Invalid optimizer/count settings')
    for key in ('student_target_iou', 'teacher_min_iou', 'iou_margin', 'privilege_mix'):
        if not math.isfinite(getattr(a, key)) or not 0 <= getattr(a, key) <= 1:
            raise ValueError('Expected [0,1]: '+key)
    if a.teacher_mode == 'normal' and a.privilege_mix != 1:
        raise ValueError('normal teacher uses privilege-mix=1 (no duplicate branch)')
    from train_timelens8b_gtmask_opsd_k100_dp2 import read_heads
    selected = read_heads(a.analysis_summary, a.student_path)
    if len(selected) != 10:
        raise ValueError('Exactly ten student GT-mask heads required, not probe Top30')
    heads = selected
    config = protocol['student_checkpoint']['config']['text_config']
    if a.head_selection == 'random-matched':
        rng = random.Random(a.seed)
        occupied = {(h['layer'], h['head']) for h in selected}
        counts = common.Counter(h['layer'] for h in selected)
        heads = [dict(layer=layer, head=head) for layer, n in sorted(counts.items()) for head in
                 rng.sample([j for j in range(config['num_attention_heads']) if (layer, j) not in occupied], n)]
    roles = {}
    if a.selection_json:
        path = Path(a.selection_json)
        manifest = common.read(path)
        previous = common.read(path.parent/'protocol.json')
        summary = common.read(path.parent/'summary.json')
        if (not manifest.get('selection_complete') or not summary.get('probe_complete')
                or manifest.get('probe_protocol_sha256') != common.sha(path.parent/'protocol.json')):
            raise ValueError('Incomplete or modified probe selection')
        prompt_mode = manifest['teacher_mode']
        if (prompt_mode in ('gt_evidence_guided', 'gt_evidence_marked', 'gt_clip', 'gt_clip_blank')
                and previous.get('teacher_prompt_policy', {}).get(prompt_mode)
                    != protocol['teacher_prompt_policy'][prompt_mode]):
            raise ValueError('Evidence prompt/detection policy differs from the selection probe')
        for key in ('teacher_checkpoint', 'student_checkpoint', 'annotation_sha256', 'processor_files'):
            if protocol[key] != previous[key]:
                raise ValueError('Selection model/data identity differs: '+key)
        for key in ('fps', 'total_tokens', 'min_tokens', 'pool_factor', 'max_pool_frames',
                    'boundary_seconds', 'max_new_tokens', 'seed', 'video_dir',
                    'teacher_processor_path', 'student_processor_path'):
            if getattr(a, key) != previous['arguments'][key]:
                raise ValueError('Selection input/generation protocol differs: '+key)
        if prompt_mode in ('gt_clip', 'gt_clip_blank') or a.teacher_mode in ('gt_clip', 'gt_clip_blank'):
            for key in ('clip_total_tokens','clip_fps','clip_max_frames','clip_padding_seconds'):
                if getattr(a,key) != previous['arguments'].get(key):
                    raise ValueError('Selection GT-clip input protocol differs: '+key)
        entries = manifest['samples']
        if len(entries) != a.max_samples:
            raise ValueError('--max-samples must equal the fixed selected dataset size')
        if dict(common.Counter(e['role'] for e in entries)) != {k:v for k,v in manifest['requested'].items() if v}:
            raise ValueError('Manifest role counts differ from its requested quotas')
        samples = [e['sample'] for e in entries]
        keys = [(s['video_path'], s['query'], s['gt_start'], s['gt_end']) for s in samples]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicated selected event')
        # Verify manifest entries and initial scores against the actual paired records.
        paired = [json.loads(line) for line in (path.parent/'paired_results.jsonl').read_text(encoding='utf-8').splitlines() if line]
        available = {(r['sample']['video_path'], r['sample']['query'], r['sample']['gt_start'], r['sample']['gt_end']): r for r in paired}
        mode = manifest['teacher_mode']
        for entry, key in zip(entries, keys):
            record = available.get(key)
            if record is None or record['sample'] != entry['sample']:
                raise ValueError('Manifest event absent from paired probe')
            t, s = record['teacher'][mode], record['student']
            if not t['valid'] or t['iou'] < .7 or entry['initial_teacher_iou'] != t['iou'] or entry['initial_student_iou'] != s['iou']:
                raise ValueError('Manifest initial prediction/teacher threshold mismatch')
            if not ((entry['role'] == 'anchor' and s['valid'] and s['iou'] >= .7)
                    or (entry['role'] == 'frontier' and s['iou'] < .3)):
                raise ValueError('Manifest role does not match initial student')
            roles[key] = entry['role']
        protocol.update(selection_sha256=common.sha(path), selection_teacher_mode=mode,
                        selection_note='Fixed samples across teacher ablations; current teacher mode may differ')
    dim = config.get('head_dim', config['hidden_size']//config['num_attention_heads'])
    counts = common.Counter(h['layer'] for h in heads)
    protocol.update(method='qwen32b_to_timelens8b_gtmask10_full_vocab_rkl_k100_v1', selected_heads=heads,
        original_gtmask_heads=selected, analysis_sha256=common.sha(a.analysis_summary),
        merge_protocol_sha256=common.sha(Path(a.analysis_summary).parent/'merge_protocol.json'),
        expected_trainable_parameters=2*a.rank*sum(config['hidden_size']+n*dim for n in counts.values()),
        objective='mean token full-vocabulary KL(student || stopgrad(teacher)), on sampled student prefixes',
        gate=dict(mode=a.gate, decision_rule='Independent greedy student/teacher predictions; '
             'GT only for training gates. Skipped events logged, not replaced; '
             'anchors checked again each epoch. none disables all greedy gate predictions.'),
        placement='student cuda:0; frozen teacher sharded over cuda:0/1; sequential forward/backward',
        inference='Exported student uses normal video/query only; no teacher or GT needed',
        resume='Adapter snapshots only; optimizer/RNG resume not implemented')
    if better:
        protocol.update(method='qwen32b_to_timelens8b_independent_budget_strict_better_opd_v2',
            candidate_count=len(samples), accepted_target=a.max_samples,
            candidate_samples_sha256=hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest(),
            sample_count_semantics='max-samples counts unique accepted updates; max-attempts caps candidates; no repeated events',
            video_protocol='Two direct native normal-video branches with independent spatial budgets; '
                           'common fps/frame cap, uniform complete-group thinning if needed; '
                           'record source frame agreement and both actual token counts',
            privilege='Teacher receives a larger normal-video budget; GT is used only in the online IoU comparison',
            teacher_prompt_policy=dict(student='official normal video/query', teacher='official normal video/query', gt_text=False),
            independent_budgets=dict(student=dict(total_tokens=a.student_total_tokens, min_tokens=a.student_min_tokens),
                                     teacher=dict(total_tokens=a.teacher_total_tokens, min_tokens=a.teacher_min_tokens),
                                     fps=a.fps, max_frames=a.max_frames),
            head_selection_transfer=f'Reuse GT-mask Top10 selected at 14336 for the {a.student_total_tokens}-token student; '
                                    'not re-ranked under the new input protocol',
            gate=dict(mode='teacher-better', decision_rule='teacher valid AND teacher greedy IoU > CURRENT student greedy IoU; '
                      'invalid student counts as zero; ties rejected; no floor/margin/student ceiling',
                      failure_policy='Rejected events logged and skipped; scan further until accepted target or candidate exhaustion'))
        if a.teacher_mode in ('gt_clip','gt_clip_blank'):
            protocol.update(method='qwen32b_gt_clip_to_timelens8b_gtmask10_matched14336_strict_better_opd_v1',
                video_protocol='Same as GT-clip probe: common dense pool; student normal uniform global view; '
                               'teacher uniform global context plus independently decoded GT reference clip; '
                               'exact matched actual visual tokens, both capped at 14336',
                privilege='GT selects the teacher reference clip and validates greedy IoU; student receives only normal video/query',
                teacher_prompt_policy=dict(student='official normal video/query',
                                          teacher=common.GT_CLIP_GUIDANCE, teacher_mode=a.teacher_mode,
                                          gt_answer_text=False, original_source_timestamps=True),
                clip_policy=dict(maximum_visual_tokens=a.clip_total_tokens, fps=a.clip_fps,
                                 max_frames=a.clip_max_frames, padding_seconds=a.clip_padding_seconds,
                                 included_in_teacher_cap=True, exactly_matches_student_visual_tokens=True),
                head_selection_transfer='Existing TimeLens-8B GT-mask Top10 at 14336; reused with probe dense-pool sampling, not re-ranked',
                shortcut_note='GT crop plus absolute timestamps exposes target location; teacher gains are not proof of visual reasoning or student gains')
        elif getattr(a, '_probe_pool_input', False):
            protocol.update(method='qwen32b_normal_to_timelens8b_gtmask10_matched14336_probe_pool_control_v1',
                video_protocol='Same dense-pool normal global view and matched actual visual tokens as the GT-clip OPD student',
                privilege='No teacher-only GT input; GT used only for strict greedy-IoU selection',
                head_selection_transfer='Same existing GT-mask Top10 and student probe-pool input as GT-clip OPD')
    protocol['source_sha256'][Path(__file__).name] = common.sha(__file__)
    protocol['source_sha256']['train_timelens8b_gtmask_opsd_k100_dp2.py'] = common.sha(
        common.ROOT/'scripts/train_timelens8b_gtmask_opsd_k100_dp2.py')
    if getattr(a, '_entry_file', None):
        entry = Path(a._entry_file)
        protocol['source_sha256'][entry.name] = common.sha(entry)
    return protocol, samples, roles


def digest(params):
    value = hashlib.sha256()
    for name, param in params:
        value.update(name.encode())
        value.update(param.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def training_inputs(processors, sample, a):
    """Prepare the actual branches used by BOTH greedy gates and OPD scoring."""
    pool = teacher_input = teacher_audit = None
    if a.gate == 'teacher-better' and a.teacher_mode == 'normal' and not getattr(a, '_probe_pool_input', False):
        student_input, student_audit = common.normal_input(processors[0], sample,
            a.student_total_tokens, a.student_min_tokens, a.fps, a.max_frames)
        teacher_input, teacher_audit = common.normal_input(processors[1], sample,
            a.teacher_total_tokens, a.teacher_min_tokens, a.fps, a.max_frames)
        if student_audit['user_prompt'] != teacher_audit['user_prompt']:
            raise RuntimeError('Normal teacher/student query prompts unexpectedly differ')
    else:
        pool = common.decode_pool(processors[0], sample, a)
        student_input, student_audit = common.build_input(processors[0], sample, pool, 'normal', a)
        if a.gate == 'teacher-better':
            teacher_input, teacher_audit = common.build_input(processors[1], sample, pool, a.teacher_mode, a)
            s_tokens, t_tokens = student_audit['actual_visual_tokens'], teacher_audit['actual_visual_tokens']
            if not (0 < s_tokens == t_tokens <= a.student_total_tokens == a.teacher_total_tokens):
                raise RuntimeError('GT-clip student/teacher actual visual budgets do not match')
    duration = student_audit['duration'] if pool is None else pool['duration']
    if (teacher_audit is not None and abs(duration-teacher_audit['duration']) > 1e-6
            or not 0 <= sample['gt_start'] < sample['gt_end'] <= duration):
        raise ValueError('Teacher/student duration mismatch or invalid training GT')
    return student_input, teacher_input, student_audit, teacher_audit, pool, duration


def main(argv=None):
    values = sys.argv[1:] if argv is None else list(argv)
    if any(x == "--resume-snapshot" or x.startswith("--resume-snapshot=") for x in values):
        return resume_main(values)
    return run(parser().parse_args(values))


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
                for device in range(2):
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
            for device in range(2):
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


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def preimage_for_shuffle(order,seed):
    indices = list(range(len(order)))
    random.Random(seed).shuffle(indices)
    result = [None]*len(order)
    for output_index,input_index in enumerate(indices):
        result[input_index] = order[output_index]
    check = list(result)
    random.Random(seed).shuffle(check)
    if check != order:
        raise RuntimeError('Candidate shuffle inversion failed')
    return result


def inspect_snapshot(snapshot, output, target):
    snapshot = Path(snapshot).resolve()
    parent = snapshot.parent
    output = Path(output).resolve()
    if output == parent or parent in output.parents or (output.exists() and any(output.iterdir())):
        raise ValueError('Use a new empty directory outside the original training directory')
    saved = read(snapshot/'adapter_config.json')
    meta = read(parent/'training_metadata.json')
    progress = read(snapshot/'summary.json')
    previous = meta['arguments']
    for key in ('student_path','teacher_path','anno_json','video_dir','analysis_summary','seed','rank','alpha',
                'gate','teacher_mode','student_total_tokens','teacher_total_tokens','grad_accum','epochs'):
        if saved['arguments'].get(key) != previous.get(key):
            raise ValueError('Snapshot and parent metadata differ: '+key)
    if (previous.get('gate') != 'teacher-better' or previous.get('teacher_mode') != 'normal'
            or previous.get('epochs') != 1 or previous.get('selection_json')
            or previous.get('privilege_mix') != 1
            or (previous.get('student_total_tokens'),previous.get('teacher_total_tokens')) != (4096,14336)):
        raise ValueError('This entry is only for the original normal-teacher 4096/14336 strict-better run')
    accepted,processed,steps = (int(progress[k]) for k in ('accepted','processed','steps'))
    if (accepted <= 0 or processed < accepted or steps <= 0 or accepted != steps*int(previous['grad_accum'])
            or target <= accepted):
        raise ValueError('Snapshot must be at a complete accumulation boundary; target must exceed accepted count')
    if not (snapshot/'adapter.pt').is_file():
        raise FileNotFoundError('Snapshot adapter.pt missing')
    samples = read(parent/'candidate_samples.json')
    sample_hash = hashlib.sha256(json.dumps(samples,sort_keys=True).encode()).hexdigest()
    if sample_hash != meta.get('candidate_samples_sha256'):
        raise ValueError('Original candidate sample list differs from training metadata')
    order = list(samples)
    random.Random(previous['seed']).shuffle(order)
    if processed >= len(order):
        raise ValueError('No candidates remain after the snapshot')
    prior = []
    with (parent/'accepted_samples.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if len(prior) >= accepted:
                break
            item = json.loads(line)
            if (item['accepted_index'] != len(prior)+1 or not 1 <= item['candidate'] <= processed
                    or item['sample'] != order[item['candidate']-1]
                    or (prior and item['candidate'] <= prior[-1]['candidate'])):
                raise ValueError('Accepted-event log does not agree with original candidate order')
            prior.append(item)
    if len(prior) != accepted or prior[-1]['candidate'] != processed:
        raise ValueError('Snapshot boundary is not confirmed by the accepted-event log')
    return saved,meta,progress,prior,order[processed:]


class ResumeRecipe:
    def __init__(self,saved,meta,progress,prior,remaining,snapshot):
        self.saved,self.meta,self.progress,self.prior,self.remaining,self.snapshot = saved,meta,progress,prior,remaining,snapshot

    def prepare(self,a):
        # Run ordinary checks against the original current model/data/head inputs.
        protocol, _, roles = prepare(a)
        for key in ('student_checkpoint','teacher_checkpoint','annotation_sha256','analysis_sha256',
                    'merge_protocol_sha256','selected_heads','expected_trainable_parameters'):
            if protocol.get(key) != self.meta.get(key) or protocol.get(key) != self.saved.get(key):
                raise ValueError('Original model/head/data identity mismatch: '+key)
        if len(self.remaining) < a.max_samples:
            raise ValueError('Remaining candidates fewer than requested additional accepted samples')
        # engine.run shuffles once; supply the inverse so its traversal exactly
        # matches the original shuffled sequence after the saved boundary.
        samples = preimage_for_shuffle(self.remaining,a.seed)
        protocol.update(method='timelens8b_qwen32b_weightsonly_continuation_v1',
            continuation=dict(snapshot=str(self.snapshot),parent_accepted=self.progress['accepted'],
                parent_processed=self.progress['processed'],parent_optimizer_steps=self.progress['steps'],
                additional_accepted_target=a.max_samples,
                total_accepted_target=int(self.progress['accepted'])+a.max_samples,
                candidate_order='original saved shuffled order following snapshot boundary',
                optimizer='new AdamW; optimizer/RNG states not stored in original snapshot',
                note='Weights-only continuation, not exact optimizer/RNG resume'),
            candidate_count=len(samples),
            candidate_samples_sha256=hashlib.sha256(json.dumps(samples,sort_keys=True).encode()).hexdigest())
        return protocol,samples,roles

    def prepare_alignment(self,student,teacher,processors):
        common.tokenizer_equivalence(processors[0].tokenizer,processors[1].tokenizer)
        if student.config.text_config.vocab_size != teacher.config.text_config.vocab_size:
            raise ValueError('Output vocabulary dimensions differ')
        return dict(mode='exact_full_vocabulary',note='Same student/teacher prefix and tokenizer')

    generate = staticmethod(common.generate)
    training_inputs = staticmethod(training_inputs)
    teacher_targets = staticmethod(lambda y:y)
    select_logits = staticmethod(lambda logits,role:logits)


def restore_adapters(adapters,path):
    import torch
    saved = torch.load(path,map_location='cpu',weights_only=True)
    if set(saved) != set(adapters):
        raise ValueError('Snapshot adapter names differ from selected-head modules')
    with torch.no_grad():
        for name,adapter in adapters.items():
            value = saved[name]
            if (not torch.equal(value['indices'].cpu(),adapter.indices.cpu())
                    or value['A'].shape != adapter.A.shape or value['B'].shape != adapter.B.shape):
                raise ValueError('Snapshot head indices or adapter shapes differ: '+name)
            adapter.A.copy_(value['A'].to(adapter.A.device,dtype=adapter.A.dtype))
            adapter.B.copy_(value['B'].to(adapter.B.device,dtype=adapter.B.dtype))


def resume_main(argv=None):
    p = argparse.ArgumentParser(description="Continue from a weights-only adapter snapshot; optimizer and RNG restart")
    p.add_argument("--resume-snapshot", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--total-accepted-target", type=int, default=1000)
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args(argv)
    saved, meta, progress, prior, remaining = inspect_snapshot(args.resume_snapshot, args.output_dir, args.total_accepted_target)
    old = dict(meta["arguments"])
    old.update(output_dir=args.output_dir, max_samples=args.total_accepted_target-int(progress["accepted"]), plan_only=args.plan_only)
    a = argparse.Namespace(**old)
    recipe = ResumeRecipe(saved, meta, progress, prior, remaining, str(Path(args.resume_snapshot).resolve()))
    import train_timelens_three_group_opsd as core
    original = core.attach_head_lora
    def attach(model, heads, rank, alpha, targets):
        adapters = original(model, heads, rank, alpha, targets)
        restore_adapters(adapters, Path(args.resume_snapshot)/"adapter.pt")
        return adapters
    core.attach_head_lora = attach
    try:
        return run(a, recipe=recipe)
    finally:
        core.attach_head_lora = original


