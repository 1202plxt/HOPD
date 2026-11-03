"""Auditable privileged Top10 Head-OPSD setup; shares the existing training loop.

Current self-teacher is the default: same model and current adapters, with
teacher logits detached. frozen_base explicitly reproduces the legacy variant.
Only training annotations supply GT. Online teacher/student IoU gating is the
default; exact GT text and experimental teacher-only frame marks are separate
privilege modes. A four-arm cue probe performs inference without training.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import train_timelens_three_group_opsd as core


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if hasattr(value, 'tolist'):
        return json_value(value.tolist())
    return value


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(json_value(value), ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def gate_decision(student, teacher, minimum, margin):
    if student['parse_failure'] or teacher['parse_failure']:
        return False, 'parse_failure'
    if teacher['rollout_iou'] < minimum:
        return False, 'teacher_below_minimum'
    gain = teacher['rollout_iou'] - student['rollout_iou']
    if gain <= 0 or gain + 1e-12 < margin:
        return False, 'insufficient_gain'
    return True, 'teacher_improves'


def greedy_answer(model, processor, inputs, args, adapters=None, teacher=False):
    import contextlib
    import torch
    validate_context(model, inputs, args)
    model.eval()
    core.reset_rope(model)
    context = core.self_teacher_context(args, adapters) if teacher else contextlib.nullcontext()
    with torch.no_grad(), context:
        generated = model.generate(**inputs, do_sample=False, num_beams=1,
            num_return_sequences=1, max_new_tokens=args.max_new_tokens,
            repetition_penalty=1., temperature=None, top_p=None, top_k=None,
            use_cache=True, pad_token_id=processor.tokenizer.pad_token_id)
    suffix = generated[0, inputs['input_ids'].shape[1]:]
    if suffix.numel() == 0:
        raise RuntimeError('Empty greedy gate response')
    return processor.tokenizer.decode(suffix.tolist(), skip_special_tokens=True)


def validate_context(model, inputs, args):
    config = getattr(model.config, 'text_config', model.config)
    maximum = getattr(config, 'max_position_embeddings', None)
    if maximum is None or inputs['input_ids'].shape[1] + args.max_new_tokens > maximum:
        raise ValueError('Prompt plus generation exceeds known context budget; never truncate/remap frame IDs')


def evaluate_gate(model, processor, student, teacher, sample, args, adapters):
    s_text = greedy_answer(model, processor, student, args)
    t_text = greedy_answer(model, processor, teacher, args, adapters, teacher=True)
    s = core.rollout_metrics(s_text, sample, args.prompt_mode)
    t = core.rollout_metrics(t_text, sample, args.prompt_mode)
    accepted, reason = gate_decision(s, t, args.teacher_min_iou, args.iou_margin)
    return dict(accepted=accepted, reason=reason,
                student=dict(output=s_text, **s), teacher=dict(output=t_text, **t),
                iou_gain=t['rollout_iou'] - s['rollout_iou'],
                note='Independent greedy answers for gating; accepted sample gets a fresh sampled student rollout')


def parser(family):
    p = core.build_parser()
    p.description = f'{family}: Top10 query-head Q/O, gated privileged on-policy self-distillation'
    p.set_defaults(model_family=family, teacher_mode='gt_text', prompt_mode='official',
                   video_processor_mode='legacy', head_groups=['gt_positive'],
                   gt_positive_k=10, gt_negative_k=0, off_positive_k=0,
                   targets=['q_proj', 'o_proj'], max_samples=2000, epochs=1,
                   grad_accum=4, rank=8, alpha=16., lr=2e-5, fps=2.,
                   min_tokens=4, total_tokens=4096, max_frames=1024,
                   max_new_tokens=80, save_every=200, self_teacher_policy='current')
    p.add_argument('--plan-only', action='store_true',
                   help='CPU validation only; creates no output and does not load weights')
    p.add_argument('--decode-timeout', type=float, default=600.)
    p.add_argument('--decode-attempts', type=int, default=3)
    p.add_argument('--no-iou-gate', dest='iou_gate', action='store_false', default=True,
                   help='Historical ungated GT-text control only; frame_hint requires the online gate')
    p.add_argument('--teacher-min-iou', type=float, default=.5)
    p.add_argument('--iou-margin', type=float, default=.2)
    p.add_argument('--max-attempts', type=int, default=0,
                   help='0 scans the full candidate pool; max-samples counts accepted trained events')
    p.add_argument('--probe-frame-hints', action='store_true',
                   help='Four-arm cue-utility probe only; no training or checkpoint export')
    p.add_argument('--probe-samples', type=int, default=40)
    return p


def model_layout(path, family):
    path = Path(path)
    config = json.loads((path / 'config.json').read_text(encoding='utf-8-sig'))
    expected = {'qwen25vl': 'qwen2_5_vl', 'qwen3vl': 'qwen3_vl'}[family]
    if config.get('model_type') != expected:
        raise ValueError(f'Expected {expected} student config')
    text = config.get('text_config') or config
    layout = {key: int(text[key]) for key in
              ('hidden_size', 'num_attention_heads', 'num_hidden_layers')}
    if min(layout.values()) <= 0:
        raise ValueError('Invalid head layout')
    if 'head_dim' not in text and layout['hidden_size'] % layout['num_attention_heads']:
        raise ValueError('Cannot infer an integral head dimension')
    layout['head_dim'] = int(text.get('head_dim',
        layout['hidden_size'] // layout['num_attention_heads']))
    if layout['head_dim'] <= 0:
        raise ValueError('Invalid head dimension')
    index = path / 'model.safetensors.index.json'
    files = (set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values())
             if index.is_file() else {'model.safetensors'})
    if not files:
        raise ValueError('Empty checkpoint weight index')
    for name in files:
        shard = (path / name).resolve()
        if not shard.is_relative_to(path.resolve()) or not shard.is_file():
            raise FileNotFoundError(f'Missing/invalid model shard: {name}')
    return layout


def validate_screen(args, layout):
    data = json.loads(Path(args.head_json).read_text(encoding='utf-8-sig'))
    meta = data.get('_meta', {})
    cfg = meta.get('intervention_config', {})
    collection = data.get('collection_summary', {})
    if (not data.get('selection_ready') or not meta.get('all_requested_work_complete')
            or data.get('n_complete_samples') != 500
            or not collection.get('complete') or not collection.get('target_reached')
            or collection.get('accepted_nonzero') != 500):
        raise ValueError('Requires a completed 500-sample nonzero-baseline knockout screen')
    if (cfg.get('model_family') != args.model_family or cfg.get('mode') != 'zero'
            or cfg.get('scope') != 'all' or cfg.get('video_processor_mode') != 'legacy'
            or cfg.get('sample_collection') != 'baseline_iou_gt_zero_until_target_v1'):
        raise ValueError('Wrong student family/knockout/input/collection protocol')
    if not cfg.get('model_path') or Path(cfg['model_path']).resolve() != Path(args.model_path).resolve():
        raise ValueError('Screen checkpoint differs from the training student')
    for key in ('fps', 'min_tokens', 'total_tokens', 'max_frames'):
        if cfg.get(key) != getattr(args, key):
            raise ValueError('Screen/training budget differs: ' + key)
    if args.model_family == 'qwen25vl':
        core.validate_qwen25_screen(args.head_json, args)
    _, heads = core.load_head_groups(args.head_json, 10, 0, 0, {'gt_positive'},
                                     args.min_effect, args.require_ci)
    all_rows = data['head_results']
    results = {(row['layer'], row['head']): row for row in all_rows}
    if len(results) != len(all_rows):
        raise ValueError('Duplicate coordinates in head_results')
    for row in heads:
        l, h = row['layer'], row['head']
        if (not isinstance(l, int) or not isinstance(h, int)
                or not 0 <= l < layout['num_hidden_layers']
                or not 0 <= h < layout['num_attention_heads']
                or results.get((l, h), {}).get('delta_miou') != row['delta_miou']):
            raise ValueError('Invalid/out-of-range/inconsistent selected head')
    return heads


def runtime_sources(entry):
    """Report deployment omissions before scanning annotations or loading CUDA."""
    source = [Path(__file__), Path(entry), Path(core.__file__),
              ROOT / 'scripts/qwen_decode_worker.py', ROOT / 'scripts/qwen25_head_inputs.py',
              ROOT / 'scripts/qwen_frame_hints.py']
    missing = [str(path) for path in source if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            'Missing OPSD runtime files. Upload the local same-named .py files '
            'to the server scripts directory before launching:\n  ' + '\n  '.join(missing))
    return source


def prepare(args, family, entry):
    source = runtime_sources(entry)
    if getattr(args, 'warm_start_snapshot', None) and not getattr(args, 'analysis_summary', None):
        raise ValueError('Use train_qwen25_head_opsd_14336.py for weights-only snapshot continuation')
    if (args.model_family != family or args.teacher_mode not in ('gt_text', 'frame_hint')
            or args.prompt_mode != 'official' or args.video_processor_mode != 'legacy'
            or args.head_groups != ['gt_positive'] or args.gt_positive_k != 10
            or args.gt_negative_k or args.off_positive_k or args.explicit_heads
            or not args.head_json or args.expected_head_count
            or set(args.targets) != {'q_proj', 'o_proj'} or len(args.targets) != 2
            or args.opsd_token_scope != 'all' or args.time_ce_weight != 0
            or not args.anno_json or args.selection_dir):
        raise ValueError('This entry requires student-knockout Top10, Q/O and full-answer privileged KL only')
    if (args.teacher_mode == 'frame_hint' and not args.iou_gate) or (args.iou_gate and args.epochs != 1):
        raise ValueError('frame_hint requires online gating; this gated entry supports one epoch only')
    if (args.max_attempts < 0 or args.probe_samples < 1 or not 0 <= args.teacher_min_iou <= 1
            or not 0 <= args.iou_margin <= 1):
        raise ValueError('Invalid gate thresholds or attempt cap')
    if args.probe_frame_hints and args.teacher_mode != 'frame_hint':
        raise ValueError('The four-arm probe requires --teacher-mode frame_hint')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Use one process; do not use torchrun')
    if (min(args.max_samples, args.epochs, args.grad_accum, args.rank,
            args.max_new_tokens, args.min_tokens, args.total_tokens) < 1
            or args.max_frames < 0 or args.max_frames % 2
            or args.min_tokens > args.total_tokens or args.save_every < 0
            or (args.save_every and args.save_every % args.grad_accum)
            or args.decode_attempts < 1 or not 0 < args.top_p <= 1):
        raise ValueError('Invalid count/budget/accumulation/generation arguments')
    for key in ('lr', 'alpha', 'temperature', 'fps', 'max_grad_norm',
                'decode_timeout', 'gpu_mem_gib'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError('Invalid positive hyperparameter: ' + key)
    for key in ('min_effect', 'kl_clip', 'gpu0_mem_gib'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            raise ValueError('Invalid nonnegative hyperparameter: ' + key)
    if args.no_gradient_checkpointing:
        raise ValueError('Keep gradient checkpointing enabled for this recipe')
    out = Path(args.output_dir).resolve()
    if out == Path(args.model_path).resolve() or (out.exists() and any(out.iterdir())):
        raise FileExistsError('Use a new empty output directory; training resume is not implemented')
    if not Path(args.video_dir).is_dir():
        raise FileNotFoundError(args.video_dir)
    layout = model_layout(args.model_path, family)
    heads = validate_screen(args, layout)
    samples, stats = core.load_samples(args.anno_json, args.video_dir,
                                       0 if args.iou_gate else args.max_samples, args.seed)
    if len(samples) < args.max_samples:
        raise ValueError('Candidate pool is smaller than the requested accepted target')
    per_layer = {}
    for row in heads:
        per_layer[row['layer']] = per_layer.get(row['layer'], 0) + 1
    expected = 2 * args.rank * sum(layout['hidden_size'] + n * layout['head_dim']
                                  for n in per_layer.values())
    teacher = ('current model including current HeadLoRA; teacher logits stop-gradient; no EMA'
               if args.self_teacher_policy == 'current' else
               'initial frozen base; HeadLoRA disabled for teacher logits')
    return dict(method=f'{family}_top10_qo_{args.teacher_mode}_opsd_{args.self_teacher_policy}_gated{int(args.iou_gate)}_v2',
                teacher_policy=teacher, layout=layout, selected_heads=heads,
                head_selection=dict(path=str(Path(args.head_json).resolve()),
                                    sha256=sha(args.head_json),
                                    ranking='positive baseline-minus-knockout delta descending; layer/head ties'),
                expected_trainable_parameters=expected, training_events=args.max_samples,
                candidate_events=len(samples),
                annotation_sha256=sha(args.anno_json), data_stats=stats,
                selected_events_sha256=hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest(),
                checkpoint_config_sha256=sha(Path(args.model_path) / 'config.json'),
                source_sha256={p.name: sha(p) for p in source},
                sample_policy=('full shuffled pool; greedy teacher IoU >= threshold and strictly better student by margin; fresh sampled rollout after acceptance'
                               if args.iou_gate else 'seed-shuffled unique training events; first N; no online IoU gate'),
                gate=dict(enabled=args.iou_gate, teacher_min_iou=args.teacher_min_iou,
                          iou_margin=args.iou_margin, max_attempts=args.max_attempts),
                loss='mean full-vocabulary KL(student || stopgrad(GT-context self-teacher)) on student rollouts',
                rollout=dict(temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens),
                gt_usage=('training-only frame borders selected by actual sampled temporal-group times; no exact GT text'
                          if args.teacher_mode == 'frame_hint' else
                          'training-only privileged teacher text; absent from student/evaluation prompts'),
                tensor_protocol=('same native geometry/grid/timestamps; teacher-only pixel marks after final budget sampling'
                                 if args.teacher_mode == 'frame_hint' else
                                 'same decoded frames and native pixel/grid/timestamps in both prompts'),
                placement='one shared model; single GPU or model sharding across visible GPUs',
                resume='adapter snapshots only; no optimizer/RNG/cursor recovery; never restart into same directory',
                verification='CPU preflight only; no CUDA inference or gradient test implied')


def main(family, entry, argv=None):
    args = parser(family).parse_args(argv)
    protocol = prepare(args, family, entry)
    print(json.dumps(protocol, ensure_ascii=False, indent=2), flush=True)
    if args.plan_only:
        print('PLAN PASS: no CUDA execution and no output created', flush=True)
        return 0
    from scripts import qwen_decode_worker
    args._head_opsd_protocol = protocol
    try:
        qwen_decode_worker.configure_decoder(timeout=args.decode_timeout, attempts=args.decode_attempts)
        if args.probe_frame_hints:
            from qwen_frame_hints import probe
            return probe(args, protocol)
        core.main(args)
    finally:
        if qwen_decode_worker._decoder is not None:
            qwen_decode_worker._decoder.close()
            qwen_decode_worker._decoder = None
    return 0
