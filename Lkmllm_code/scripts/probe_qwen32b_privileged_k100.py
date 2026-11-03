"""Paired training-only privilege probe; shared runtime for the 32B -> 8B OPD.

One process, two visible K100s. Frozen teacher is sharded; student is on GPU0.
No inference/training starts on import. --plan-only needs no torch installation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'scripts'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

MODES = ('normal', 'gt_evidence', 'gt_evidence_guided', 'gt_evidence_marked',
         'gt_clip', 'gt_clip_blank', 'gt_text', 'wrong_evidence', 'shuffled_evidence')
EVIDENCE_MODES = {'gt_evidence', 'gt_evidence_guided', 'gt_evidence_marked',
                  'wrong_evidence', 'shuffled_evidence'}
DENSE_EVIDENCE_GUIDANCE = (
    "\nThe video is sampled non-uniformly. Densely sampled regions provide additional "
    "visual evidence for the queried event. Pay particular attention to the content "
    "and action transitions in those regions, while using the rest of the video as "
    "global context. Compare the states before, during, and after the event to "
    "identify its onset and offset. Use the dense regions to guide your search, "
    "and determine the boundaries from visible action transitions and the displayed "
    "original-video timestamps. The first and last densely sampled frames are "
    "not necessarily the event boundaries. Return only the requested answer: "
    "The event happens in <start time> - <end time> seconds."
)
MARKED_EVIDENCE_GUIDANCE = (
    "\nEvidence map on the original video timeline:\n{evidence_map}\n"
    "Inspect the visual content and action transitions in these explicitly listed "
    "regions first. Compare their preceding and following frames to distinguish "
    "the event's onset, continuation, and offset; use the remaining video for "
    "global context. These ranges identify sampled evidence, not verified event "
    "boundaries. Determine the actual start and end from the visible actions and "
    "the original-video timestamps shown with the frames. Return only: "
    "The event happens in <start time> - <end time> seconds."
)
FOCUS_REGION_POLICY = dict(version='visible_timestamp_dense_runs_v1',
    gap_fraction_of_normal_median=.75, minimum_groups=3, maximum_regions=8,
    source='Actual rendered video-group timestamps; region detector takes no GT interval')
GT_CLIP_GUIDANCE = (
    "The first video gives full-source context. The second video is a training-only "
    "reference clip illustrating the queried event from the same source. All frame "
    "timestamps in BOTH videos use the original full-video timeline. Inspect the "
    "reference clip to identify the event's visual content, then compare the "
    "surrounding states in the full video to determine its actual onset and offset. "
    "The first and last sampled reference frames are not necessarily the exact "
    "event boundaries. Answer using the original full-video timeline and return "
    "only the requested start-end sentence."
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


def uniform_indices(count, keep):
    if not 1 <= keep <= count:
        raise ValueError('Invalid number of temporal groups')
    return [0] if keep == 1 else [i * (count - 1) // (keep - 1) for i in range(keep)]


def select_groups(times, keep, span=None, boundary_seconds=2.):
    """Half global / quarter inside / quarter boundaries; fixed unique group count.

    Works on original consecutive temporal groups, never joins two unrelated
    frames into a new temporal patch. Empty/saturated regions fall back globally.
    """
    if span is None:
        return uniform_indices(len(times), keep)
    start, end = span
    global_count = max(1, math.ceil(keep / 2))
    selected = set(uniform_indices(len(times), global_count))
    inside = [i for i, t in enumerate(times) if start <= t <= end and i not in selected]
    n = min(len(inside), keep // 4)
    if n:
        selected.update(inside[j] for j in uniform_indices(len(inside), n))
    remaining = [i for i in range(len(times)) if i not in selected]
    boundary = [i for i in remaining if min(abs(times[i]-start), abs(times[i]-end)) <= boundary_seconds]
    # Nearest boundary groups are used if a narrow window contains no sample.
    boundary = sorted(boundary or remaining,
                      key=lambda i: (min(abs(times[i]-start), abs(times[i]-end)), i))
    selected.update(boundary[:min(keep-len(selected), max(1, keep // 4))])
    remaining = [i for i in range(len(times)) if i not in selected]
    n = keep - len(selected)
    if n:
        selected.update(remaining[j] for j in uniform_indices(len(remaining), n))
    result = sorted(selected)
    if len(result) != keep:
        raise RuntimeError('Frame allocation did not preserve the matched budget')
    return result


def visible_group_times(processor, metadata):
    """Use the native processor arithmetic and the timestamps actually rendered."""
    indices = [int(i) for i in metadata['frames_indices']]
    temporal = int(processor.video_processor.temporal_patch_size)
    fps = float(metadata['fps'])
    if not indices or len(indices) % temporal or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid complete temporal groups for evidence-map timestamps')
    calculate = getattr(processor, '_calculate_timestamps', None)
    if callable(calculate):
        times = list(map(float, calculate(indices, fps, temporal)))
    else:
        times = [(indices[i]+indices[i+temporal-1])/2/fps for i in range(0, len(indices), temporal)]
    if len(times) != len(indices)//temporal or not all(map(math.isfinite, times)):
        raise ValueError('Native video-group timestamp count mismatch')
    return [float(f'{t:.1f}') for t in times]


def dense_focus_regions(times, normal_times):
    """Find explicit evidence windows from sampling density, without receiving GT."""
    if any(b < a for seq in (times, normal_times) for a, b in zip(seq, seq[1:])):
        raise ValueError('Evidence-map timestamps must be ordered')
    baseline_gaps = [b-a for a, b in zip(normal_times, normal_times[1:]) if b > a]
    if not baseline_gaps:
        return [], dict(normal_median_gap=None, dense_gap_threshold=None, detected_regions=0)
    median = statistics.median(baseline_gaps)
    threshold = median*FOCUS_REGION_POLICY['gap_fraction_of_normal_median']
    runs = []
    start = None
    def add_run(begin, end):
        if begin is not None and end-begin+1 >= FOCUS_REGION_POLICY['minimum_groups']:
            runs.append(dict(start_seconds=times[begin], end_seconds=times[end],
                             first_group_1based=begin+1, last_group_1based=end+1,
                             groups=end-begin+1))
    for i, (left, right) in enumerate(zip(times, times[1:])):
        if 0 < right-left < threshold:
            if start is None:
                start = i
        else:
            add_run(start, i)
            start = None
    add_run(start, len(times)-1)
    # Limit prompt size; retain the largest dense runs, then restore chronological order.
    chosen = sorted(sorted(runs, key=lambda r: (-r['groups'], r['start_seconds']))[
                    :FOCUS_REGION_POLICY['maximum_regions']], key=lambda r:r['start_seconds'])
    return chosen, dict(normal_median_gap=median, dense_gap_threshold=threshold, detected_regions=len(runs))


def marked_guidance(regions):
    if not regions:
        return ('\nNo clearly denser region was detected in the provided sampled video. '
                'Inspect the full video using its displayed original-video timestamps '
                'and return the requested event start and end.')
    evidence_map = '\n'.join(
        f"Focus region {i}: {r['start_seconds']:.1f} to {r['end_seconds']:.1f} seconds "
        f"(displayed video groups {r['first_group_1based']} through {r['last_group_1based']})."
        for i, r in enumerate(regions, 1))
    return MARKED_EVIDENCE_GUIDANCE.format(evidence_map=evidence_map)


def wrong_span(span, duration):
    """Deterministic farthest same-length interval; record overlap if GT is long."""
    length = span[1] - span[0]
    choices = [(0., length), (duration-length, duration)]
    return max(choices, key=lambda x: abs((x[0]+x[1]-span[0]-span[1])/2))


def parse_prediction(text, gt, duration, truncated=False):
    match = re.fullmatch(r'\s*The event happens in\s+(-?\d+(?:\.\d+)?)\s*-\s*'
                         r'(-?\d+(?:\.\d+)?)\s+seconds\.?\s*', text, re.I)
    pred = list(map(float, match.groups())) if match else None
    valid = bool(not truncated and pred and all(map(math.isfinite, pred))
                 and 0 <= pred[0] < pred[1] <= duration)
    iou = 0.
    if valid:
        intersection = max(0., min(pred[1], gt[1])-max(pred[0], gt[0]))
        iou = intersection / (pred[1]-pred[0]+gt[1]-gt[0]-intersection)
    return dict(text=text, prediction=pred, valid=valid, iou=iou, truncated=truncated)


def metrics(rows):
    values = [r['iou'] for r in rows]
    n = len(values)
    return dict(n=n, valid=sum(r['valid'] for r in rows),
                mIoU=math.fsum(values)/n if n else None,
                **{f'R@{t}': sum(v >= t for v in values)/n if n else None for t in (.3, .5, .7)})


def paired_delta(records, mode, repeats=2000):
    clusters = {}
    deltas = []
    for r in records:
        d = r['teacher'][mode]['iou'] - r['teacher']['normal']['iou']
        deltas.append(d)
        clusters.setdefault(r['sample']['video_path'], []).append(d)
    groups = [(math.fsum(ds), len(ds)) for ds in clusters.values()]
    rng = random.Random(20261031)
    bootstrap = []
    for _ in range(repeats if groups else 0):
        chosen = [groups[rng.randrange(len(groups))] for _ in groups]
        bootstrap.append(math.fsum(g[0] for g in chosen)/sum(g[1] for g in chosen))
    bootstrap.sort()
    return dict(delta_mIoU=math.fsum(deltas)/len(deltas) if deltas else None,
                improved=sum(d > 1e-12 for d in deltas), regressed=sum(d < -1e-12 for d in deltas),
                unchanged=sum(abs(d) <= 1e-12 for d in deltas),
                video_cluster_bootstrap_95=[bootstrap[int(.025*repeats)], bootstrap[int(.975*repeats)]]
                if bootstrap else None)


def add_common(p):
    for name in ('teacher-path', 'student-path', 'anno-json', 'video-dir', 'output-dir'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--teacher-processor-path', help='Explicit fallback only; default teacher checkpoint')
    p.add_argument('--student-processor-path', help='Explicit fallback only; default student checkpoint')
    p.add_argument('--max-samples', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--total-tokens', type=int, default=14336)
    p.add_argument('--min-tokens', type=int, default=64)
    p.add_argument('--fps', type=float, default=2.)
    p.add_argument('--pool-factor', type=int, default=2, help='Decode 4 FPS pool for matched 2 FPS views')
    p.add_argument('--max-pool-frames', type=int, default=1536)
    p.add_argument('--boundary-seconds', type=float, default=2.)
    p.add_argument('--clip-total-tokens', type=int, default=4096,
                   help='Maximum visual-token allocation for the second GT clip; included in total-tokens')
    p.add_argument('--clip-fps', type=float, default=4.)
    p.add_argument('--clip-max-frames', type=int, default=128)
    p.add_argument('--clip-padding-seconds', type=float, default=0., help='0 decodes exactly the GT interval')
    p.add_argument('--max-new-tokens', type=int, default=80)
    p.add_argument('--teacher-gpu0-gib', type=float, default=22.)
    p.add_argument('--teacher-gpu1-gib', type=float, default=44.)
    p.add_argument('--decode-timeout', type=float, default=600.)
    p.add_argument('--plan-only', action='store_true')


def checkpoint_identity(path, expected_family='qwen3_vl'):
    path = Path(path)
    config = read(path/'config.json')
    if config.get('model_type') != expected_family:
        raise ValueError(f'Expected {expected_family} config, got {path}: {config.get("model_type")}')
    index = path/'model.safetensors.index.json'
    shards = sorted(set(read(index)['weight_map'].values())) if index.is_file() else ['model.safetensors']
    if any(not (path/s).is_file() for s in shards):
        raise FileNotFoundError('Missing checkpoint weight shard: '+str(path))
    return dict(path=str(path.resolve()), config_sha256=sha(path/'config.json'),
                index_sha256=sha(index) if index.is_file() else None,
                auxiliary_sha256={p.name:sha(p) for p in sorted(path.iterdir())
                                  if p.is_file() and p.suffix in ('.json', '.jinja')},
                config=config, files=[dict(name=s, bytes=(path/s).stat().st_size) for s in shards],
                weight_identity_note='Config/index hashes and shard sizes; not a full weight-content hash')


def validate_common(a, sample_limit=None):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('One process only; do not use torchrun')
    if 'train' not in {s.lower() for s in Path(a.anno_json).parts}:
        raise ValueError('Privilege probing and selection require annotations under Train')
    if (min(a.max_samples, a.total_tokens, a.min_tokens, a.max_new_tokens, a.pool_factor) < 1
            or a.min_tokens > a.total_tokens or a.pool_factor < 2 or a.max_pool_frames < 4
            or a.max_pool_frames % 2):
        raise ValueError('Invalid sample/video/token counts')
    if not all(math.isfinite(x) and x > 0 for x in
               (a.fps, a.boundary_seconds, a.teacher_gpu0_gib, a.teacher_gpu1_gib, a.decode_timeout)):
        raise ValueError('Invalid positive runtime setting')
    uses_clip = bool(set(getattr(a, 'modes', ())) & {'gt_clip','gt_clip_blank'}) or getattr(a, 'teacher_mode', None) in ('gt_clip','gt_clip_blank')
    if uses_clip and (a.clip_total_tokens < 1 or a.clip_total_tokens >= a.total_tokens or a.clip_max_frames < 4
            or a.clip_max_frames % 2 or not math.isfinite(a.clip_fps) or a.clip_fps <= 0
            or not math.isfinite(a.clip_padding_seconds) or a.clip_padding_seconds < 0):
        raise ValueError('Invalid GT clip budget/fps/frame cap/padding')
    if Path(a.output_dir).exists() and any(Path(a.output_dir).iterdir()):
        raise FileExistsError('Use a new empty output directory; no implicit resume')
    teacher = checkpoint_identity(a.teacher_path)
    student = checkpoint_identity(a.student_path, getattr(a, '_student_family', 'qwen3_vl'))
    if Path(a.teacher_path).resolve() == Path(a.student_path).resolve():
        raise ValueError('Independent 32B teacher and 8B student required')
    # Validate actual local shapes rather than infer checkpoint identity from its name.
    tc = teacher['config'].get('text_config', teacher['config'])
    sc = student['config'].get('text_config', student['config'])
    if tc['hidden_size'] <= sc['hidden_size'] or tc['num_hidden_layers'] < sc['num_hidden_layers']:
        raise ValueError('Teacher config is not larger than the student; inspect actual downloaded weights')
    import train_timelens_three_group_opsd as core
    limit = a.max_samples if sample_limit is None else sample_limit
    samples, stats = core.load_samples(a.anno_json, a.video_dir, limit, a.seed)
    if not samples or (limit and len(samples) != limit):
        raise ValueError(f'Only {len(samples)} valid events, requested {limit}')
    protocol = dict(method='qwen32b_privileged_probe_and_timelens8b_opd_v1', arguments=vars(a),
        teacher_checkpoint=teacher, student_checkpoint=student, data_stats=stats,
        processor_files={role: {p.name:sha(p) for p in sorted(Path(path).iterdir())
                          if p.is_file() and p.suffix in ('.json', '.jinja')}
                         for role, path in (('student', a.student_processor_path or a.student_path),
                                            ('teacher', a.teacher_processor_path or a.teacher_path))},
        annotation_sha256=sha(a.anno_json), source_sha256={name: sha(ROOT/'scripts'/name) for name in
            (Path(__file__).name, 'train_timelens_three_group_opsd.py', 'qwen_decode_worker.py')},
        video_protocol='decode common dense pool; normal uniform groups, privileged GT groups; '
                       'equal group count, original source timestamps, actual tokens <= budget',
        privilege='Only teacher gt_text prompt or GT-guided frame selection; student normal video/query',
        teacher_prompt_policy=dict(gt_evidence_guided=DENSE_EVIDENCE_GUIDANCE,
            gt_evidence_marked=dict(template=MARKED_EVIDENCE_GUIDANCE, detection=FOCUS_REGION_POLICY),
            gt_clip=dict(guidance=GT_CLIP_GUIDANCE,
                         construction='native time-range decode, original source indices; global+clip exact matched visual tokens',
                         shortcut='GT-derived crop and absolute timestamps expose approximate target location'),
            gt_clip_blank=dict(guidance=GT_CLIP_GUIDANCE, control='identical GT clip timestamps and grid, black reference pixels'),
            note='Guided adds a general reminder; marked lists explicit sampled evidence windows. '
                 'Only gt_text directly inserts annotated start/end values.'),
        evidence_status='Code/preflight only; hardware fit and evaluation gain unverified')
    return protocol, samples


def auto_teacher_budgets(free_bytes, student_checkpoint_bytes, student_device,
                         student_reserve_gib, teacher_reserve_gib):
    """Reserve student weights/activation space and per-teacher GPU headroom."""
    count = len(free_bytes)
    if count < 2 or not 0 <= student_device < count:
        raise ValueError('Requires at least two GPUs and a valid student GPU index')
    if student_checkpoint_bytes <= 0 or not all(math.isfinite(x) and x >= 0
            for x in (student_reserve_gib, teacher_reserve_gib)):
        raise ValueError('Invalid student checkpoint size or reserve')
    weights_and_reserve = student_checkpoint_bytes + int(student_reserve_gib*1024**3)
    if free_bytes[student_device] < weights_and_reserve:
        raise RuntimeError('Student GPU cannot fit estimated student checkpoint plus configured reserve')
    return {i: max(0, free-int((student_reserve_gib if i == student_device else
                teacher_reserve_gib)*1024**3)-(student_checkpoint_bytes if i == student_device else 0))
            for i, free in enumerate(free_bytes)}


def load_runtime(a):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
    from accelerate import init_empty_weights, infer_auto_device_map
    if torch.version.cuda is None or torch.version.hip is not None or torch.cuda.device_count() < 2:
        raise RuntimeError('Requires NVIDIA CUDA PyTorch and at least two visible NVIDIA GPUs')
    torch.manual_seed(a.seed)
    processors = [AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=True,
                       padding_side='left', do_resize=False) for path in
                  (a.student_processor_path or a.student_path, a.teacher_processor_path or a.teacher_path)]
    for role, p in zip(('student', 'teacher'), processors):
        if p.tokenizer.pad_token_id is None:
            p.tokenizer.pad_token = p.tokenizer.eos_token
        if role == 'student' and getattr(a, '_student_family', None) == 'qwen2_5_vl':
            if p.__class__.__name__ != 'TimeLensProcessor' or int(p.image_processor.patch_size) != 14:
                raise ValueError('TimeLens-7B requires its native TimeLensProcessor with patch_size=14')
            continue
        vp = p.video_processor
        if tuple(int(getattr(vp, k)) for k in ('patch_size', 'merge_size', 'temporal_patch_size')) != (16, 2, 2):
            raise ValueError('This native video protocol requires patch/merge/temporal = 16/2/2')
    # Reserve enough physical space for the student before sharding the teacher.
    # The student's checkpoint shard bytes are a conservative estimate, not an
    # exact runtime size; caller-provided headroom covers activations and buffers.
    info = checkpoint_identity(a.student_path)
    student_checkpoint_bytes = sum(x['bytes'] for x in info['files'])
    student_device = int(getattr(a, 'student_device', 0))
    count = torch.cuda.device_count()
    if not 0 <= student_device < count:
        raise ValueError('student-device is outside visible GPUs')
    student_headroom = float(getattr(a, 'student_reserve_gib', 12.0))
    teacher_headroom = float(getattr(a, 'teacher_reserve_gib', 4.0))
    if not all(math.isfinite(x) and x >= 0 for x in (student_headroom, teacher_headroom)):
        raise ValueError('GPU reserve values must be finite and nonnegative')
    teacher_config = AutoConfig.from_pretrained(a.teacher_path, local_files_only=True, trust_remote_code=True)
    with init_empty_weights():
        empty = AutoModelForImageTextToText.from_config(teacher_config, trust_remote_code=True,
                                                      torch_dtype=torch.bfloat16)
    empty.tie_weights()
    free_before = [torch.cuda.mem_get_info(device)[0] for device in range(count)]
    budget = auto_teacher_budgets(free_before, student_checkpoint_bytes, student_device,
                                  student_headroom, teacher_headroom)
    print('Automatic teacher GPU weight budgets (GiB): '+json.dumps(
        {str(k):round(v/1024**3,2) for k,v in budget.items()}), flush=True)
    map_kwargs = dict(dtype=torch.bfloat16,
        no_split_module_classes=empty._no_split_modules or ['Qwen3VLDecoderLayer', 'Qwen3VLVisionBlock'])
    # Prefer keeping the student's GPU entirely free of teacher parameters.
    other_budget = {i: cap for i, cap in budget.items() if i != student_device and cap > 0}
    device_map = infer_auto_device_map(empty, max_memory=other_budget, **map_kwargs) if other_budget else {'': 'cpu'}
    if any(str(device) in ('cpu', 'disk') for device in device_map.values()):
        print('Teacher needs the reserved student GPU too; retrying with its safe budget.', flush=True)
        device_map = infer_auto_device_map(empty, max_memory=budget, **map_kwargs)
    if any(str(device) in ('cpu', 'disk') for device in device_map.values()):
        raise RuntimeError('Teacher does not fit available GPU weight budgets; CPU/disk fallback refused')
    used = {str(v).replace('cuda:', '') for v in device_map.values()}
    if not used or not used.issubset({str(i) for i in range(count)}):
        raise RuntimeError(f'Teacher device map contains unexpected devices: {used}')
    del empty
    print('Teacher device_map: '+json.dumps(device_map), flush=True)
    teacher = AutoModelForImageTextToText.from_pretrained(a.teacher_path, local_files_only=True,
        trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=device_map, attn_implementation='sdpa')
    teacher.eval().requires_grad_(False)
    student = AutoModelForImageTextToText.from_pretrained(a.student_path, local_files_only=True,
        trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={'': student_device}, attn_implementation='sdpa')
    student.eval().requires_grad_(False)
    for model in (student, teacher):
        if 'logits_to_keep' not in inspect.signature(model.forward).parameters:
            raise RuntimeError('Installed model forward must support logits_to_keep')
    import scripts.qwen_decode_worker as decoder
    decoder.configure_decoder(timeout=a.decode_timeout, attempts=3)
    print(json.dumps(dict(teacher_parameters=sum(p.numel() for p in teacher.parameters()),
                          student_parameters=sum(p.numel() for p in student.parameters()),
                          memory=memory())), flush=True)
    return student, teacher, processors


def memory():
    import torch
    return [dict(gpu=i, allocated_gib=torch.cuda.memory_allocated(i)/1024**3,
                 reserved_gib=torch.cuda.memory_reserved(i)/1024**3,
                 peak_allocated_gib=torch.cuda.max_memory_allocated(i)/1024**3,
                 free_gib=torch.cuda.mem_get_info(i)[0]/1024**3) for i in range(torch.cuda.device_count())]


def runtime_identity():
    import torch
    versions = {}
    for package in ('torch', 'transformers', 'accelerate', 'qwen-vl-utils', 'decord'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(versions=versions, hip=torch.version.hip, cuda=torch.version.cuda,
                devices=[dict(index=i, name=torch.cuda.get_device_name(i),
                              total_gib=torch.cuda.get_device_properties(i).total_memory/1024**3) for i in range(torch.cuda.device_count())])


def decode_pool(processor, sample, a):
    import scripts.qwen_decode_worker as decoder
    import train_timelens_three_group_opsd as core
    patch = int(processor.video_processor.patch_size) * int(processor.video_processor.merge_size)
    video = dict(type='video', video=sample['video_path'], fps=a.fps*a.pool_factor,
                 min_pixels=a.min_tokens*patch**2,
                 total_pixels=a.total_tokens*a.pool_factor*patch**2, max_frames=a.max_pool_frames)
    messages = core.messages_for(sample, video, prompt_mode='official')
    images, videos, options = decoder.process_vision_info(messages, image_patch_size=patch//2,
        return_video_kwargs=True, return_video_metadata=True)
    if images or len(videos) != 1:
        raise ValueError('Expected one video')
    frames, metadata = videos[0]
    metadata = copy.deepcopy(metadata)
    indices = metadata['frames_indices']
    indices = indices.tolist() if hasattr(indices, 'tolist') else list(indices)
    indices = [int(index) for index in indices]
    temporal = int(processor.video_processor.temporal_patch_size)
    if len(frames) != len(indices) or len(frames) < temporal:
        raise ValueError('Invalid dense-pool source indices')
    # Last incomplete group is removed explicitly; never silently pad a new group.
    count = len(frames)//temporal
    height, width = map(int, frames.shape[-2:])
    if height % patch or width % patch:
        raise ValueError('Unexpected decoded spatial geometry')
    per_group = (height//patch)*(width//patch)
    keep = min(count, max(1, math.ceil(count/a.pool_factor)), a.total_tokens//per_group)
    if keep < 1:
        raise ValueError('One visual group exceeds the token budget')
    fps = float(metadata['fps'])
    if (not math.isfinite(fps) or fps <= 0 or any(b < a for a, b in zip(indices, indices[1:]))
            or indices[0] < 0 or indices[-1] >= int(metadata['total_num_frames'])):
        raise ValueError('Dense-pool source timestamps are invalid')
    duration = float(metadata['total_num_frames'])/fps
    if not math.isfinite(duration) or not 0 <= sample['gt_start'] < sample['gt_end'] <= duration:
        raise ValueError('GT is outside the original video duration')
    times = [sum(indices[i*temporal:(i+1)*temporal])/temporal/fps for i in range(count)]
    return dict(frames=frames, metadata=metadata, options=options, indices=indices, times=times,
                temporal=temporal, per_group=per_group, keep=keep, duration=duration,
                decoded_frames=len(frames), dropped_tail_frames=len(frames)-count*temporal)


def normal_input(processor, sample, total_tokens, min_tokens, fps, max_frames=0):
    """Direct native video input with its OWN budget; no GT-based selection or prompt.

    Native decoder chooses time points from fps/cap. Spatial resizing uses this
    branch's budget. If still over budget, thin complete temporal groups uniformly.
    """
    import scripts.qwen_decode_worker as decoder
    import train_timelens_three_group_opsd as core
    vp = processor.video_processor
    patch, merge, temporal = (int(getattr(vp, key)) for key in
                              ('patch_size', 'merge_size', 'temporal_patch_size'))
    factor = patch*merge
    video = dict(type='video', video=sample['video_path'], fps=fps,
                 min_pixels=min_tokens*factor**2, total_pixels=total_tokens*factor**2)
    if max_frames:
        video['max_frames'] = max_frames
    messages = core.messages_for(sample, video, privileged=False, prompt_mode='official')
    images, videos, options = decoder.process_vision_info(messages, image_patch_size=patch,
        return_video_kwargs=True, return_video_metadata=True)
    if images or len(videos) != 1:
        raise ValueError('Expected exactly one native video')
    frames, meta = videos[0]
    meta = copy.deepcopy(meta)
    indices = meta['frames_indices']
    indices = [int(i) for i in (indices.tolist() if hasattr(indices, 'tolist') else indices)]
    height, width = map(int, frames.shape[-2:])
    source_fps = float(meta['fps'])
    if (len(indices) != len(frames) or not indices or height % factor or width % factor
            or not math.isfinite(source_fps) or source_fps <= 0
            or indices[0] < 0 or indices[-1] >= int(meta['total_num_frames'])
            or any(b < a for a, b in zip(indices, indices[1:]))):
        raise ValueError('Invalid native video geometry or source timestamps')
    per_group = (height//factor)*(width//factor)
    count = math.ceil(len(indices)/temporal)
    keep = min(count, total_tokens//per_group)
    if keep < 1:
        raise ValueError('One native temporal group exceeds the entire visual budget')
    groups = uniform_indices(count, keep)
    positions = [i for group in groups for i in range(group*temporal, min((group+1)*temporal, len(indices)))]
    meta['frames_indices'] = [indices[i] for i in positions]
    selected = frames[positions]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = dict(processor(text=[text], videos=[selected], video_metadata=[meta],
        return_tensors='pt', padding=True, **dict(options, do_resize=False, do_sample_frames=False)))
    core.verify_video_timestamps(processor, inputs, [meta])
    actual = int((inputs['input_ids'] == processor.tokenizer.convert_tokens_to_ids('<|video_pad|>')).sum())
    if actual != keep*per_group or not 0 < actual <= total_tokens:
        raise RuntimeError(f'Actual native visual tokens {actual}; expected {keep*per_group}, cap {total_tokens}')
    duration = float(meta['total_num_frames'])/source_fps
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid original video duration')
    audit = dict(mode='normal', pipeline='direct_native_fps_uniform_group_cap_v1',
        actual_visual_tokens=actual, target_tokens=total_tokens, min_tokens=min_tokens,
        requested_fps=fps, max_frames=max_frames, source_fps=source_fps, duration=duration,
        decoded_frames=len(indices), used_frames=len(positions), limited=keep<count,
        spatial_height=height, spatial_width=width, spatial_tokens_per_group=per_group,
        temporal_patch_size=temporal, processor_tail_padding_frames=(-len(positions)) % temporal,
        original_frame_indices=indices, source_frame_indices=meta['frames_indices'],
        user_prompt=messages[-1]['content'][-1]['text'], guidance_applied=False,
        processing='native branch-specific resize; cap whole groups uniformly; preserve original timestamps')
    return inputs, audit


def checked_clip_metadata(metadata, full_metadata, start, end, frame_count):
    """Decord time-range indices must already be ABSOLUTE. Never guess an offset.

    Some qwen-vl-utils versions put the clipped frame count in total_num_frames
    while retaining absolute frames_indices; restore full-source length explicitly.
    """
    result = copy.deepcopy(metadata)
    fps, full_fps = float(result['fps']), float(full_metadata['fps'])
    full_count = int(full_metadata['total_num_frames'])
    if not math.isfinite(fps) or fps <= 0 or abs(fps-full_fps) > 1e-6:
        raise ValueError('Clip and full-video source FPS disagree')
    indices = result['frames_indices']
    indices = [int(i) for i in (indices.tolist() if hasattr(indices, 'tolist') else indices)]
    first = max(0, math.ceil(start*fps-1e-8))
    last = min(full_count-1, math.floor(end*fps+1e-8))
    if (not indices or len(indices) != frame_count or first >= last
            or indices[0] < first or indices[-1] > last
            or any(b < a for a,b in zip(indices, indices[1:]))):
        raise ValueError('Clip decoder ignored the requested range or reset timestamps; absolute source indices required')
    old_count = int(result['total_num_frames']) if result.get('total_num_frames') is not None else None
    result['frames_indices'] = indices
    result['total_num_frames'] = full_count
    return result, dict(decoder_total_num_frames=old_count, restored_full_total_num_frames=full_count,
                        allowed_source_frame_range=[first,last], timestamp_policy='absolute source indices; no offset guessing')


def combined_video_options(global_options, clip_options):
    result = {}
    for key in set(global_options) | set(clip_options):
        if key in ('do_resize', 'do_sample_frames'):
            continue
        if key == 'fps':
            values = []
            for options in (global_options, clip_options):
                value = options.get(key)
                value = list(value) if isinstance(value, (list, tuple)) else [value]
                if len(value) != 1 or value[0] is None:
                    raise ValueError('Expected one decoder sampling FPS value per video')
                values.extend(value)
            result[key] = values
        elif key not in global_options or key not in clip_options or global_options[key] != clip_options[key]:
            raise ValueError('Incompatible native two-video processor option: '+key)
        else:
            result[key] = global_options[key]
    result.update(do_resize=False, do_sample_frames=False)
    return result


def build_gt_clip_input(processor, sample, pool, mode, a):
    """Actually decode a second GT-range video and append it to global context.

    Same spatial grid as the global pool: split its total group allowance between
    global and clip so normal vs GT-clip have EXACTLY the same visual-token count.
    """
    import scripts.qwen_decode_worker as decoder
    import train_timelens_three_group_opsd as core
    vp = processor.video_processor
    temporal, patch, merge = (int(getattr(vp,k)) for k in ('temporal_patch_size','patch_size','merge_size'))
    height, width = map(int, pool['frames'].shape[-2:])
    per = pool['per_group']
    clip_allowance = min(a.clip_total_tokens//per, pool['keep']-2)
    if temporal != pool['temporal'] or clip_allowance < 1:
        raise ValueError('Matched video budget cannot retain two global endpoint groups and one GT clip group')
    start = max(0., sample['gt_start']-a.clip_padding_seconds)
    end = min(pool['duration'], sample['gt_end']+a.clip_padding_seconds)
    video = dict(type='video', video=sample['video_path'], video_start=start, video_end=end,
        fps=a.clip_fps, max_frames=a.clip_max_frames, min_pixels=a.min_tokens*(patch*merge)**2,
        total_pixels=a.clip_total_tokens*(patch*merge)**2, resized_height=height, resized_width=width)
    # These start/end fields control decoder I/O; the chat template receives no GT answer text.
    request = core.messages_for(sample, video, privileged=False, prompt_mode='official')
    images, videos, clip_options = decoder.process_vision_info(request, image_patch_size=patch,
        return_video_kwargs=True, return_video_metadata=True)
    if images or len(videos) != 1:
        raise ValueError('Expected one independently decoded GT reference clip')
    clip_frames, clip_meta = videos[0]
    clip_meta, meta_audit = checked_clip_metadata(clip_meta, pool['metadata'], start, end, len(clip_frames))
    if tuple(map(int, clip_frames.shape[-2:])) != (height,width) or len(clip_frames) % temporal:
        raise ValueError('GT clip must retain the shared spatial grid and complete temporal groups')
    clip_groups = uniform_indices(len(clip_frames)//temporal, min(len(clip_frames)//temporal, clip_allowance))
    clip_positions = [i for g in clip_groups for i in range(g*temporal,(g+1)*temporal)]
    all_clip_indices = list(clip_meta['frames_indices'])
    clip_meta['frames_indices'] = [all_clip_indices[i] for i in clip_positions]
    clip_tensor = clip_frames[clip_positions]
    if mode == 'gt_clip_blank':
        import torch
        clip_tensor = torch.zeros_like(clip_tensor)
    global_keep = pool['keep']-len(clip_groups)
    global_groups = uniform_indices(len(pool['times']), global_keep)
    global_positions = [i for g in global_groups for i in range(g*temporal,(g+1)*temporal)]
    global_meta = copy.deepcopy(pool['metadata'])
    global_meta['frames_indices'] = [pool['indices'][i] for i in global_positions]
    global_tensor = pool['frames'][global_positions]
    messages = core.messages_for(sample, dict(type='video',video=sample['video_path']), prompt_mode='official')
    content = messages[-1]['content']
    content.insert(0, dict(type='text',text='Full-source context video:'))
    content.insert(2, dict(type='text',text=GT_CLIP_GUIDANCE+'\nReference clip:'))
    content.insert(3, dict(type='video',video=sample['video_path']))
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = dict(processor(text=[text], videos=[global_tensor,clip_tensor],
        video_metadata=[copy.deepcopy(global_meta),copy.deepcopy(clip_meta)], return_tensors='pt', padding=True,
        **combined_video_options(pool['options'], clip_options)))
    core.verify_video_timestamps(processor, inputs, [global_meta,clip_meta])
    expected_grid = [[global_keep,height//patch,width//patch], [len(clip_groups),height//patch,width//patch]]
    if inputs['video_grid_thw'].tolist() != expected_grid:
        raise RuntimeError('Native two-video geometry changed during processor encoding')
    actual = int((inputs['input_ids'] == processor.tokenizer.convert_tokens_to_ids('<|video_pad|>')).sum())
    expected = pool['keep']*per
    if actual != expected or not 0 < actual <= a.total_tokens:
        raise RuntimeError(f'Two-video visual budget mismatch: actual {actual}, matched normal {expected}, cap {a.total_tokens}')
    audit = dict(mode=mode, actual_visual_tokens=actual, target_tokens=a.total_tokens,
        global_visual_tokens=global_keep*per, clip_visual_tokens=len(clip_groups)*per,
        full_view_reallocated=True, total_budget_policy='global+clip exactly matches normal total visual tokens',
        duration=pool['duration'], guidance_applied=True, user_prompt=content[2]['text']+'\n'+content[-1]['text'],
        focus_regions=[], original_chat_messages=messages, global_source_frame_indices=global_meta['frames_indices'],
        clip=dict(requested_interval=[start,end], gt_interval=[sample['gt_start'],sample['gt_end']],
                  padding_seconds=a.clip_padding_seconds, requested_fps=a.clip_fps,
                  decoded_frames=len(clip_frames), used_frames=len(clip_positions),
                  source_fps=float(clip_meta['fps']), source_frame_indices=clip_meta['frames_indices'],
                  sampled_timestamp_range=[clip_meta['frames_indices'][0]/float(clip_meta['fps']),
                                           clip_meta['frames_indices'][-1]/float(clip_meta['fps'])],
                  content='black pixels control' if mode == 'gt_clip_blank' else 'actual decoded RGB',
                  metadata_audit=meta_audit),
        shortcut_note='Exact GT crop selection plus original timestamps reveals target location; content benefit needs blank/control comparison')
    return inputs, audit


def build_input(processor, sample, pool, mode, a):
    import train_timelens_three_group_opsd as core
    if mode not in MODES:
        raise ValueError(mode)
    if mode in ('gt_clip', 'gt_clip_blank'):
        return build_gt_clip_input(processor, sample, pool, mode, a)
    span = [sample['gt_start'], sample['gt_end']]
    selected_span = wrong_span(span, pool['duration']) if mode == 'wrong_evidence' else span
    groups = select_groups(pool['times'], pool['keep'],
                           selected_span if mode in EVIDENCE_MODES else None, a.boundary_seconds)
    positions = [i for g in groups for i in range(g*pool['temporal'], (g+1)*pool['temporal'])]
    content_positions = list(positions)
    if mode == 'shuffled_evidence' and len(groups) > 1:
        # A deterministic derangement of entire frame groups; timestamps/density unchanged.
        shifted = groups[1:]+groups[:1]
        content_positions = [i for g in shifted for i in range(g*pool['temporal'], (g+1)*pool['temporal'])]
    frames = pool['frames'][content_positions]
    meta = copy.deepcopy(pool['metadata'])
    meta['frames_indices'] = [pool['indices'][i] for i in positions]
    video = dict(type='video', video=sample['video_path'])
    # This is the only branch that inserts exact GT values in a model prompt.
    messages = core.messages_for(sample, video, privileged=mode == 'gt_text', prompt_mode='official')
    focus_regions = []
    focus_detection = None
    if mode == 'gt_evidence_marked':
        normal_groups = select_groups(pool['times'], pool['keep'])
        normal_meta = copy.deepcopy(pool['metadata'])
        normal_meta['frames_indices'] = [pool['indices'][i] for g in normal_groups
            for i in range(g*pool['temporal'], (g+1)*pool['temporal'])]
        focus_regions, focus_detection = dense_focus_regions(
            visible_group_times(processor, meta), visible_group_times(processor, normal_meta))
        messages[-1]['content'][-1]['text'] += marked_guidance(focus_regions)
    if mode == 'gt_evidence_guided':
        messages[-1]['content'][-1]['text'] += DENSE_EVIDENCE_GUIDANCE
    user_prompt = messages[-1]['content'][-1]['text']
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    options = dict(pool['options'], do_resize=False, do_sample_frames=False)
    inputs = dict(processor(text=[text], videos=[frames], video_metadata=[meta],
                            return_tensors='pt', padding=True, **options))
    core.verify_video_timestamps(processor, inputs, [meta])
    actual = int((inputs['input_ids'] == processor.tokenizer.convert_tokens_to_ids('<|video_pad|>')).sum())
    expected = pool['keep']*pool['per_group']
    if actual != expected or not 0 < actual <= a.total_tokens:
        raise RuntimeError(f'Actual visual tokens {actual}, expected {expected}, budget {a.total_tokens}')
    audit = dict(mode=mode, actual_visual_tokens=actual, target_tokens=a.total_tokens,
        decoded_frames=pool['decoded_frames'], used_frames=len(positions),
        dropped_tail_frames=pool['dropped_tail_frames'], source_fps=float(meta['fps']),
        source_frame_indices=meta['frames_indices'],
        content_frame_indices=[pool['indices'][i] for i in content_positions],
        selected_interval=selected_span if mode in EVIDENCE_MODES else None,
        guidance_applied=mode in ('gt_evidence_guided', 'gt_evidence_marked'), user_prompt=user_prompt,
        focus_regions=focus_regions, focus_detection=focus_detection,
        groups_different_from_normal=len(set(groups).symmetric_difference(
            select_groups(pool['times'], pool['keep']))),
        duration=pool['duration'], dense_pool_fps=a.fps*a.pool_factor,
        processing='cap whole temporal groups; sorted original timestamps; fixed matched group count')
    return inputs, audit


def move(inputs, model):
    device = model.get_input_embeddings().weight.device
    return {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}


def generate(model, processor, inputs, sample, duration, a, sampled=False, generation_options=None):
    import torch
    import train_timelens_three_group_opsd as core
    model.eval()
    core.reset_rope(model)
    cfg = getattr(model.config, 'text_config', model.config)
    maximum = cfg.max_position_embeddings
    if inputs['input_ids'].shape[1] + a.max_new_tokens > maximum:
        raise ValueError('Prompt plus continuation exceeds model context')
    with torch.no_grad():
        output = model.generate(**inputs, do_sample=sampled, num_beams=1, num_return_sequences=1,
            max_new_tokens=a.max_new_tokens, use_cache=True, repetition_penalty=1.,
            logits_to_keep=1, output_attentions=False, output_hidden_states=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            **(dict(temperature=1., top_p=1., top_k=0) if sampled else {}),
            **(generation_options or {}))
    y = output[:, inputs['input_ids'].shape[1]:].detach().cpu().clone()
    del output
    if y.numel() == 0:
        raise RuntimeError('Empty continuation')
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    truncated = y.shape[1] >= a.max_new_tokens and int(y[0, -1]) not in eos
    result = parse_prediction(processor.tokenizer.decode(y[0].tolist(), skip_special_tokens=True),
                             [sample['gt_start'], sample['gt_end']], duration, truncated)
    result['tokens'] = y.shape[1]
    return y, result


def tokenizer_equivalence(student, teacher):
    if student.get_vocab() != teacher.get_vocab():
        raise ValueError('Token-ID vocabularies differ; direct vocabulary KL is invalid')
    for name in ('bos_token_id', 'eos_token_id', 'pad_token_id'):
        if getattr(student, name) != getattr(teacher, name):
            raise ValueError('Tokenizer special ID mismatch: '+name)
    def backend(t):
        value = json.loads(t.backend_tokenizer.to_str())
        for key in ('padding', 'truncation'):
            value.pop(key, None)
        return value
    if backend(student) != backend(teacher):
        raise ValueError('Tokenizer segmentation/normalization differs')


def scoring_logits(model, inputs, targets):
    import train_timelens_three_group_opsd as core
    inputs = move(inputs, model)
    targets = targets.to(inputs['input_ids'].device)
    scoring, count = core.scoring_inputs(inputs, targets)
    core.reset_rope(model)
    logits = model(**scoring, logits_to_keep=count, use_cache=False,
                   output_attentions=False, output_hidden_states=False).logits
    if logits.shape[:2] != targets.shape:
        raise RuntimeError('Incorrect causal positions: P-1 must predict the first rollout token')
    return logits


def choose_manifest(records, mode, anchor_count, frontier_count, seed):
    eligible = [r for r in records if r['teacher'][mode]['valid'] and r['teacher'][mode]['iou'] >= .7]
    anchors = [dict(sample=r['sample'], role='anchor', initial_student_iou=r['student']['iou'],
                    initial_teacher_iou=r['teacher'][mode]['iou']) for r in eligible
               if r['student']['valid'] and r['student']['iou'] >= .7]
    frontier = [dict(sample=r['sample'], role='frontier', initial_student_iou=r['student']['iou'],
                     initial_teacher_iou=r['teacher'][mode]['iou']) for r in eligible if r['student']['iou'] < .3]
    rng = random.Random(seed)
    rng.shuffle(anchors)
    rng.shuffle(frontier)
    ready = len(anchors) >= anchor_count and len(frontier) >= frontier_count
    chosen = anchors[:anchor_count] + frontier[:frontier_count]
    rng.shuffle(chosen)
    return dict(selection_complete=ready, teacher_mode=mode, thresholds=dict(teacher=.7, anchor=.7, frontier=.3),
                requested=dict(anchor=anchor_count, frontier=frontier_count),
                available=dict(anchor=len(anchors), frontier=len(frontier)), samples=chosen,
                note='Initial greedy predictions only; never duplicate events to fill a quota')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_common(p)
    p.add_argument('--modes', nargs='+', choices=MODES, default=['normal', 'gt_clip'])
    p.add_argument('--selection-mode', choices=MODES, default='gt_clip')
    p.add_argument('--anchor-count', type=int, default=100)
    p.add_argument('--frontier-count', type=int, default=900)
    a = p.parse_args(argv)
    if 'normal' not in a.modes or a.selection_mode not in a.modes or len(set(a.modes)) != len(a.modes):
        p.error('Unique modes must include normal and selection-mode')
    if min(a.anchor_count, a.frontier_count) < 0 or a.anchor_count+a.frontier_count < 1:
        p.error('Invalid selection quotas')
    protocol, samples = validate_common(a)
    print(json.dumps(protocol, ensure_ascii=False, indent=2), flush=True)
    if a.plan_only:
        return 0
    import torch
    import scripts.qwen_decode_worker as decoder
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write(out/'protocol.json', protocol)
    write(out/'candidate_samples.json', samples)
    records = []
    try:
        student, teacher, processors = load_runtime(a)
        protocol.update(teacher_device_map=teacher.hf_device_map, runtime=runtime_identity())
        write(out/'protocol.json', protocol)
        for i, sample in enumerate(samples):
            started = time.monotonic()
            for device in range(2):
                torch.cuda.reset_peak_memory_stats(device)
            pool = decode_pool(processors[0], sample, a)
            inputs, audit = build_input(processors[0], sample, pool, 'normal', a)
            inputs = move(inputs, student)
            _, student_result = generate(student, processors[0], inputs, sample, pool['duration'], a)
            del inputs
            result = dict(index=i, sample=sample, student=student_result, student_video=audit, teacher={}, video={})
            for mode in a.modes:
                inputs, audit = build_input(processors[1], sample, pool, mode, a)
                inputs = move(inputs, teacher)
                _, result['teacher'][mode] = generate(teacher, processors[1], inputs, sample, pool['duration'], a)
                result['video'][mode] = audit
                del inputs
            result.update(memory=memory(), seconds=time.monotonic()-started)
            records.append(result)
            append(out/'paired_results.jsonl', result)
            print(json.dumps(dict(sample=i+1, total=len(samples), student_iou=student_result['iou'],
                teacher_iou={k:v['iou'] for k,v in result['teacher'].items()}, seconds=result['seconds'],
                teacher_guidance={k:v['guidance_applied'] for k,v in result['video'].items()},
                visual_tokens={k:v['actual_visual_tokens'] for k,v in result['video'].items()},
                teacher_focus_regions={k:v['focus_regions'] for k,v in result['video'].items()
                                       if k == 'gt_evidence_marked'},
                teacher_clips={k:v['clip'] for k,v in result['video'].items() if k in ('gt_clip','gt_clip_blank')},
                memory=result['memory']), ensure_ascii=False), flush=True)
            del pool
        summary = dict(probe_complete=True, n=len(records), student=metrics([r['student'] for r in records]),
            teacher={m:metrics([r['teacher'][m] for r in records]) for m in a.modes},
            paired_vs_normal={m:paired_delta(records, m) for m in a.modes if m != 'normal'},
            note='Training-set diagnostic; privilege gains do not establish held-out student gains')
        write(out/'summary.json', summary)
        manifest = choose_manifest(records, a.selection_mode, a.anchor_count, a.frontier_count, a.seed)
        manifest['probe_protocol_sha256'] = sha(out/'protocol.json')
        write(out/'selection.json', manifest)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as error:
        write(out/'failure.json', dict(probe_complete=False, completed=len(records), error=repr(error)))
        raise
    finally:
        if decoder._decoder is not None:
            decoder._decoder.close()
            decoder._decoder = None


