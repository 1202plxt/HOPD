"""Frozen 32B runtime and matched global-video/GT-reference-clip inputs."""
import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
import re
from pathlib import Path


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
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError('Requires CUDA or HIP PyTorch and at least two visible GPUs')
    torch.manual_seed(a.seed)
    processors = [AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=True,
                       padding_side='left', do_resize=False) for path in
                  (a.student_path, a.teacher_path)]
    for role, p in zip(('student', 'teacher'), processors):
        if p.tokenizer.pad_token_id is None:
            p.tokenizer.pad_token = p.tokenizer.eos_token
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
    import qwen_decode_worker as decoder
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
    import qwen_decode_worker as decoder
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
    import qwen_decode_worker as decoder
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
    if getattr(a,'token_local_gate',False):
        content[-1]['text'] += (f"\nThe reference temporal interval is {sample['gt_start']} to {sample['gt_end']} seconds. Analyze the video evidence and give your own prediction.")
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
        duration=pool['duration'], gt_answer_text=bool(getattr(a,'token_local_gate',False)), guidance_applied=True, user_prompt=content[2]['text']+'\n'+content[-1]['text'],
        focus_regions=[], original_chat_messages=messages, global_source_frame_indices=global_meta['frames_indices'],
        clip=dict(requested_interval=[start,end], gt_interval=[sample['gt_start'],sample['gt_end']],
                  padding_seconds=a.clip_padding_seconds, requested_fps=a.clip_fps,
                  decoded_frames=len(clip_frames), used_frames=len(clip_positions),
                  source_fps=float(clip_meta['fps']), source_frame_indices=clip_meta['frames_indices'],
                  sampled_timestamp_range=[clip_meta['frames_indices'][0]/float(clip_meta['fps']),
                                           clip_meta['frames_indices'][-1]/float(clip_meta['fps'])],
                  content='actual decoded RGB',
                  metadata_audit=meta_audit),
        shortcut_note='Exact GT crop selection plus original timestamps reveals target location; content benefit needs blank/control comparison')
    return inputs, audit


def move(inputs, model):
    device = model.get_input_embeddings().weight.device
    return {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}


def generate(model, processor, inputs, sample, duration, a, sampled=False, generation_options=None):
    import torch
    import train_timelens_three_group_opsd as core
    inputs = move(inputs, model)
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
            **(dict(temperature=1., top_p=1., top_k=0) if sampled else
               dict(temperature=None, top_p=None, top_k=None)),
            **(generation_options or {}))
    y = output[:, inputs['input_ids'].shape[1]:].detach().cpu().clone()
    del output
    if y.numel() == 0:
        raise RuntimeError('Empty continuation')
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    truncated = y.shape[1] >= a.max_new_tokens and int(y[0, -1]) not in eos
    result = parse_prediction(processor.tokenizer.decode(y[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False),
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


def training_inputs(processors, sample, a):
    import train_timelens_three_group_opsd as core
    pool = decode_pool(processors[0], sample, a)
    groups = uniform_indices(len(pool['times']), pool['keep'])
    positions = [i for g in groups for i in range(g*pool['temporal'], (g+1)*pool['temporal'])]
    meta = copy.deepcopy(pool['metadata'])
    meta['frames_indices'] = [pool['indices'][i] for i in positions]
    messages = core.messages_for(sample, dict(type='video',video=sample['video_path']))
    text = processors[0].apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    student = dict(processors[0](text=[text], videos=[pool['frames'][positions]], video_metadata=[meta],
        return_tensors='pt', padding=True, **dict(pool['options'],do_resize=False,do_sample_frames=False)))
    core.verify_video_timestamps(processors[0], student, [meta])
    actual = int((student['input_ids'] == processors[0].tokenizer.convert_tokens_to_ids('<|video_pad|>')).sum())
    if not 0 < actual == pool['keep']*pool['per_group'] <= a.total_tokens:
        raise RuntimeError('Normal student visual budget mismatch')
    teacher, audit = build_gt_clip_input(processors[1], sample, pool, 'gt_clip', a)
    if audit['actual_visual_tokens'] != actual:
        raise RuntimeError('Student/teacher matched visual budget mismatch')
    return student, teacher, dict(actual_visual_tokens=actual, source_frame_indices=meta['frames_indices'],
        user_prompt=messages[-1]['content'][-1]['text']), audit, pool['duration']


def metrics(rows):
    values = [r['iou'] for r in rows]
    n = len(values)
    return dict(n=n, valid=sum(r['valid'] for r in rows),
                mIoU=math.fsum(values)/n if n else None,
                **{f'R@{t}': sum(v >= t for v in values)/n if n else None for t in (.3, .5, .7)})


def normal_input(processor, sample, total_tokens, min_tokens, fps, max_frames=0):
    """Direct native video input with its OWN budget; no GT-based selection or prompt.

    Native decoder chooses time points from fps/cap. Spatial resizing uses this
    branch's budget. If still over budget, thin complete temporal groups uniformly.
    """
    import qwen_decode_worker as decoder
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

