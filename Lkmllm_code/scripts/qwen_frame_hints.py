"""Teacher-only visible hints on actual budgeted frames; no new video tokens.

An experimental cue, not a claim that a model understands these marks. Use the
four-arm probe before training: plain / IDs-only / correct / shifted evidence.
"""
import copy
import math
import numpy as np


class NoEvidenceFrame(ValueError):
    pass


HINT_TEXT = ('Training-only visual guide: frame labels F0, F1, ... identify sampled '
             'temporal groups. A green border marks candidate visual evidence for '
             'the queried event. Inspect the video content and its original timeline '
             'to determine the event boundaries; do not output the frame label.')


def group_index(metadata, frame_count, temporal_patch_size, start, end):
    indices = np.asarray(metadata['frames_indices'], dtype=float).reshape(-1)
    fps = float(metadata['fps'])
    total = int(metadata['total_num_frames'])
    patch = int(temporal_patch_size)
    if (len(indices) != frame_count or not frame_count or patch < 1
            or not math.isfinite(fps) or fps <= 0 or total <= 0
            or not np.isfinite(indices).all() or np.any(np.diff(indices) < 0)
            or np.any(indices < 0) or np.any(indices >= total)
            or not math.isfinite(start + end) or not 0 <= start < end):
        raise ValueError('Invalid native frame index, fps, temporal patch or GT span')
    rows = []
    for offset in range(0, frame_count, patch):
        positions = list(range(offset, min(offset + patch, frame_count)))
        original = indices[positions].tolist()
        padded = original + [original[-1]] * (patch - len(original))
        time = sum(padded) / patch / fps
        rows.append(dict(group_id=len(rows), positions=positions,
                         source_indices=original, timestamp=time,
                         evidence=bool(start <= time <= end)))
    if not any(r['evidence'] for r in rows):
        raise NoEvidenceFrame('No actual sampled temporal-group center in GT; do not invent or remap a frame')
    return rows


def shifted_ids(rows):
    marked = {r['group_id'] for r in rows if r['evidence']}
    candidates = []
    for shift in range(1, len(rows)):
        moved = {(i + shift) % len(rows) for i in marked}
        candidates.append((len(moved & marked), shift, moved))
    if not candidates:
        return None
    overlap, shift, moved = min(candidates, key=lambda x: (x[0], x[1]))
    return moved if overlap == 0 else None


def render_teacher_frames(frames, rows, arm='correct'):
    from PIL import Image, ImageDraw, ImageFont
    if arm not in ('correct', 'shifted', 'ids_only'):
        raise ValueError('Unknown frame-hint control arm')
    source = np.asarray(frames)
    if (source.ndim != 4 or source.shape[1] != 3 or min(source.shape[2:]) < 16
            or not np.isfinite(source).all() or source.min() < 0 or source.max() > 255):
        raise ValueError('Expected finite unnormalized TCHW RGB pixels in [0,255]')
    output = source.copy()
    height, width = output.shape[-2:]
    border = max(1, min(4, min(height, width) // 20))
    marked = {r['group_id'] for r in rows if r['evidence']}
    if arm == 'ids_only':
        marked = set()
    elif arm == 'shifted':
        marked = shifted_ids(rows)
        if marked is None:
            raise NoEvidenceFrame('No disjoint shifted-evidence control exists at this temporal resolution')
    font = ImageFont.load_default()
    for row in rows:
        label = f"F{row['group_id']}"
        glyph = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(glyph)
        box = draw.textbbox((border + 1, border + 1), label, font=font)
        if box[2] + 1 >= width or box[3] + 1 >= height:
            try:
                small_font = ImageFont.load_default(size=6)
            except TypeError as error:
                raise NoEvidenceFrame('Final budgeted frame cannot fit a complete frame label with this Pillow font') from error
            box = draw.textbbox((border + 1, border + 1), label, font=small_font)
            if box[2] + 1 >= width or box[3] + 1 >= height:
                raise NoEvidenceFrame('Final budgeted frame is too small for a complete frame label')
        else:
            small_font = font
        draw.text((border + 1, border + 1), label, font=small_font, fill=255)
        mask = np.asarray(glyph) > 0
        for pos in row['positions']:
            image = output[pos]
            image[:, border: min(height, box[3] + 1), border: min(width, box[2] + 1)] = 0
            image[:, mask] = 255
            if row['group_id'] in marked:
                color = np.asarray([0, 255, 0], dtype=image.dtype)[:, None, None]
                image[:, :border, :] = color
                image[:, -border:, :] = color
                image[:, :, :border] = color
                image[:, :, -border:] = color
    return output, sorted(marked)


def prepare_pair(processor, sample, args, arm='correct', cache=None):
    import torch
    import train_timelens_three_group_opsd as core
    from qwen_head_opsd import json_value
    if args.model_family == 'qwen25vl':
        import qwen25_head_inputs as native
        native.validate_processor(processor)
        video = native.video_message(sample, args.fps, args.min_tokens,
                                     args.total_tokens, args.max_frames)[-1]['content'][0]
    else:
        from scripts.qwen_decode_worker import process_vision_info
        video = dict(type='video', video=sample['video_path'], fps=args.fps,
                     min_pixels=args.min_tokens * 32**2, total_pixels=args.total_tokens * 32**2)
        if args.max_frames:
            video['max_frames'] = args.max_frames
    messages = core.messages_for(sample, video, privileged=False, prompt_mode='official')
    if cache is not None and 'decoded' in cache:
        decoded = cache['decoded']
    elif args.model_family == 'qwen25vl':
        decoded = native.decode_video(messages)
    else:
        images, videos, kwargs = process_vision_info(messages, image_patch_size=16,
            return_video_metadata=True, return_video_kwargs=True)
        if images or not videos or len(videos) != 1:
            raise ValueError('Expected exactly one native decoded video')
        tensor, metadata = videos[0]
        decoded = dict(tensor=tensor, metadata=metadata, kwargs=kwargs)
    if cache is not None:
        cache['decoded'] = decoded
    tensor = decoded['tensor']
    patch = int(processor.video_processor.temporal_patch_size)
    rows = group_index(decoded['metadata'], len(tensor), patch,
                       float(sample['gt_start']), float(sample['gt_end']))
    pixels, marked = render_teacher_frames(tensor.detach().cpu().numpy(), rows, arm)
    teacher_decoded = dict(decoded, tensor=torch.from_numpy(pixels).to(tensor.dtype))
    teacher_messages = copy.deepcopy(messages)
    teacher_messages[-1]['content'][-1]['text'] += '\n' + HINT_TEXT

    def process(msgs, data):
        if args.model_family == 'qwen25vl':
            return native.process_messages(processor, msgs, data)
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        options = dict(data['kwargs'], do_resize=False, do_sample_frames=False)
        result = dict(processor(text=[text], videos=[data['tensor'].clone()],
            video_metadata=[copy.deepcopy(data['metadata'])], return_tensors='pt', padding=True, **options))
        result.pop('video_metadata', None)
        core.verify_video_timestamps(processor, result, [data['metadata']])
        return result

    student, teacher = process(messages, decoded), process(teacher_messages, teacher_decoded)
    grid = student['video_grid_thw']
    if not torch.equal(grid, teacher['video_grid_thw']):
        raise RuntimeError('Frame hint changed the native temporal/spatial grid')
    if torch.equal(student['pixel_values_videos'], teacher['pixel_values_videos']):
        raise RuntimeError('The visible teacher marks did not survive pixel processing')
    expected = [(len(tensor) + patch - 1)//patch,
                int(tensor.shape[2])//int(processor.video_processor.patch_size),
                int(tensor.shape[3])//int(processor.video_processor.patch_size)]
    if grid.tolist() != [expected] or len(rows) != expected[0]:
        raise RuntimeError('Processor resampled or resized the final budgeted frames')
    merge = int(processor.video_processor.merge_size)
    count = math.prod(expected)//merge**2
    video_id = processor.tokenizer.convert_tokens_to_ids('<|video_pad|>')
    for inputs in (student, teacher):
        if inputs['input_ids'][0].tolist().count(video_id) != count or count > args.total_tokens:
            raise RuntimeError('Video placeholder/grid/budget mismatch')
    if args.model_family == 'qwen25vl' and student['second_per_grid_ts'] != teacher['second_per_grid_ts']:
        raise RuntimeError('Teacher/student native time spacing differs')
    args._last_video_audit = dict(pipeline='native_budgeted_teacher_pixel_hint_v1', arm=arm,
        frame_index=json_value(rows), marked_group_ids=marked, video_shape=list(tensor.shape),
        video_grid_thw=grid.tolist(), n_video_tokens=count,
        native_metadata=json_value(decoded['metadata']), cue_instruction=HINT_TEXT,
        cue='teacher-only F-index label and green border, after final sampling/resize',
        student_pixels='unaltered', do_resize=False, do_sample_frames=False,
        exact_gt_text=False, note='cue readability/utility must be tested; grid integrity does not prove understanding')
    return student, teacher


def probe(args, protocol):
    """Evaluate four contexts on identical budgeted frames; no LoRA updates."""
    from pathlib import Path
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import train_timelens_three_group_opsd as core
    from qwen_head_opsd import atomic_json, append_json, greedy_answer, json_value
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the actual cue-utility probe')
    n = torch.cuda.device_count()
    if n not in (1, 2) or (args.device_map == 'single' and n != 1):
        raise ValueError('Expose one GPU for single or two GPUs for balanced model sharding')
    device_map = {'': 0} if n == 1 else args.device_map
    memory = None if n == 1 else {0: f'{args.gpu_mem_gib:g}GiB', 1: f'{args.gpu_mem_gib:g}GiB', 'cpu': 0}
    model = AutoModelForImageTextToText.from_pretrained(args.model_path,
        dtype=torch.bfloat16, device_map=device_map, max_memory=memory,
        local_files_only=True, attn_implementation=args.attn_implementation).eval()
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, padding_side='left')
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        if parameter.device.type != 'cuda':
            raise RuntimeError('Probe weights must remain on CUDA')
    device = model.get_input_embeddings().weight.device
    samples, _ = core.load_samples(args.anno_json, args.video_dir, args.probe_samples, args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'protocol.json', dict(protocol, probe=True, training=False, arguments=json_value(vars(args))))
    completed, paired, skipped = [], [], 0
    try:
        for i, sample in enumerate(samples):
            row = dict(index=i, **sample, arms={})
            cache = {}
            try:
                for arm in ('ids_only', 'correct', 'shifted'):
                    student, teacher = prepare_pair(processor, sample, args, arm=arm, cache=cache)
                    if arm == 'ids_only':
                        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in student.items()}
                        text = greedy_answer(model, processor, inputs, args)
                        row['arms']['plain'] = dict(output=text, **core.rollout_metrics(text, sample))
                        del inputs
                    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in teacher.items()}
                    text = greedy_answer(model, processor, inputs, args)
                    row['arms'][arm] = dict(output=text, audit=args._last_video_audit,
                                            **core.rollout_metrics(text, sample))
                    del inputs, student, teacher
                row['status'] = 'paired_complete'
                paired.append(row)
            except NoEvidenceFrame as error:
                row.update(status='no_valid_paired_control', error=str(error))
                skipped += 1
            append_json(out / 'probe.jsonl', row)
            completed.append(row)
            print(f'[frame-hint probe] {i+1}/{len(samples)} paired={len(paired)} skipped={skipped}', flush=True)
    finally:
        means = {arm: (100 * sum(r['arms'][arm]['rollout_iou'] for r in paired) / len(paired)
                        if paired else None) for arm in ('plain', 'ids_only', 'correct', 'shifted')}
        atomic_json(out / 'probe_summary.json', dict(
            complete=len(completed) == len(samples), attempted=len(completed), requested=len(samples),
            paired_complete=len(paired), skipped_no_valid_control=skipped,
            paired_miou_percent=means,
            paired_parse_failures={arm: sum(r['arms'][arm]['parse_failure'] for r in paired) for arm in means},
            note='Same training events and decoded frames in four arms. Positive paired gaps suggest cue utility; they do not prove head mediation, transfer, or statistical significance.',
            trained=False, merged=False))
    return 0 if paired else 2
