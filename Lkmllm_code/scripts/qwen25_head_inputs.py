"""Native Qwen2.5-VL base inputs for attention probing and GT replacement.

The official processor retains its native video tokens/mRoPE. No TimeLens
processor or Qwen3 frame-timestamp expansion is substituted.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PIPELINE = "qwen25_native_file_v1"
TASK = ("Please find the visual event described by the sentence '{query}', determining its "
        "starting and ending times. The format should be: "
        "'The event happens in <start time> - <end time> seconds'.")


def validate_processor(processor):
    if processor.__class__.__name__ != "Qwen2_5_VLProcessor":
        raise ValueError("Use the original Qwen2.5-VL processor, not TimeLensProcessor")
    vp = getattr(processor, "video_processor", None)
    if vp is None or (int(vp.patch_size), int(vp.merge_size), int(vp.temporal_patch_size)) != (14, 2, 2):
        raise ValueError("Requires the native Qwen2.5 video processor (14/2/2)")


def video_message(sample, fps, min_tokens, total_tokens, max_frames):
    if fps <= 0 or min_tokens <= 0 or total_tokens <= 0:
        raise ValueError("fps and video token budgets must be positive")
    if max_frames and (max_frames < 2 or max_frames % 2):
        raise ValueError("max_frames must be zero or a positive even number")
    content = dict(type="video", video=str(sample["video_path"]), fps=float(fps),
                   min_pixels=int(min_tokens) * 28 * 28,
                   total_pixels=int(total_tokens) * 28 * 28)
    if max_frames:
        content["max_frames"] = int(max_frames)
    return [dict(role="system", content=[dict(type="text", text="You are a video time analysis assistant.")]),
            dict(role="user", content=[content, dict(type="text", text=TASK.format(query=sample["query"]))])]


def decode_video(messages):
    from scripts.qwen_decode_worker import process_vision_info
    images, videos, kwargs = process_vision_info(
        messages, image_patch_size=14, return_video_metadata=True, return_video_kwargs=True)
    if images or not videos or len(videos) != 1 or not isinstance(videos[0], tuple):
        raise ValueError("Expected one native (video tensor, metadata) pair")
    tensor, metadata = videos[0]
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError("Expected decoded video [frames,3,height,width]")
    return dict(tensor=tensor, metadata=metadata, kwargs=kwargs)


def temporal_layout(processor, inputs, decoded):
    """Validate native video-token layout and use actual sampled-frame times."""
    vp = getattr(processor, "video_processor", None)
    if vp is None:
        raise ValueError("Requires native Qwen2.5 video_processor (Transformers 5.x)")
    patch, merge, temporal = int(vp.patch_size), int(vp.merge_size), int(vp.temporal_patch_size)
    if (patch, merge, temporal) != (14, 2, 2):
        raise ValueError(f"Unexpected native Qwen2.5 video layout: {patch,merge,temporal}")
    tensor, meta = decoded["tensor"], decoded["metadata"]
    frames, _, height, width = map(int, tensor.shape)
    grid = [int(v) for v in inputs["video_grid_thw"][0].tolist()]
    expected = [(frames + temporal - 1) // temporal, height // patch, width // patch]
    if grid != expected or height % (patch * merge) or width % (patch * merge):
        raise RuntimeError(f"Processor changed the decoded grid: {grid} != {expected}")
    per_group = grid[1] * grid[2] // (merge * merge)
    video_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    positions = np.flatnonzero(np.asarray(inputs["input_ids"][0].tolist()) == video_id)
    if len(positions) != grid[0] * per_group or np.any(np.diff(positions) != 1):
        raise RuntimeError("Native video token count/order differs from video_grid_thw")
    indices = np.asarray(meta["frames_indices"], dtype=float).reshape(-1)
    native_fps = float(meta["fps"])
    if (len(indices) != frames or native_fps <= 0 or not np.isfinite(native_fps)
            or not np.isfinite(indices).all() or np.any(np.diff(indices) < 0)):
        raise RuntimeError("Invalid original sampled-frame metadata")
    padded = np.pad(indices, (0, (-len(indices)) % temporal), mode="edge")
    times = padded.reshape(-1, temporal).mean(axis=1) / native_fps
    return grid, per_group, times, positions


def process_messages(processor, messages, videos, *, answer=None):
    msgs = copy.deepcopy(messages)
    if answer is not None:
        msgs.append(dict(role="assistant", content=[dict(type="text", text=answer)]))
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=answer is None)
    meta = copy.deepcopy(videos["metadata"])
    options = dict(videos["kwargs"])
    options.update(do_resize=False, do_sample_frames=False)
    result = processor(text=[text], videos=[videos["tensor"].clone()],
                       video_metadata=[meta], padding=True, return_tensors="pt", **options)
    # Match the official Qwen2.5 fps argument, including max_frames-limited fps.
    # Some 5.x processors infer sampled_fps from source metadata differently.
    sampled_fps = options.get("fps")
    if isinstance(sampled_fps, (list, tuple)):
        sampled_fps = sampled_fps[0]
    if sampled_fps is None:
        sampled_fps = (float(meta["fps"]) * len(meta["frames_indices"])
                       / float(meta["total_num_frames"]))
    sampled_fps = float(sampled_fps)
    if not np.isfinite(sampled_fps) or sampled_fps <= 0:
        raise ValueError("Decoder did not provide a valid effective sampling fps")
    result["second_per_grid_ts"] = [int(processor.video_processor.temporal_patch_size) / sampled_fps]
    # Diagnostic objects must never be forwarded as model kwargs.
    result.pop("video_metadata", None)
    temporal_layout(processor, result, videos)
    return result


def audit_inputs(processor, inputs, decoded, start=None, end=None):
    grid, per_group, times, positions = temporal_layout(processor, inputs, decoded)
    meta = decoded["metadata"]
    audit = dict(pipeline=PIPELINE, native_fps=float(meta["fps"]),
                 total_num_frames=int(meta["total_num_frames"]),
                 video_shape=list(decoded["tensor"].shape), video_grid_thw=grid,
                 n_video_tokens=len(positions), sampled_temporal_patch_times=times.tolist(),
                 second_per_grid_ts=list(inputs["second_per_grid_ts"]),
                 frame_indices=np.asarray(meta["frames_indices"]).tolist(),
                 do_resize=False, do_sample_frames=False, patch_size=14,
                 gt_mapping="mean original timestamp of each temporal patch; inclusive GT bounds")
    if start is not None:
        inside = np.flatnonzero((times >= start) & (times <= end))
        if not len(inside):
            raise ValueError("No sampled temporal patch falls in GT; skip rather than invent a GT key")
        audit["gt_video_token_range"] = [int(inside[0]) * per_group, (int(inside[-1]) + 1) * per_group]
    return audit


def build_inputs(processor, video_path, query, duration, device, sample_fps,
                 gt_start=0., gt_end=0., min_tokens=4, total_tokens=4096,
                 video_metadata=None, timelens_model=False, max_frames=1024,
                 taro_model=False, video_audit=None, prompt_only=False,
                 answer_text_override=None, vision_cache=None):
    if timelens_model or taro_model:
        raise ValueError("Qwen2.5 base must use its own processor and prompt")
    messages = video_message(dict(video_path=video_path, query=query), sample_fps,
                             min_tokens, total_tokens, max_frames)
    decoded = (vision_cache.get("qwen25_video") if vision_cache is not None else None)
    if decoded is None:
        decoded = decode_video(messages)
        if vision_cache is not None:
            vision_cache["qwen25_video"] = decoded
    answer = (answer_text_override if answer_text_override is not None else
              f"The event happens in {gt_start:.1f} - {gt_end:.1f} seconds.")
    inputs = process_messages(processor, messages, decoded, answer=None if prompt_only else answer)
    audit = audit_inputs(processor, inputs, decoded, gt_start, gt_end)
    if video_audit is not None:
        video_audit.update(audit)
    if video_metadata is not None:
        video_metadata.update(copy.deepcopy(decoded["metadata"]))
        video_metadata["duration"] = float(decoded["metadata"]["total_num_frames"]) / float(decoded["metadata"]["fps"])
    import torch
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in inputs.items()}
    return inputs, None, answer


def load_model(path, *, device_map="auto", gpu_mem_gib=16., gpu_reserve_gib=4., require_all_gpus=False):
    import torch
    from scripts.new_d_head_attribution import load_model_and_processor, device_map_for_metadata, validate_gpu_only_map
    model, processor, _ = load_model_and_processor(Path(path), device_map=device_map,
                          gpu_mem_gib=gpu_mem_gib, gpu_reserve_gib=gpu_reserve_gib)
    if model.config.model_type != "qwen2_5_vl" or processor.__class__.__name__ != "Qwen2_5_VLProcessor":
        raise ValueError("Expected the original Qwen2.5-VL model and native processor")
    validate_processor(processor)
    used = validate_gpu_only_map(device_map_for_metadata(model), torch.cuda.device_count())
    if require_all_gpus and len(used) != torch.cuda.device_count():
        raise RuntimeError("Model was not distributed over all visible GPUs")
    model.generation_config.num_beams = 1
    model.generation_config.num_return_sequences = 1
    return model, processor


def generate(model, processor, inputs, max_new_tokens):
    from scripts.new_d_head_attribution import generate_probe_answer
    text, parsed = generate_probe_answer(model, processor, inputs, max_new_tokens)
    return text, list(parsed[:2]) if parsed is not None else None
