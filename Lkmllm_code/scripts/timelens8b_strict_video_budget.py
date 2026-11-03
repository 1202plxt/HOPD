"""Limit native Qwen3 video groups after decode, preserving source timestamps."""
import contextlib
import copy
import math


def select_indices(count, keep):
    if not (1 <= keep <= count):
        raise ValueError('invalid frame counts')
    if keep == count:
        return list(range(count))
    if keep == 1:
        return [0]
    return [i * (count - 1) // (keep - 1) for i in range(keep)]


@contextlib.contextmanager
def strict_video_budget(processor, budget):
    """Wrap the isolated decoder for one input; do not mutate shared implementations."""
    from scripts import qwen_decode_worker as decoder
    original = decoder.process_vision_info
    report = {}

    def limited(messages, **kwargs):
        images, videos, options = original(messages, **kwargs)
        if images or len(videos) != 1 or not isinstance(videos[0], tuple):
            raise ValueError('strict TimeLens-8B budget requires one native video')
        frames, meta = videos[0]
        meta = copy.deepcopy(meta)
        vp = processor.video_processor
        patch, merge, temporal = int(vp.patch_size), int(vp.merge_size), int(vp.temporal_patch_size)
        height, width = map(int, frames.shape[-2:])
        if patch < 1 or merge < 1 or temporal < 1 or height % (patch * merge) or width % (patch * merge):
            raise ValueError('unsupported native visual geometry')
        per_group = (height // (patch * merge)) * (width // (patch * merge))
        max_groups = budget // per_group
        if max_groups < 1:
            raise ValueError('one temporal group exceeds the complete visual budget')
        before = len(frames)
        indices = meta['frames_indices']
        if hasattr(indices, 'tolist'): indices = indices.tolist()
        indices = list(indices)
        if len(indices) != before:
            raise ValueError('source frame indices do not match decoded frame count')
        keep = min(before, max_groups * temporal)
        selected = select_indices(before, keep)
        if keep < before:
            frames = frames[selected]
            meta['frames_indices'] = [indices[i] for i in selected]
        estimated = math.ceil(keep / temporal) * per_group
        if estimated > budget:
            raise RuntimeError('strict frame cap calculation exceeded the budget')
        report.update(decoded_frames=before, used_frames=keep, spatial_tokens_per_group=per_group,
                      estimated_visual_tokens=estimated, total_tokens=budget,
                      temporal_patch_size=temporal, limited=keep < before,
                      original_frame_indices=indices, used_frame_indices=[indices[i] for i in selected],
                      source_fps=meta.get('fps'))
        return images, [(frames, meta)], options

    decoder.process_vision_info = limited
    try:
        yield report
    finally:
        decoder.process_vision_info = original
