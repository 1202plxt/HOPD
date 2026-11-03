"""Qwen3 GT-mask OPD: common 8192-token video + separate GT clip for teacher."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import probe_qwen32b_privileged_k100 as common
import train_qwen32b_qwen3_gtclip_opd_k100 as base


def parser():
    p = base.parser()
    p.description = __doc__
    p.set_defaults(total_tokens=8192, min_tokens=4, student_total_tokens=8192,
                   teacher_total_tokens=8192, student_min_tokens=4, teacher_min_tokens=4,
                   fps=2., max_frames=768, pool_factor=2, max_pool_frames=1536,
                   clip_total_tokens=1024, clip_fps=4., clip_max_frames=128,
                   clip_padding_seconds=0., lr=2e-5, grad_accum=4, save_every_steps=25)
    return p


def prepare(a):
    if (a.total_tokens, a.student_total_tokens, a.teacher_total_tokens) != (8192, 8192, 8192):
        raise ValueError('Full-video budget must be 8192 on both branches')
    if (a.min_tokens, a.student_min_tokens, a.teacher_min_tokens, a.fps,
            a.max_frames, a.pool_factor, a.max_pool_frames) != (4, 4, 4, 2., 768, 2, 1536):
        raise ValueError('Required full-video settings: min4 fps2 max768, pool factor2 and max1536')
    if a.teacher_mode != 'gt_clip' or a.gate != 'teacher-better' or a.epochs != 1 or a.selection_json or a.privilege_mix != 1:
        raise ValueError('Requires gt_clip, teacher-better, one epoch and no selection or privilege mixing')
    if (a.student_target_iou, a.teacher_min_iou, a.iou_margin) != (1., 0., 0.):
        raise ValueError('Teacher must strictly exceed student with no additional thresholds')
    # Reuse the original entry's identity and optimizer validation without changing its
    # long-established 14336 defaults or modifying the actual 8192 runtime arguments.
    validated = copy.copy(a)
    validated.total_tokens = validated.student_total_tokens = validated.teacher_total_tokens = 14336
    validated.min_tokens = validated.student_min_tokens = validated.teacher_min_tokens = 64
    validated.max_frames = 0
    protocol, samples, roles = base.prepare(validated)
    protocol['arguments'] = vars(a).copy()
    protocol.update(method='qwen3_gtmask10_32b_extra_gtclip_8192_v1',
        video_protocol='Shared dense-pool full video, actual <=8192; teacher additionally receives independent GT clip',
        clip_policy=dict(maximum_visual_tokens=a.clip_total_tokens, included_in_teacher_cap=False,
                         fps=a.clip_fps, max_frames=a.clip_max_frames, padding_seconds=a.clip_padding_seconds),
        head_selection_transfer='Qwen3 own GT-mask Top10 selected under 4096/min4; reused at 8192, not re-ranked',
        source_sha256={**protocol['source_sha256'], Path(__file__).name: common.sha(__file__)})
    return protocol, samples, roles


def training_inputs(processors, sample, a):
    import scripts.qwen_decode_worker as decoder
    import train_timelens_three_group_opsd as core
    pool = common.decode_pool(processors[0], sample, a)
    student, sa = common.build_input(processors[0], sample, pool, 'normal', a)
    vp = processors[1].video_processor
    patch, merge, temporal = (int(getattr(vp, k)) for k in ('patch_size', 'merge_size', 'temporal_patch_size'))
    if temporal != pool['temporal']:
        raise ValueError('Teacher/student temporal video grids do not match')
    height, width = map(int, pool['frames'].shape[-2:])
    per = pool['per_group']
    if (height // (patch * merge)) * (width // (patch * merge)) != per:
        raise ValueError('Teacher/student spatial video grids do not match')
    allowance = a.clip_total_tokens // per
    if allowance < 1:
        raise ValueError(f'GT clip budget {a.clip_total_tokens} is smaller than one spatial group {per}')
    start = max(0., sample['gt_start'] - a.clip_padding_seconds)
    end = min(pool['duration'], sample['gt_end'] + a.clip_padding_seconds)
    video = dict(type='video', video=sample['video_path'], video_start=start, video_end=end,
        fps=a.clip_fps, max_frames=a.clip_max_frames, min_pixels=a.min_tokens*(patch*merge)**2,
        total_pixels=a.clip_total_tokens*(patch*merge)**2, resized_height=height, resized_width=width)
    request = core.messages_for(sample, video, privileged=False, prompt_mode='official')
    images, videos, options = decoder.process_vision_info(request, image_patch_size=patch,
        return_video_kwargs=True, return_video_metadata=True)
    if images or len(videos) != 1:
        raise ValueError('Expected one independently decoded GT clip')
    frames, clip_meta = videos[0]
    clip_meta, meta_audit = common.checked_clip_metadata(clip_meta, pool['metadata'], start, end, len(frames))
    if tuple(map(int, frames.shape[-2:])) != (height, width) or len(frames) % temporal:
        raise ValueError('Invalid GT clip dimensions or temporal groups')
    clip_groups = common.uniform_indices(len(frames)//temporal, min(len(frames)//temporal, allowance))
    if not clip_groups:
        raise ValueError('Empty GT clip')
    cp = [i for g in clip_groups for i in range(g*temporal, (g+1)*temporal)]
    clip_meta['frames_indices'] = [clip_meta['frames_indices'][i] for i in cp]
    groups = common.select_groups(pool['times'], pool['keep'])
    gp = [i for g in groups for i in range(g*temporal, (g+1)*temporal)]
    global_meta = copy.deepcopy(pool['metadata'])
    global_meta['frames_indices'] = [pool['indices'][i] for i in gp]
    if global_meta['frames_indices'] != sa['source_frame_indices'] or len(gp) > a.max_frames:
        raise RuntimeError('Teacher full video differs from student or exceeds frame cap')
    messages = core.messages_for(sample, dict(type='video', video=sample['video_path']), prompt_mode='official')
    content = messages[-1]['content']
    content.insert(0, dict(type='text', text='Full-source context video:'))
    content.insert(2, dict(type='text', text=common.GT_CLIP_GUIDANCE+'\nReference clip:'))
    content.insert(3, dict(type='video', video=sample['video_path']))
    text = processors[1].apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    teacher = dict(processors[1](text=[text], videos=[pool['frames'][gp], frames[cp]],
        video_metadata=[global_meta, clip_meta], return_tensors='pt', padding=True,
        **common.combined_video_options(pool['options'], options)))
    core.verify_video_timestamps(processors[1], teacher, [global_meta, clip_meta])
    expected_grid = [[len(groups), height//patch, width//patch], [len(clip_groups), height//patch, width//patch]]
    if teacher['video_grid_thw'].tolist() != expected_grid:
        raise RuntimeError('Unexpected teacher video grids')
    full, clip = len(groups)*per, len(clip_groups)*per
    actual = int((teacher['input_ids'] == processors[1].tokenizer.convert_tokens_to_ids('<|video_pad|>')).sum())
    if full != sa['actual_visual_tokens'] or full > 8192 or clip > a.clip_total_tokens or actual != full+clip:
        raise RuntimeError('Student/teacher full video or separate clip budget mismatch')
    indices = clip_meta['frames_indices']
    ta = dict(mode='gt_clip_extra', actual_visual_tokens=actual, global_visual_tokens=full,
        clip_visual_tokens=clip, target_tokens=8192+a.clip_total_tokens, duration=pool['duration'],
        source_frame_indices=global_meta['frames_indices'], guidance_applied=True,
        user_prompt=content[2]['text']+'\n'+content[-1]['text'],
        clip=dict(requested_interval=[start,end], source_frame_indices=indices,
                  sampled_timestamp_range=[indices[0]/float(clip_meta['fps']), indices[-1]/float(clip_meta['fps'])],
                  metadata_audit=meta_audit), full_view_reallocated=False)
    return student, teacher, sa, ta, pool, pool['duration']


class Recipe(base.Recipe):
    prepare = staticmethod(prepare)
    training_inputs = staticmethod(training_inputs)


def main(argv=None):
    return base.run(parser().parse_args(argv), recipe=Recipe())


