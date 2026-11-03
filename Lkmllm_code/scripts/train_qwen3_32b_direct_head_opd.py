"""NVIDIA two-GPU Qwen3 head OPD using a preselected head list, without probing."""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import probe_qwen32b_privileged_k100 as common
import train_qwen32b_qwen3_gtclip_opd_k100 as base
import train_qwen32b_qwen3_gtclip_extra8192_k100 as extra
import train_qwen32b_timelens8b_opd_k100 as engine


def parser():
    p = base.parser()
    p.description = __doc__
    p.epilog = 'Preselected heads; one fresh OPD round, no automatic resume or head screening.'
    for action in p._actions:
        if action.dest == 'analysis_summary':
            action.required = False
            action.help = 'Unused legacy option; omit it.'
        elif action.dest in ('teacher_gpu0_gib', 'teacher_gpu1_gib'):
            action.help = argparse.SUPPRESS  # Legacy parameters do not control automatic placement.
    p.add_argument('--heads-json', default=str(ROOT/'heads'/'qwen3_gtmask_top10.json'),
                   help='JSON containing a heads list of zero-based layer/head coordinates')
    p.add_argument('--input-scheme', choices=('extra8192', 'matched14336'), default='extra8192')
    p.add_argument('--student-device', type=int, default=0,
                   help='Visible GPU index for the student model and its training inputs')
    p.add_argument('--student-reserve-gib', type=float, default=12.,
                   help='Free activation/buffer margin after reserving student weights')
    p.add_argument('--teacher-reserve-gib', type=float, default=4.,
                   help='Free margin on each non-student GPU')
    p.set_defaults(total_tokens=8192, min_tokens=4, student_total_tokens=8192,
        teacher_total_tokens=8192, student_min_tokens=4, teacher_min_tokens=4,
        fps=2., max_frames=768, pool_factor=2, max_pool_frames=1536,
        clip_total_tokens=1024, clip_fps=4., clip_max_frames=128,
        clip_padding_seconds=0., lr=2e-5, grad_accum=4, save_every_steps=25)
    return p


def prepare(a):
    if a.teacher_mode != 'gt_clip' or a.gate != 'teacher-better' or a.epochs != 1:
        raise ValueError('Requires GT-clip teacher, strict teacher-better gate and one epoch')
    if a.selection_json or a.analysis_summary or a.head_selection != 'selected' or a.privilege_mix != 1:
        raise ValueError('Specify the selected head list only; legacy screening/selection options are not used')
    if (a.student_target_iou, a.teacher_min_iou, a.iou_margin) != (1., 0., 0.):
        raise ValueError('The teacher-better gate has no additional IoU threshold or margin')
    if a.input_scheme == 'extra8192':
        required = (8192, 4, 768)
    else:
        required = (14336, 64, 0)
    for role in ('student', 'teacher'):
        if (getattr(a, role+'_total_tokens'), getattr(a, role+'_min_tokens')) != required[:2]:
            raise ValueError(f'{a.input_scheme}: {role} video budget must match {required[:2]}')
    if (a.total_tokens, a.min_tokens, a.max_frames) != required:
        raise ValueError(f'{a.input_scheme}: total/min/max-frames must be {required}')
    if a.fps != 2. or a.pool_factor != 2 or a.max_pool_frames != 1536:
        raise ValueError('Requires fps=2, pool factor=2 and max pool frames=1536')
    if (min(a.rank, a.grad_accum, a.max_samples) < 1 or a.max_attempts < 0 or a.save_every_steps < 0
            or not all(math.isfinite(x) and x > 0 for x in (a.lr, a.alpha, a.max_grad_norm))):
        raise ValueError('Invalid optimizer or sample parameters')
    protocol, samples = common.validate_common(a, sample_limit=0)
    if a.max_attempts:
        samples = samples[:a.max_attempts]
    if len(samples) < a.max_samples:
        raise ValueError('Candidate pool is smaller than the accepted sample target')
    config = protocol['student_checkpoint']['config']['text_config']
    if (config['num_hidden_layers'], config['num_attention_heads'], config['hidden_size']) != (36, 32, 4096):
        raise ValueError('Bundled head coordinates require the expected Qwen3-VL-8B layout')
    dim = config.get('head_dim', config['hidden_size']//config['num_attention_heads'])
    if dim != 128:
        raise ValueError('Unexpected Qwen3-VL-8B head dimension')
    payload = common.read(a.heads_json)
    heads = payload['heads'] if isinstance(payload, dict) else payload
    if not isinstance(heads, list) or not heads:
        raise ValueError('Head list must be non-empty')
    coords = []
    for h in heads:
        if not isinstance(h, dict) or type(h.get('layer')) is not int or type(h.get('head')) is not int:
            raise ValueError('Each head must have integer layer and head fields')
        pair = h['layer'], h['head']
        if not (0 <= pair[0] < config['num_hidden_layers'] and 0 <= pair[1] < config['num_attention_heads']):
            raise ValueError(f'Head out of range: {pair}')
        coords.append(pair)
    if len(coords) != len(set(coords)):
        raise ValueError('Duplicate head coordinates')
    selected = [dict(layer=l, head=h) for l, h in coords]
    counts = Counter(l for l, _ in coords)
    extra_scheme = a.input_scheme == 'extra8192'
    protocol.update(method='qwen3_32b_direct_preselected_heads_opd_v1',
        selected_heads=selected, heads_path=str(Path(a.heads_json).resolve()),
        heads_sha256=common.sha(a.heads_json),
        head_provenance='Direct user-supplied coordinates; no runtime probe or GT-mask analysis',
        model_identity_note='Architecture is checked; config/index hashes and shard sizes do not prove weight identity',
        expected_trainable_parameters=2*a.rank*sum(config['hidden_size']+n*dim for n in counts.values()),
        candidate_count=len(samples), accepted_target=a.max_samples,
        candidate_samples_sha256=hashlib.sha256(json.dumps(samples,sort_keys=True).encode()).hexdigest(),
        objective='Full-vocabulary KL(student || stop-gradient frozen teacher) on sampled student continuation',
        gate=dict(mode='teacher-better',decision_rule='teacher valid and teacher greedy IoU > sampled student IoU'),
        placement='Teacher automatically sharded across available GPUs with student and activation reserves; student on selected GPU',
        video_protocol=('Same complete video for student and teacher; GT clip added outside 8192 cap' if extra_scheme else
                        'Teacher global video plus GT clip share the 14336 cap'),
        clip_policy=dict(maximum_visual_tokens=a.clip_total_tokens,included_in_teacher_cap=not extra_scheme,
                         fps=a.clip_fps,max_frames=a.clip_max_frames,padding_seconds=a.clip_padding_seconds),
        privilege='GT selects teacher reference clip and is used for training IoU gate; not student inference',
        inference='Exported student uses normal video and query only',
        resume='Weights-only snapshots; optimizer/RNG resume not implemented')
    for name in (Path(__file__).name, 'train_qwen32b_qwen3_gtclip_opd_k100.py',
                 'train_qwen32b_qwen3_gtclip_extra8192_k100.py', 'train_qwen32b_timelens8b_opd_k100.py'):
        protocol['source_sha256'][name] = common.sha(ROOT/'scripts'/name)
    return protocol, samples, {}


class Recipe(base.Recipe):
    prepare = staticmethod(prepare)

    @staticmethod
    def training_inputs(processors, sample, a):
        if a.input_scheme == 'extra8192':
            return extra.training_inputs(processors, sample, a)
        return engine.training_inputs(processors, sample, a)


if __name__ == '__main__':
    raise SystemExit(base.run(parser().parse_args(), recipe=Recipe()))
