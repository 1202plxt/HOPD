"""Round-frozen 8B self-teacher, GT clip + reference text, four two-GPU groups."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))


def parser():
    import train_qwen3_32b_direct_head_opd as original
    p = original.parser()
    p.description = __doc__
    for action in p._actions:
        if action.dest == 'teacher_path':
            action.required = False
            action.help = 'Omit: teacher starts from the student checkpoint'
    p.add_argument('--rounds', type=int, choices=(1, 2, 3), default=2)
    p.add_argument('--teacher-refresh', choices=('per-round', 'fixed'), default='per-round')
    p.add_argument('--candidate-policy', choices=('replay', 'disjoint'), default='replay',
                   help='Replay the shuffled training pool each round, or consume without reuse')
    p.add_argument('--without-reference-text', action='store_true', help='GT clip only ablation')
    return p


def prepare(a):
    import train_qwen3_32b_direct_head_opd as original
    if a.teacher_path and Path(a.teacher_path).resolve() != Path(a.student_path).resolve():
        raise ValueError('Self-teacher requires the same initial student and teacher checkpoint')
    if a.student_processor_path or a.teacher_processor_path:
        raise ValueError('Use the initial checkpoint processor for both self-teacher branches')
    a.teacher_path = a.student_path
    a._round_self_teacher = True
    a._gt_reference_text = not a.without_reference_text
    # original.prepare verifies head layout, budgets, gate and candidate identities.
    # max-attempts is PER ROUND; disjoint mode needs the full validated pool.
    limit = a.max_attempts
    if a.candidate_policy == 'disjoint':
        a.max_attempts = 0
    try:
        protocol, samples, roles = original.prepare(a)
    finally:
        a.max_attempts = limit
    protocol.update(method='qwen3_round_frozen_self_gtclip_gttext_opsd_dp4_mp2_v1',
        teacher_policy=dict(initial='Exact initial student checkpoint plus synchronized initial adapters',
            refresh=a.teacher_refresh, frozen_within_round=True, stop_gradient=True,
            normal_student_input=True, reference_text=a._gt_reference_text),
        rounds=a.rounds, accepted_target_per_round=a.max_samples,
        max_attempts_per_round=limit, candidate_policy=a.candidate_policy,
        optimizer_policy='One AdamW and continuous student adapters across rounds; no reset',
        source_note='Same-architecture self-distillation; no independent 32B teacher loaded')
    protocol['teacher_prompt_policy']['gt_answer_text'] = a._gt_reference_text
    protocol['teacher_prompt_policy']['reference_template'] = original.common.reference_interval_text(
        dict(gt_start=1., gt_end=2.)) if a._gt_reference_text else None
    protocol['placement'] = 'Four groups: 8B student on local GPU0, frozen 8B teacher on local GPU1'
    protocol['source_sha256'][Path(__file__).name] = original.common.sha(Path(__file__))
    return protocol, samples, roles


def load_runtime(a):
    import inspect
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import scripts.qwen_decode_worker as decoder
    if torch.version.cuda is None or torch.version.hip is not None or torch.cuda.device_count() != 2:
        raise RuntimeError('Require two visible NVIDIA CUDA GPUs per worker')
    torch.manual_seed(a.seed)
    models, processors = [], []
    # A separate frozen 8B teacher on GPU1, trainable student on GPU0.
    for device in (0, 1):
        model = AutoModelForImageTextToText.from_pretrained(a.student_path, local_files_only=True,
            trust_remote_code=True, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', device_map={'': device}).eval().requires_grad_(False)
        if model.config.model_type != 'qwen3_vl' or 'logits_to_keep' not in inspect.signature(model.forward).parameters:
            raise ValueError('Require Qwen3-VL with logits_to_keep support')
        processor = AutoProcessor.from_pretrained(a.student_path, local_files_only=True,
            trust_remote_code=True, padding_side='left', do_resize=False)
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        if tuple(int(getattr(processor.video_processor, k)) for k in
                 ('patch_size', 'merge_size', 'temporal_patch_size')) != (16, 2, 2):
            raise ValueError('Unexpected native processor geometry')
        models.append(model)
        processors.append(processor)
    decoder.configure_decoder(timeout=a.decode_timeout, attempts=3)
    return models[0], models[1], processors


def refresh_teacher(student_adapters, teacher_adapters):
    import torch
    if set(student_adapters) != set(teacher_adapters):
        raise ValueError('Self-teacher adapter module mismatch')
    with torch.no_grad():
        for name, student in student_adapters.items():
            teacher = teacher_adapters[name]
            if not torch.equal(student.indices.cpu(), teacher.indices.cpu()):
                raise ValueError('Self-teacher head index mismatch')
            for key in ('A', 'B'):
                source, target = getattr(student, key), getattr(teacher, key)
                if source.shape != target.shape:
                    raise ValueError('Self-teacher parameter shape mismatch')
                target.copy_(source.to(target.device))
                if not torch.equal(source.detach().cpu(), target.detach().cpu()):
                    raise RuntimeError('Self-teacher refresh copy mismatch')
                target.requires_grad_(False)


def Recipe():
    import train_qwen3_32b_direct_head_opd as original
    return original.Recipe()


if __name__ == '__main__':
    from train_qwen3_32b_head_opd_dp4_mp2 import main
    raise SystemExit(main(entry_override=sys.modules[__name__]))
