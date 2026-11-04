"""Strict boundary-error gates and mapping of ACTUAL generated token IDs."""
import math
import re

ANSWER = re.compile(r'\s*The event happens in\s+(\d+(?:\.\d+)?)\s*-\s*'
                    r'(\d+(?:\.\d+)?)\s+seconds\.?\s*', re.I)


def boundary_decision(student, teacher, gt, epsilon=0.):
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError('epsilon must be finite and nonnegative (seconds)')
    if not student.get('valid') or not teacher.get('valid'):
        return None, 'invalid_student_or_teacher'
    s, t = student.get('prediction'), teacher.get('prediction')
    if any(not isinstance(v, (list, tuple)) or len(v) != 2 for v in (s, t, gt)):
        return None, 'invalid_boundary_values'
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for p in (s, t, gt) for v in p):
        return None, 'invalid_boundary_values'
    if any(not 0 <= p[0] < p[1] for p in (s, t, gt)):
        return None, 'invalid_boundary_values'
    se = [abs(s[i] - gt[i]) for i in (0, 1)]
    te = [abs(t[i] - gt[i]) for i in (0, 1)]
    gain = [se[i] - te[i] for i in (0, 1)]
    result = dict(student_abs_errors=se, teacher_abs_errors=te,
                  start_gain=gain[0], end_gain=gain[1],
                  start=gain[0] > epsilon, end=gain[1] > epsilon)
    return result, 'boundary_improved' if result['start'] or result['end'] else 'no_boundary_improved'


def boundary_token_groups(tokenizer, ids):
    """Use monotonic prefix decoding; never re-encode a canonical/GT answer.

    A selected token may contain adjacent whitespace but no other answer text.
    Non-monotonic decoding and tokens spanning both numbers are rejected.
    """
    def decode(seq):
        return tokenizer.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    text = decode(ids)
    match = ANSWER.fullmatch(text)
    if not match:
        raise ValueError('unsupported_answer_format')
    offsets, previous = [], 0
    for index in range(1, len(ids) + 1):
        prefix = decode(ids[:index])
        if not text.startswith(prefix) or len(prefix) < previous:
            raise ValueError('unstable_token_offsets')
        offsets.append((previous, len(prefix)))
        previous = len(prefix)
    groups = []
    for field in (1, 2):
        left, right = match.span(field)
        indices = [i for i, (a, b) in enumerate(offsets) if a < right and b > left]
        if not indices or offsets[indices[0]][0] > left or offsets[indices[-1]][1] < right:
            raise ValueError('uncovered_boundary')
        groups.append(indices)
    if set(groups[0]) & set(groups[1]):
        raise ValueError('shared_start_end_token')
    for field, indices in enumerate(groups, 1):
        left, right = match.span(field)
        for index in indices:
            a, b = offsets[index]
            if text[a:min(left, b)].strip() or text[max(right, a):b].strip():
                raise ValueError('mixed_number_and_text_token')
    return groups


def selected_indices(decision, groups):
    return sorted((groups[0] if decision['start'] else []) + (groups[1] if decision['end'] else []))


def reverse_kl(student_logits, teacher_log_probs, indices):
    """One example: mean KL over only selected positions; full vocab at each."""
    import torch
    if not indices:
        raise ValueError('No selected tokens')
    if student_logits.shape != teacher_log_probs.shape:
        raise ValueError('Student/teacher logits shape mismatch')
    log_p = student_logits[:, indices, :].float().log_softmax(-1)
    log_q = teacher_log_probs[:, indices, :].detach().to(log_p.device)
    return (log_p.exp() * (log_p - log_q)).sum(-1).mean()


if __name__ == '__main__':
    import sys
    from train_qwen3_32b_head_opd_dp4_mp2 import main
    args = sys.argv[1:]
    if '--boundary-selective' not in args:
        args.append('--boundary-selective')
    raise SystemExit(main(args))
