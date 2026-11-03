"""Select final-answer timestamp prediction rows from actual TaRO rollout IDs.

No GT answer or re-tokenized text is substituted into the student rollout.
Time punctuation (decimal point / clock colon) belongs to the timestamp;
reasoning, tags, units, separators and EOS do not get independent loss rows.
"""
import math

try:
    from .taro_protocol import FINAL_ANSWER_RE, time_to_seconds
except ImportError:
    from taro_protocol import FINAL_ANSWER_RE, time_to_seconds


def timestamp_selection(tokenizer, token_ids):
    ids = list(token_ids)
    decode = lambda values: tokenizer.decode(
        values, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    text = decode(ids)
    matches = list(FINAL_ANSWER_RE.finditer(text))
    if len(matches) != 1:
        raise ValueError('expected exactly one complete final <answer> interval')
    match = matches[0]
    if match.start() < text.lower().rfind('</think>'):
        raise ValueError('answer interval occurs inside reasoning')
    # An open reasoning block must be closed before the final answer.
    before = text[:match.start()].lower()
    if before.rfind('<think>') > before.rfind('</think>'):
        raise ValueError('unclosed reasoning block before answer')
    start, end = (time_to_seconds(match.group(i)) for i in (1, 2))
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        raise ValueError('invalid generated start/end interval')
    spans = [match.span(1), match.span(2)]
    # Prefer offsets only if re-encoding reproduces the ACTUAL generated IDs.
    offsets = None
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if list(encoded['input_ids']) == ids:
            offsets = [tuple(pair) for pair in encoded['offset_mapping']]
    except (TypeError, ValueError, NotImplementedError):
        pass
    if offsets is None:
        # Slow-tokenizer / noncanonical segmentation fallback. Prefix decoding
        # can be unstable inside Unicode pieces; only stable timestamp pieces
        # are eligible, and full timestamp coverage is checked below.
        offsets = []
        previous = ''
        for i in range(1, len(ids) + 1):
            current = decode(ids[:i])
            stable = (text.startswith(previous) and text.startswith(current)
                      and len(current) >= len(previous))
            offsets.append((len(previous), len(current)) if stable else (0, 0))
            previous = current
    if len(offsets) != len(ids):
        raise ValueError('generated ID/offset length mismatch')
    masks = []
    fused = set()
    for left, right in spans:
        mask = [a < right and b > left for a, b in offsets]
        covered = set()
        for i, ((a, b), keep) in enumerate(zip(offsets, mask)):
            if keep:
                covered.update(range(max(a, left), min(b, right)))
                if a < left or b > right:
                    fused.add(i)
        if not any(mask) or covered != set(range(left, right)):
            raise ValueError('cannot align all generated timestamp characters')
        masks.append(mask)
    if any(a and b for a, b in zip(*masks)):
        raise ValueError('one token spans both timestamps')
    indices = [i for i, (a, b) in enumerate(zip(*masks)) if a or b]
    return dict(indices=indices, start_indices=[i for i, v in enumerate(masks[0]) if v],
                end_indices=[i for i, v in enumerate(masks[1]) if v],
                timestamp_text=[match.group(1), match.group(2)],
                boundary_fused_tokens=len(fused))


def select_prediction_rows(logits, indices):
    """Select causal token positions, retaining every vocabulary column."""
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('timestamp loss needs unique nonempty indices')
    if min(indices) < 0 or max(indices) >= logits.shape[1]:
        raise ValueError('timestamp index outside rollout prediction rows')
    return logits[:, indices, :]


def accumulation_scale(group_size, valid_samples):
    """Correct provisional loss/group_size to mean over valid samples only."""
    if not 0 <= valid_samples <= group_size or group_size < 1:
        raise ValueError('invalid gradient accumulation counts')
    return group_size / valid_samples if valid_samples else None
