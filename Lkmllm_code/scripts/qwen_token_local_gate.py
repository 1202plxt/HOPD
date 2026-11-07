"""Gate real token substitutions by independent boundary absolute error."""
import re
import math

PAT = re.compile(r'event\s+(?:happens|occurs)\s+in\s+(\d+(?:\.\d+)?)\s*(?:[-–—]|to)\s*(\d+(?:\.\d+)?)\s*seconds?', re.I)
SECONDS_INSTRUCTION = (
    "\nExpress both timestamps as decimal numbers in seconds on the original full-video timeline. "
    "Do not use minutes, clock notation, or colons in either timestamp. "
    "Return exactly one sentence: The event happens in <start_seconds> - <end_seconds> seconds."
)
TOP10_POLICY = dict(k=10, candidates='teacher Top10 on the actual student prefix; independent real-token substitutions; original suffix retained',
    criterion='teacher probability-weighted absolute error must be strictly below both student probability-weighted error on the SAME valid candidate set and original sampled boundary error',
    boundary_validity='Nonnegative finite numeric replacement; other field/content unchanged; hypothetical start/end crossing allowed; original rollout remains unchanged',
    normalization='each distribution renormalized over valid teacher-Top10 replacements; excluded/remaining probability mass logged, not certified',
    sample_acceptance='at least one accepted boundary position; otherwise no backward, accumulation or accepted-sample increment',
    loss='unchanged full-vocabulary KL on ordinary positions plus accepted boundary positions; outside-Top10 noise is not guaranteed removed')


def seconds_messages(builder):
    def wrapped(*args, **kwargs):
        messages = builder(*args, **kwargs)
        for message in reversed(messages):
            if message['role'] == 'user':
                for item in reversed(message['content']):
                    if item.get('type') == 'text':
                        if SECONDS_INSTRUCTION not in item['text']:
                            item['text'] += SECONDS_INSTRUCTION
                        return messages
        raise ValueError('No user text to append seconds-only instruction')
    return wrapped


def gate_change(row):
    if 'teacher_expected_error' in row:
        return (f"error_original={row['original_error']:.6g} "
                f"E_student={row['student_expected_error']:.6g} E_teacher={row['teacher_expected_error']:.6g}s "
                f"valid_mass_teacher={row['teacher_valid_mass']:.6g} "
                f"gain_vs_sample={row['gain_vs_sample_seconds']:.6g} "
                f"gain_vs_student={row['gain_vs_student_seconds']:.6g} {row['reason']}")
    return row['reason']


def validate_seconds_answer(tokenizer, ids):
    text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if not re.fullmatch(r'The\s+event\s+(?:happens|occurs)\s+in\s+\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\s+seconds?\.?', text.strip(), re.I):
        raise ValueError('Expected one seconds-only answer; no colon or padding')


def topk_gate_rows(tokenizer, ids, groups, candidate_rows, gt, epsilon=0.):
    """Pure-Python numeric gate. candidate_rows contains detached full-vocab log probabilities."""
    validate_seconds_answer(tokenizer, ids)
    kept, audit = [], []
    for field, positions in enumerate(groups):
        for pos in positions:
            evaluated = []
            for candidate in candidate_rows[pos]:
                proposals = list(ids)
                proposals[pos] = candidate['token_id']
                isolated = [[], []]; isolated[field] = [pos]
                _, _, rows = local_gate(tokenizer, ids, isolated, proposals, gt, epsilon)
                evaluated.append(dict(rows[0], **candidate))
            valid = [r for r in evaluated if 'candidate_error' in r]
            first = evaluated[0]
            row = dict(position=pos, field=('start','end')[field], student_piece=first['student_piece'],
                teacher_piece=first['teacher_piece'], original_value=first['original_value'],
                original_error=first['original_error'], accepted=False, top_k=len(evaluated),
                candidates=evaluated, valid_candidates=len(valid), reason='no_valid_top10_candidates')
            if valid:
                means = []
                for side in ('teacher','student'):
                    logs = [r[side+'_log_probability'] for r in valid]
                    maximum = max(logs)
                    weights = [math.exp(v-maximum) for v in logs]
                    total = sum(weights)
                    mean = sum(w*r['candidate_error'] for w,r in zip(weights,valid))/total
                    means.append(mean)
                    row[side+'_expected_error'] = mean
                    row[side+'_valid_mass'] = sum(math.exp(v) for v in logs)
                row['teacher_topk_mass'] = sum(math.exp(r['teacher_log_probability']) for r in evaluated)
                row['student_topk_mass'] = sum(math.exp(r['student_log_probability']) for r in evaluated)
                row['gain_vs_sample_seconds'] = row['original_error']-means[0]
                row['gain_vs_student_seconds'] = means[1]-means[0]
                row['gain_seconds'] = min(row['gain_vs_sample_seconds'], row['gain_vs_student_seconds'])
                row['accepted'] = row['gain_seconds'] > epsilon
                row['reason'] = 'top10_expected_error_improved' if row['accepted'] else 'top10_expected_error_not_improved'
                if row['accepted']: kept.append(pos)
            audit.append(row)
    excluded = set(groups[0]+groups[1])-set(kept)
    return ([i for i in range(len(ids)) if i not in excluded] if kept else []), kept, audit


def topk_gate(tokenizer, ids, groups, teacher_scores, student_scores, gt, epsilon=0., k=10):
    import torch
    positions = sorted(set(groups[0]+groups[1]))
    if teacher_scores.shape != student_scores.shape or teacher_scores.shape[0] != len(ids):
        raise ValueError('Top10 gate teacher/student scores must share the actual trajectory and vocabulary')
    with torch.no_grad():
        teacher = teacher_scores[positions].detach().float().log_softmax(-1)
        values, indices = teacher.topk(min(k,teacher.shape[-1]), dim=-1)
        student = student_scores[positions].detach().float().log_softmax(-1)
        probabilities = student.gather(-1, indices.to(student.device)).cpu().tolist()
        values, indices = values.cpu().tolist(), indices.cpu().tolist()
    rows = {p:[dict(token_id=i,teacher_log_probability=t,student_log_probability=s)
               for i,t,s in zip(indices[j],values[j],probabilities[j])] for j,p in enumerate(positions)}
    return topk_gate_rows(tokenizer, ids, groups, rows, gt, epsilon)

def local_gate(tokenizer, ids, groups, teacher_ids, gt, epsilon=0.):
    decode = lambda x: tokenizer.decode(x, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    original = decode(ids)
    matches = list(PAT.finditer(original))
    if len(matches) != 1:
        raise ValueError('Ambiguous timestamp answer')
    m = matches[0]
    values = [float(m.group(1)), float(m.group(2))]
    keep, audit = [], []
    for field, positions in enumerate(groups):
        for pos in positions:
            proposal = int(teacher_ids[pos])
            row = dict(position=pos, field=('start','end')[field], student_token=ids[pos], teacher_token=proposal, accepted=False,
                       original_value=values[field], gt_value=gt[field],
                       original_error=abs(values[field]-gt[field]),
                       student_piece=decode([ids[pos]]), teacher_piece=decode([proposal]))
            changed = list(ids); changed[pos] = proposal
            text = decode(changed); found = list(PAT.finditer(text))
            if len(found) != 1 or proposal in tokenizer.all_special_ids:
                row['reason']='invalid_replacement'
            else:
                n = found[0]; new = [float(n.group(1)), float(n.group(2))]
                # A fused token may not modify formatting or the other field.
                left,right=m.span(field+1); nl,nr=n.span(field+1)
                if (not all(math.isfinite(v) for v in new) or text[:nl]!=original[:left]
                        or text[nr:]!=original[right:]):
                    row['reason']='changed_other_content_or_nonfinite_boundary'
                else:
                    gain=abs(values[field]-gt[field])-abs(new[field]-gt[field])
                    row.update(candidate_value=new[field], candidate_error=abs(new[field]-gt[field]),
                               candidate_interval_valid=new[0]<new[1], independent_boundary=True,
                               gain_seconds=gain,accepted=gain>epsilon,reason='improved' if gain>epsilon else 'not_improved')
                    if gain>epsilon: keep.append(pos)
            audit.append(row)
    excluded=set(groups[0]+groups[1])-set(keep)
    return [i for i in range(len(ids)) if i not in excluded], keep, audit


def completion_positions(tokenizer, ids):
    decode = lambda x: tokenizer.decode(x, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    text = decode(ids)
    matches = list(PAT.finditer(text))
    if len(matches) != 1:
        raise ValueError('Ambiguous timestamp answer')
    spans=[]; previous=0
    for i in range(len(ids)):
        prefix=decode(ids[:i+1])
        if not text.startswith(prefix) or len(prefix)<previous:
            raise ValueError('Unstable timestamp token offsets')
        spans.append((previous,len(prefix))); previous=len(prefix)
    groups=[]
    for field in (1,2):
        start,end=matches[0].span(field)
        group=[i for i,(left,right) in enumerate(spans) if left<end and right>start]
        # Include the first decision which ends the number, including a fused
        # numeric/terminator token. This position must not bypass the gate.
        termination=next((i for i,(left,right) in enumerate(spans) if right>end or left>=end),None)
        if termination is None:
            raise ValueError('No timestamp termination position')
        groups.append(sorted(set(group+[termination])))
    if set(groups[0]) & set(groups[1]):
        raise ValueError('Shared start/end token cannot be independently gated')
    return text,matches[0],groups
