"""Gate real token substitutions by independent boundary absolute error."""
import re
import math

PAT = re.compile(r'event\s+(?:happens|occurs)\s+in\s+(\d+(?:\.\d+)?)\s*(?:[-–—]|to)\s*(\d+(?:\.\d+)?)\s*seconds?', re.I)

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
                        or text[nr:]!=original[right:] or new[0]>=new[1]):
                    row['reason']='changed_other_content_or_invalid_interval'
                else:
                    gain=abs(values[field]-gt[field])-abs(new[field]-gt[field])
                    row.update(candidate_value=new[field], candidate_error=abs(new[field]-gt[field]),
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

