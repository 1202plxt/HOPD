"""Frozen TimeLens-8B GT-clip teacher -> Qwen3 GT-mask10, one 40GB GPU.

Both native visual caps are 4096. Reuses the existing 32B-probe module only
for its video/crop, scoring and JSON helpers; no 32B model is loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
import probe_qwen32b_privileged_k100 as common
import train_timelens_three_group_opsd as core


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('teacher-path','student-path','anno-json','video-dir','output-dir','analysis-summary'):
        p.add_argument('--'+name,required=True)
    for name in ('teacher-processor-path','student-processor-path'):
        p.add_argument('--'+name)
    for name,default in (('max-samples',1000),('seed',42),('total-tokens',4096),('min-tokens',4),
                         ('pool-factor',2),('max-pool-frames',1024),('clip-total-tokens',1024),
                         ('clip-max-frames',128),('max-new-tokens',80)):
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--fps',type=float,default=2.)
    p.add_argument('--clip-fps',type=float,default=4.)
    p.add_argument('--clip-padding-seconds',type=float,default=0.)
    p.add_argument('--decode-timeout',type=float,default=600.)
    p.add_argument('--plan-only',action='store_true')
    p.set_defaults(boundary_seconds=2.)
    p.add_argument('--teacher-mode', choices=('gt_clip','normal','gt_clip_blank'), default='gt_clip')
    p.add_argument('--head-selection', choices=('selected','random-matched'), default='selected')
    p.add_argument('--max-attempts', type=int, default=10000,
                   help='Maximum candidate events; 0 scans all. max-samples counts accepted events.')
    p.add_argument('--max-frames', type=int, default=1024)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--alpha', type=float, default=16.)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--top-p', type=float, default=.9)
    p.add_argument('--max-grad-norm', type=float, default=1.)
    p.add_argument('--save-every-steps', type=int, default=25)
    p.add_argument('--offload-saved-tensors', action='store_true',
                   help='Offload saved student activations to CPU RAM; model weights stay on GPU.')
    p.add_argument('--skip-merge', action='store_true')
    return p


def read_heads(summary_path, model_path, config):
    """Verify the actual completed single-GPU strict-v5 Qwen3 diagnostic."""
    path = Path(summary_path)
    data = common.read(path)
    protocol = common.read(path.parent/'protocol.json')
    a = protocol['arguments']
    if (protocol.get('method') != 'qwen3_gt_only_answer_iou_rank_strict_v5'
            or (a.get('shard_index'),a.get('shard_count'),a.get('target')) != (0,1,300)
            or Path(a['model_path']).resolve() != Path(model_path).resolve()
            or (a.get('fps'),a.get('min_tokens'),a.get('total_tokens'),a.get('max_frames')) != (2.,4,4096,1024)):
        raise ValueError('Require the student own strict-v5 single-GPU 300-pair 4096 diagnostic')
    if (data.get('complete') is not True or data.get('n_paired') != 300
            or data.get('target') != 300 or len(protocol.get('heads',[])) != 30):
        raise ValueError('Diagnostic is incomplete or does not contain 30 candidates')
    allowed = {(h['layer'],h['head']) for h in protocol['heads']}
    ranking = data.get('ranking',[])
    base = float(data['plain']['mIoU'])
    if len(allowed) != 30 or len(ranking) != 30 or not math.isfinite(base):
        raise ValueError('Invalid diagnostic ranking/baseline')
    seen, positive = set(), []
    for row in ranking:
        head = row['head']
        l,h = head['layer'],head['head']
        damage = float(row['selected_gt_damage_pp'])
        actual = 100*(base-float(row['metrics']['selected_gt']['mIoU']))
        if (type(l) is not int or type(h) is not int or (l,h) in seen or (l,h) not in allowed
                or not 0 <= l < config['num_hidden_layers'] or not 0 <= h < config['num_attention_heads']
                or not math.isfinite(damage) or abs(damage-actual) > 1e-7):
            raise ValueError('Invalid/duplicate head or inconsistent GT damage score')
        seen.add((l,h))
        if damage > 0:
            positive.append(dict(layer=l,head=h,diagnostic_score_pp=damage))
    positive.sort(key=lambda h:(-h['diagnostic_score_pp'],h['layer'],h['head']))
    selected = positive[:10]
    if len(selected) != 10:
        raise ValueError('Require ten positive-damage heads; do not fill with zero/negative scores')
    if data.get('positive_head_count',len(positive)) != len(positive):
        raise ValueError('Positive-head count differs from the full ranking')
    if 'selected_heads' in data and data['selected_heads'] != selected:
        raise ValueError('Saved selected_heads differs from verified ranking')
    return selected


def prepare(a):
    if int(os.environ.get('WORLD_SIZE','1')) != 1:
        raise ValueError('One process only; do not use torchrun')
    if 'train' not in {p.lower() for p in Path(a.anno_json).parts}:
        raise ValueError('Use training annotations under Train')
    if (a.total_tokens,a.min_tokens,a.fps,a.max_frames) != (4096,4,2.,1024):
        raise ValueError('Preserve the student diagnostic budget: 4096/min4/fps2/maxframes1024')
    if (min(a.max_samples,a.rank,a.grad_accum,a.max_new_tokens) < 1 or a.max_attempts < 0
            or a.save_every_steps < 0 or a.pool_factor < 2 or a.max_pool_frames < 4
            or a.max_pool_frames % 2 or a.clip_max_frames < 4 or a.clip_max_frames % 2
            or not 0 < a.clip_total_tokens < a.total_tokens):
        raise ValueError('Invalid sample/optimizer/video counts')
    if (not all(math.isfinite(x) and x > 0 for x in (a.lr,a.alpha,a.max_grad_norm,a.clip_fps,a.decode_timeout,a.temperature))
            or not math.isfinite(a.top_p) or not 0 < a.top_p <= 1
            or not math.isfinite(a.clip_padding_seconds) or a.clip_padding_seconds < 0):
        raise ValueError('Invalid positive settings or clip padding')
    if Path(a.output_dir).exists() and any(Path(a.output_dir).iterdir()):
        raise FileExistsError('Use a new empty output directory; no implicit resume')
    if Path(a.student_path).resolve() == Path(a.teacher_path).resolve():
        raise ValueError('Use distinct student and teacher checkpoints')
    # This helper in the original 32B probe verifies a local Qwen3-VL family
    # checkpoint; it imposes no teacher-size rule. Both 8B models are supported.
    student = common.checkpoint_identity(a.student_path)
    teacher = common.checkpoint_identity(a.teacher_path)
    cfg = student['config']['text_config']
    tc = teacher['config']['text_config']
    if any(cfg[k] != tc[k] for k in ('hidden_size','num_hidden_layers','num_attention_heads')):
        raise ValueError('This entry requires the matching Qwen3-VL 8B layouts; inspect actual configs')
    original = read_heads(a.analysis_summary,a.student_path,cfg)
    heads = original
    if a.head_selection == 'random-matched':
        rng = random.Random(a.seed)
        occupied = {(h['layer'],h['head']) for h in original}
        heads = [dict(layer=l,head=h) for l,n in sorted(Counter(h['layer'] for h in original).items())
                 for h in rng.sample([i for i in range(cfg['num_attention_heads']) if (l,i) not in occupied],n)]
    samples,stats = core.load_samples(a.anno_json,a.video_dir,0,a.seed)
    if a.max_attempts:
        samples = samples[:a.max_attempts]
    if len(samples) < a.max_samples:
        raise ValueError('Candidate pool smaller than accepted target')
    dim = cfg.get('head_dim',cfg['hidden_size']//cfg['num_attention_heads'])
    counts = Counter(h['layer'] for h in heads)
    sources = [Path(__file__),Path(common.__file__),Path(core.__file__),ROOT/'scripts/qwen_decode_worker.py']
    protocol = dict(method='timelens8b_gtclip_to_qwen3_gtmask10_4096_single_gpu_opd_v1',
        arguments=vars(a),student_checkpoint=student,teacher_checkpoint=teacher,
        selected_heads=heads,original_gtmask_heads=original,
        analysis_sha256=common.sha(a.analysis_summary),
        analysis_protocol_sha256=common.sha(Path(a.analysis_summary).parent/'protocol.json'),
        source_sha256={p.name:common.sha(p) for p in sources},annotation_sha256=common.sha(a.anno_json),
        data_stats=stats,candidate_count=len(samples),accepted_target=a.max_samples,
        candidate_samples_sha256=hashlib.sha256(json.dumps(samples,sort_keys=True).encode()).hexdigest(),
        expected_trainable_parameters=2*a.rank*sum(cfg['hidden_size']+n*dim for n in counts.values()),
        objective='Full-vocabulary KL(student || stop-gradient frozen teacher) on fresh student sampled prefixes, including EOS',
        rollout_sampling=dict(temperature=a.temperature,top_p=a.top_p,top_k=0,repetition_penalty=1.),
        token_alignment='Exact tokenizer vocabulary, normalization and segmentation plus output-width checks before KL',
        gate='Independent unrestricted greedy predictions; valid teacher IoU > current student IoU; invalid student=0; ties rejected; no floor or margin',
        reference_failure_policy='Log and skip samples whose decoded reference budget cannot retain two global groups and one clip group; other input/runtime failures abort with failure.json',
        placement='Both BF16 models on the sole visible cuda:0; teacher sequential no-grad scoring; only student Q/O compact LoRA trainable',
        video_protocol=dict(student='Native normal full video/query, fps2/min4/4096/maxframes1024; no GT-based selection',
                            teacher='Independent native dense-pool global context plus GT reference, original source timestamps',
                            student_visual_cap=4096,teacher_visual_cap=4096,
                            equal_actual_tokens_required=False,clip_allocation_in_teacher_cap=True),
        teacher_mode=a.teacher_mode,teacher_guidance=common.GT_CLIP_GUIDANCE if a.teacher_mode != 'normal' else None,
        privilege='GT selects the teacher crop and training IoU gate; student receives only normal video/query' if a.teacher_mode != 'normal' else 'GT only for training IoU gate',
        shortcut_note='GT crop and absolute timestamps expose approximate target location; teacher improvement does not prove visual reasoning or held-out student gains',
        inference='Exported student uses normal video/query only; no GT, crop or teacher',
        resume='Weights-only snapshots; optimizer/RNG resume is not implemented',
        evidence_status='Code and CPU checks only; actual single-40GB memory fit and GPU training unverified')
    return protocol,samples


def memory():
    import torch
    free,total = torch.cuda.mem_get_info(0)
    return dict(allocated_gib=torch.cuda.memory_allocated(0)/1024**3,
                reserved_gib=torch.cuda.memory_reserved(0)/1024**3,
                peak_allocated_gib=torch.cuda.max_memory_allocated(0)/1024**3,
                free_gib=free/1024**3,total_gib=total/1024**3)


def runtime_identity():
    import torch
    versions = {}
    for name in ('torch','transformers','accelerate','qwen-vl-utils','decord'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(versions=versions,hip=torch.version.hip,
                devices=[dict(index=0,name=torch.cuda.get_device_name(0),
                              total_gib=torch.cuda.get_device_properties(0).total_memory/1024**3)])


def load_runtime(a):
    import torch
    from transformers import AutoModelForImageTextToText,AutoProcessor
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU with CUDA_VISIBLE_DEVICES=0')
    torch.cuda.set_device(0)
    torch.manual_seed(a.seed)
    processors = [AutoProcessor.from_pretrained(path,local_files_only=True,trust_remote_code=True,
                    do_resize=False,padding_side='left') for path in
                  (a.student_processor_path or a.student_path,a.teacher_processor_path or a.teacher_path)]
    for p in processors:
        p.tokenizer.padding_side = 'left'
        if p.tokenizer.pad_token_id is None:
            p.tokenizer.pad_token = p.tokenizer.eos_token
        if tuple(int(getattr(p.video_processor,k)) for k in ('patch_size','merge_size','temporal_patch_size')) != (16,2,2):
            raise ValueError('Both models require their native Qwen3-VL 16/2/2 video processor')
    common.tokenizer_equivalence(processors[0].tokenizer,processors[1].tokenizer)
    models = []
    for path in (a.student_path,a.teacher_path):
        model = AutoModelForImageTextToText.from_pretrained(path,local_files_only=True,trust_remote_code=True,
                    torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':0})
        model.eval().requires_grad_(False)
        if ('logits_to_keep' not in inspect.signature(model.forward).parameters
                or any(p.device.type != 'cuda' or p.device.index != 0 for p in model.parameters())):
            raise RuntimeError('Require logits_to_keep support and all weights on the sole GPU')
        models.append(model)
    student,teacher = models
    if student.get_output_embeddings().weight.shape[0] != teacher.get_output_embeddings().weight.shape[0]:
        raise ValueError('Teacher/student output vocab widths differ; direct full-vocabulary KL is invalid')
    from scripts import qwen_decode_worker as decoder
    decoder.configure_decoder(timeout=a.decode_timeout,attempts=3)
    print(json.dumps(dict(student_parameters=sum(p.numel() for p in student.parameters()),
                          teacher_parameters=sum(p.numel() for p in teacher.parameters()),memory=memory())),flush=True)
    return student,teacher,processors


class ReferenceUnavailable(ValueError):
    pass


def training_inputs(processors,sample,a):
    s,sa = common.normal_input(processors[0],sample,a.total_tokens,a.min_tokens,a.fps,a.max_frames)
    pool = common.decode_pool(processors[1],sample,a)
    if a.teacher_mode != 'normal' and (pool['keep'] < 3 or pool['per_group'] > a.clip_total_tokens):
        raise ReferenceUnavailable('Cannot fit two global groups plus a reference group: '
                                   f'keep={pool["keep"]}, tokens/group={pool["per_group"]}, clip cap={a.clip_total_tokens}')
    t,ta = common.build_input(processors[1],sample,pool,a.teacher_mode,a)
    duration = sa['duration']
    if (abs(duration-pool['duration']) > .1
            or not 0 <= sample['gt_start'] < sample['gt_end'] <= duration):
        raise ValueError('Original duration disagreement or GT outside video')
    if any(not 0 < audit['actual_visual_tokens'] <= a.total_tokens for audit in (sa,ta)):
        raise ValueError('Actual native visual tokens exceed the 4096 cap')
    return s,t,sa,ta,duration


def accept(student,teacher):
    if not teacher['valid']:
        return False,'teacher_parse_failure'
    s,t = student['iou'] if student['valid'] else 0.,teacher['iou']
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (s,t)):
        raise ValueError('Invalid gate IoU')
    return (True,'teacher_strictly_better') if t > s else (False,'teacher_not_better')


def sample_rollout(model,processor,inputs,sample,duration,a):
    """Fresh on-policy student rollout; sampling matches the earlier OPD recipe."""
    import torch
    model.eval()
    core.reset_rope(model)
    cfg = getattr(model.config,'text_config',model.config)
    if inputs['input_ids'].shape[1]+a.max_new_tokens > cfg.max_position_embeddings:
        raise ValueError('Prompt plus continuation exceeds model context')
    with torch.no_grad():
        output = model.generate(**inputs,do_sample=True,num_beams=1,num_return_sequences=1,
            max_new_tokens=a.max_new_tokens,use_cache=True,repetition_penalty=1.,logits_to_keep=1,
            output_attentions=False,output_hidden_states=False,pad_token_id=processor.tokenizer.pad_token_id,
            temperature=a.temperature,top_p=a.top_p,top_k=0)
    y = output[:,inputs['input_ids'].shape[1]:].detach().cpu().clone()
    del output
    if y.numel() == 0:
        raise RuntimeError('Empty continuation')
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos,list) else [eos]
    truncated = y.shape[1] >= a.max_new_tokens and int(y[0,-1]) not in eos
    result = common.parse_prediction(processor.tokenizer.decode(y[0].tolist(),skip_special_tokens=True),
        [sample['gt_start'],sample['gt_end']],duration,truncated)
    result['tokens'] = y.shape[1]
    return y,result


def digest(params):
    h = hashlib.sha256()
    for name,p in params:
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(a):
    protocol,samples = prepare(a)
    print(json.dumps(protocol,ensure_ascii=False,indent=2),flush=True)
    if a.plan_only:
        return 0
    import torch
    from scripts import qwen_decode_worker as decoder
    out = Path(a.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    common.write(out/'training_metadata.json',protocol)
    common.write(out/'candidate_samples.json',samples)
    trained = []
    common.write(out/'train_samples.json',trained)
    accepted = processed = steps = pending = 0
    reasons = Counter()
    try:
        student,teacher,processors = load_runtime(a)
        adapters = core.attach_head_lora(student,protocol['selected_heads'],a.rank,a.alpha,['q_proj','o_proj'])
        params = sorted((n,p) for n,p in student.named_parameters() if p.requires_grad)
        if ({id(p) for n,p in params} != {id(p) for x in adapters.values() for p in (x.A,x.B)}
                or sum(p.numel() for n,p in params) != protocol['expected_trainable_parameters']):
            raise RuntimeError('Trainable parameters differ from selected-head Q/O recipe')
        for module in student.modules():
            if isinstance(module,torch.nn.Dropout):
                module.p = 0.
            if hasattr(module,'attention_dropout'):
                module.attention_dropout = 0.
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        student.config.use_cache = False
        optimizer = torch.optim.AdamW([p for n,p in params],lr=a.lr,weight_decay=0.)
        optimizer.zero_grad(set_to_none=True)
        initial = digest(params)
        protocol.update(trainable_parameters=sum(p.numel() for n,p in params),runtime=runtime_identity())
        common.write(out/'training_metadata.json',protocol)

        def update():
            nonlocal steps,pending
            if not pending:
                return
            for name,p in params:
                if p.grad is None or not torch.isfinite(p.grad).all():
                    raise RuntimeError('Missing/nonfinite adapter gradient: '+name)
                p.grad.div_(pending)
            norm = torch.nn.utils.clip_grad_norm_([p for n,p in params],a.max_grad_norm,error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            pending = 0
            if a.save_every_steps and steps % a.save_every_steps == 0:
                folder = out/f'step-{steps:06d}'
                core.save_adapter(folder,adapters,protocol)
                common.write(folder/'summary.json',dict(accepted=accepted,candidates=processed,steps=steps,
                    gradient_norm=float(norm),note='Weights-only snapshot; not evaluation or optimizer resume'))

        order = list(samples)
        random.Random(a.seed).shuffle(order)
        for sample in order:
            if accepted >= a.max_samples:
                break
            processed += 1
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats(0)
            try:
                si,ti,sa,ta,duration = training_inputs(processors,sample,a)
            except ReferenceUnavailable as error:
                reason = 'reference_budget_unavailable'
                reasons[reason] += 1
                record = dict(candidate=processed,sample=sample,accepted=False,gate_reason=reason,
                              error=str(error),accepted_total=accepted,optimizer_steps=steps,
                              seconds=time.monotonic()-started,memory=memory())
                common.append(out/'train.jsonl',record)
                print(json.dumps(record,ensure_ascii=False),flush=True)
                continue
            si = common.move(si,student)
            _,sg = common.generate(student,processors[0],si,sample,duration,a)
            _,tg = common.generate(teacher,processors[1],common.move(ti,teacher),sample,duration,a)
            active,reason = accept(sg,tg)
            reasons[reason] += 1
            record = dict(candidate=processed,sample=sample,accepted=active,gate_reason=reason,
                          student_gate=sg,teacher_gate=tg,student_video=sa,teacher_video=ta)
            if active:
                y,rollout = sample_rollout(student,processors[0],si,sample,duration,a)
                # Only detached teacher token distributions survive into student backward.
                with torch.no_grad():
                    tlogits = common.scoring_logits(teacher,ti,y)
                    log_q = tlogits.float().log_softmax(-1).cpu()
                    del tlogits
                del ti
                ti = None
                student.train()
                context = torch.autograd.graph.save_on_cpu(pin_memory=True) if a.offload_saved_tensors else contextlib.nullcontext()
                with context:
                    logits = common.scoring_logits(student,si,y)
                    log_p = logits.float().log_softmax(-1)
                    if log_p.shape != log_q.shape:
                        raise RuntimeError('Student/teacher scoring shapes differ')
                    log_q = log_q.to(log_p.device)
                    loss = (log_p.exp()*(log_p-log_q)).sum(-1).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite OPD reverse KL')
                loss.backward()
                accepted += 1
                pending += 1
                trained.append(sample)
                common.append(out/'accepted_samples.jsonl',dict(accepted_index=accepted,candidate=processed,
                              sample=sample,student_gate=sg,teacher_gate=tg))
                record.update(loss=float(loss.detach()),rollout=rollout)
                del logits,log_p,log_q,loss,y
                if pending == a.grad_accum:
                    update()
            record.update(accepted_total=accepted,optimizer_steps=steps,seconds=time.monotonic()-started,memory=memory())
            common.append(out/'train.jsonl',record)
            print(json.dumps(dict(candidate=processed,accepted=active,gate_reason=reason,
                accepted_total=accepted,accepted_target=a.max_samples,optimizer_steps=steps,
                student_iou=sg['iou'],teacher_iou=tg['iou'],loss=record.get('loss'),
                student_tokens=sa['actual_visual_tokens'],teacher_tokens=ta['actual_visual_tokens'],
                teacher_clip_tokens=ta.get('clip_visual_tokens',0),seconds=record['seconds'],memory=record['memory'])),flush=True)
            del si,ti
        update()
        final = digest(params)
        complete = accepted == a.max_samples and accepted > 0 and final != initial
        core.save_adapter(out/('final-adapter' if complete else 'partial-adapter'),adapters,protocol)
        common.write(out/'train_samples.json',trained)
        summary = dict(training_complete=complete,accepted_events=accepted,accepted_target=a.max_samples,
            candidate_events=processed,optimizer_steps=steps,gate_counts=dict(reasons),merged=False,
            target_reached=accepted==a.max_samples,candidate_pool_exhausted=accepted<a.max_samples,
            selected_heads=protocol['selected_heads'],initial_adapter_sha256=initial,final_adapter_sha256=final,
            note='Training only; held-out normal-input student evaluation is required')
        common.write(out/'summary.json',summary)
        if not complete:
            raise RuntimeError(f'Incomplete training target/update: accepted={accepted}/{a.max_samples}; partial adapter saved')
        if not a.skip_merge:
            del teacher
            torch.cuda.empty_cache()
            for name,adapter in adapters.items():
                adapter.merge()
                layer,target = name.split('.')
                setattr(student.model.language_model.layers[int(layer)].self_attn,target,adapter.base)
            student.gradient_checkpointing_disable()
            student.config.use_cache = True
            student.save_pretrained(out,safe_serialization=True,max_shard_size='4GB')
            processors[0].save_pretrained(out)
            summary['merged'] = True
            common.write(out/'summary.json',summary)
        return 0
    except Exception as error:
        common.write(out/'train_samples.json',trained)
        common.write(out/'failure.json',dict(training_complete=False,candidate_events=processed,
                      accepted_events=accepted,optimizer_steps=steps,error=repr(error)))
        raise
    finally:
        if decoder._decoder is not None:
            decoder._decoder.close()
            decoder._decoder = None


def main(argv=None):
    return run(parser().parse_args(argv))


