"""Single-GPU Qwen2.5/Qwen3 native evaluation with durable, resumable records.

Reuses e_head_eval's annotation expansion, file resolver, short task prompt,
and time parser. Every selected sample remains in the metric denominator.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / 'scripts'):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

THRESHOLDS = (.3, .5, .7)
STATUSES = ('ok', 'zero_iou', 'parse_fail', 'no_video', 'decode_fail', 'oom', 'error')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-family', required=True, choices=['qwen25vl', 'qwen3vl'])
    for name in ('model-path', 'anno-json', 'video-dir', 'output-dir', 'split'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--fps', type=float, default=2.)
    p.add_argument('--min-tokens', type=int, default=4)
    p.add_argument('--total-tokens', type=int, default=4096)
    p.add_argument('--max-frames', type=int, default=1024)
    p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--repetition-penalty', type=float, default=1.05)
    p.add_argument('--max-samples', type=int, default=0)
    p.add_argument('--attn-implementation', choices=['sdpa', 'flash_attention_2'], default='sdpa')
    p.add_argument('--decode-timeout', type=float, default=120.)
    p.add_argument('--decode-max-attempts', type=int, default=3)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--shard-index', type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument('--shard-count', type=int, default=1, help=argparse.SUPPRESS)
    return p


def validate(args):
    if (not math.isfinite(args.fps) or args.fps <= 0 or min(args.min_tokens, args.total_tokens,
            args.max_new_tokens, args.decode_max_attempts) < 1 or args.max_samples < 0
            or args.max_frames < 0 or args.max_frames % 2
            or not math.isfinite(args.repetition_penalty) or args.repetition_penalty <= 0
            or not math.isfinite(args.decode_timeout) or args.decode_timeout <= 0):
        raise ValueError('Invalid video/generation/decode parameters')
    if any(c in args.split for c in '/\\') or args.split in ('.', '..'):
        raise ValueError('split must be a filename label')
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError('Invalid sample shard')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_key(sample):
    return hashlib.sha256(json.dumps([sample['video_id'], sample['query'],
        sample['gt_start'], sample['gt_end']], ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def read_progress(path, samples, resume):
    if not path.exists():
        return {}
    if not resume:
        raise FileExistsError('Records already exist; use --resume or a new output directory')
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        # Only discard an uncommitted final fragment. Never ignore corrupt complete lines.
        committed = raw.rfind(b'\n') + 1
        print('[resume] discard uncommitted final journal fragment', flush=True)
        with path.open('r+b') as stream:
            stream.truncate(committed)
        raw = raw[:committed]
    rows = {}
    for line in raw.decode('utf-8').splitlines():
        row = json.loads(line)
        i = row['index']
        if not isinstance(i, int) or i < 0 or i >= len(samples) or i in rows:
            raise ValueError('Duplicate/out-of-range record index')
        if row['sample_key'] != sample_key(samples[i]):
            raise ValueError('Record/sample identity mismatch')
        if row['status'] not in STATUSES or not math.isfinite(row['iou']) or not 0 <= row['iou'] <= 1:
            raise ValueError('Invalid persisted status/IoU')
        if row['status'] not in ('ok', 'zero_iou') and row['iou'] != 0:
            raise ValueError('Failed sample must have zero IoU')
        rows[i] = row
    return rows


def aggregate(rows, expected):
    counts = Counter(row['status'] for row in rows.values())
    values = [row['iou'] for row in rows.values()]
    denominator = expected
    hits = {str(t): sum(v >= t for v in values) for t in THRESHOLDS}
    result = dict(total=expected, evaluated=len(rows), complete=len(rows) == expected,
        status_counts=dict(counts), parse_fail=counts['parse_fail'], no_video=counts['no_video'],
        decode_fail=counts['decode_fail'], num_oom=counts['oom'], num_errors=counts['error'],
        metric_denominator='all selected samples; failed inference counts as zero',
        recall_hits=hits, mIoU=math.fsum(values) / denominator if denominator else None,
        partial_metrics_note=None if len(rows) == expected else 'incomplete; unattempted samples currently count as zero; not a final score')
    for t in THRESHOLDS:
        result[f'R@1_IoU{t}'] = hits[str(t)] / denominator if denominator else None
    return result


def protocol_key(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def progress_state(rows, total, key):
    return dict(protocol_key=key, total=total, evaluated=len(rows),
                iou_sum=math.fsum(row['iou'] for row in rows.values()),
                failures=sum(row['status'] not in ('ok', 'zero_iou') for row in rows.values()))


def update_progress_state(state, row):
    state['evaluated'] += 1
    state['iou_sum'] += row['iou']
    state['failures'] += row['status'] not in ('ok', 'zero_iou')


def progress_label(state):
    count = state['evaluated']
    mean = f'{100*state["iou_sum"]/count:.2f}%' if count else '--'
    return f'mIoU(done)={mean}, failed={state["failures"]}'


def print_summary(result, split):
    def percent(value):
        return f'{100*value:.2f}%' if value is not None else '--'
    values = ' '.join(f'R@{t:g}={percent(result[f"R@1_IoU{t}"])}' for t in THRESHOLDS)
    print(f'[{split}] {result["evaluated"]}/{result["total"]} '
          f'mIoU={percent(result["mIoU"])} {values}', flush=True)


def protocol_for(args, samples, all_samples=None):
    protocol = {k:v for k,v in vars(args).items() if k not in ('resume', 'output_dir')}
    for key in ('model_path', 'anno_json', 'video_dir'):
        protocol[key] = str(Path(protocol[key]).resolve())
    protocol.update(version=1, selected_samples=len(samples), annotation_sha256=digest(args.anno_json),
        sample_keys_sha256=hashlib.sha256('\n'.join(map(sample_key,samples)).encode()).hexdigest(),
        prompt='official short user-only TimeLens task prompt; no TaRO answer tags',
        parse_protocol='e_head_eval.parse_pred; finite ordered nonnegative span required',
        video_pipeline='native Qwen family; isolated qwen-vl-utils; no processor resize/resampling',
        pixel_budget_factor=28*28 if args.model_family=='qwen25vl' else 32*32,
        source_sha256={name:digest(ROOT/'scripts'/name) for name in
            ('qwen_baseline_eval.py','e_head_eval.py','qwen25_head_inputs.py','qwen_decode_worker.py')},
        model_metadata_sha256={name:digest(Path(args.model_path)/name) for name in
            ('config.json','generation_config.json','model.safetensors.index.json','processor_config.json',
             'preprocessor_config.json','video_preprocessor_config.json','chat_template.json','chat_template.jinja')
            if (Path(args.model_path)/name).is_file()})
    all_samples=samples if all_samples is None else all_samples
    protocol.update(all_selected_samples=len(all_samples),
        all_sample_keys_sha256=hashlib.sha256('\n'.join(map(sample_key,all_samples)).encode()).hexdigest())
    return protocol


def main(argv=None):
    args = parser().parse_args(argv); validate(args)
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    from scripts import e_head_eval as base
    import qwen25_head_inputs as native25
    from scripts.qwen_decode_worker import configure_decoder, process_vision_info, VideoDecodeError
    base.process_vision_info=process_vision_info
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Set CUDA_VISIBLE_DEVICES to exactly one GPU')
    model_dir=Path(args.model_path).resolve()
    config=AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    expected='qwen2_5_vl' if args.model_family=='qwen25vl' else 'qwen3_vl'
    if config.model_type != expected:
        raise ValueError(f'Model family mismatch: expected {expected}, got {config.model_type}')
    base._require_full_model(model_dir)
    video_dir=Path(args.video_dir).resolve()
    all_samples=base.load_samples(Path(args.anno_json).resolve(),video_dir,args.max_samples)
    if not all_samples:
        raise ValueError('No selected samples')
    global_indices=list(range(args.shard_index,len(all_samples),args.shard_count))
    samples=[all_samples[i] for i in global_indices]
    out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    protocol=protocol_for(args,samples,all_samples)
    manifest=out/'protocol.json'; journal=out/f'details_{args.split}.jsonl'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Protocol/model/annotation/code changed; use a new output directory')
    else:
        if journal.exists():
            raise ValueError('Existing journal has no protocol manifest')
        atomic_json(manifest,protocol)
    rows=read_progress(journal,samples,args.resume)
    progress = progress_state(rows, len(samples), protocol_key(protocol))
    progress_file = out/f'progress_{args.split}.json'
    atomic_json(progress_file, progress)
    runtime=dict(torch_version=torch.__version__,transformers_version=transformers.__version__,
                 device_name=torch.cuda.get_device_name(0), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    started=time.monotonic()
    def save_metrics():
        result=aggregate(rows,len(samples))
        result.update(arguments=vars(args),protocol=protocol,runtime=runtime,
                      elapsed_sec_current_session=time.monotonic()-started)
        atomic_json(out/f'metrics_{args.split}.json',result)
        return result
    if len(rows)==len(samples):
        print_summary(save_metrics(), args.split); return 0
    configure_decoder(timeout=args.decode_timeout,attempts=args.decode_max_attempts)
    processor=AutoProcessor.from_pretrained(str(model_dir),local_files_only=True,
                                            padding_side='left',do_resize=False)
    if args.model_family=='qwen25vl': native25.validate_processor(processor)
    if hasattr(processor,'video_processor'):
        processor.video_processor.do_sample_frames=False
    model=AutoModelForImageTextToText.from_pretrained(str(model_dir),local_files_only=True,
        dtype=torch.bfloat16,device_map={'':0},attn_implementation=args.attn_implementation).eval()
    if any(p.device.type!='cuda' for p in model.parameters()):
        raise RuntimeError('All weights must reside on the single GPU')
    model.generation_config.num_beams=1; model.generation_config.num_return_sequences=1
    device=model.get_input_embeddings().weight.device
    print(f'{args.model_family}: {args.split}, total={len(samples)}, resumed={len(rows)}, devices={sorted({str(p.device) for p in model.parameters()})}',flush=True)
    from tqdm import tqdm
    managed = os.environ.get('QWEN_EVAL_PARENT_PROGRESS') == '1'
    bar = tqdm(total=len(samples), initial=len(rows), desc=args.split, unit='sample',
               dynamic_ncols=True, mininterval=1., disable=managed)
    bar.set_postfix_str(progress_label(progress), refresh=False)
    try:
        with journal.open('a',encoding='utf-8') as stream:
            for i,sample in enumerate(samples):
                if i in rows: continue
                before=time.monotonic()
                record=dict(index=i,global_index=global_indices[i],sample_key=sample_key(sample),**sample,status='error',iou=0.,
                            prediction=None,model_output='',error=None)
                inputs=outputs=decoded=None
                try:
                    video=base.resolve_video_path(video_dir,sample)
                    if video is None or not video.is_file():
                        record['status']='no_video'
                    else:
                        if args.model_family=='qwen25vl':
                            messages=native25.video_message(dict(video_path=video,query=sample['query']),
                                args.fps,args.min_tokens,args.total_tokens,args.max_frames)
                            # Same official short user-only task prompt as the Qwen3 evaluator.
                            messages=messages[-1:]
                            decoded=native25.decode_video(messages)
                            inputs=native25.process_messages(processor,messages,decoded)
                            record['video_processing']=native25.audit_inputs(processor,inputs,decoded)
                            inputs={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
                            decoded=None
                        else:
                            inputs=base.build_inputs_timelens(processor,video,sample['query'],device,
                                args.fps,args.min_tokens,args.total_tokens,'official',args.max_frames)
                            record['video_processing']=dict(patch_size=16,
                                video_grid_thw=inputs['video_grid_thw'].tolist(),
                                do_resize=False,do_sample_frames=False)
                        if hasattr(model.model,'rope_deltas'): model.model.rope_deltas=None
                        with torch.inference_mode():
                            outputs=model.generate(**inputs,max_new_tokens=args.max_new_tokens,
                                repetition_penalty=args.repetition_penalty,do_sample=False,use_cache=True,
                                temperature=None,top_p=None,top_k=None,
                                pad_token_id=processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id)
                        suffix=outputs[0,inputs['input_ids'].shape[1]:]
                        text=processor.tokenizer.decode(suffix,skip_special_tokens=True)
                        record.update(model_output=text,generated_tokens=int(suffix.numel()),
                                      reached_generation_limit=int(suffix.numel())>=args.max_new_tokens)
                        duration=max(float(sample.get('duration',0.) or 0.),float(sample['gt_end']))
                        ps,pe=base.parse_pred(text,duration)
                        if not (math.isfinite(ps) and math.isfinite(pe) and 0<=ps<pe):
                            record['status']='parse_fail'
                        else:
                            value=float(base.temporal_iou(ps,pe,float(sample['gt_start']),float(sample['gt_end'])))
                            if not math.isfinite(value) or not 0<=value<=1: raise ValueError('Invalid computed IoU')
                            record.update(prediction=[ps,pe],iou=value,status='ok' if value>0 else 'zero_iou')
                except torch.OutOfMemoryError as exc:
                    record.update(status='oom',error=f'{type(exc).__name__}: {exc}')
                except VideoDecodeError as exc:
                    record.update(status='decode_fail',error=f'{type(exc).__name__}: {exc}')
                except Exception as exc:
                    record.update(status='error',error=f'{type(exc).__name__}: {exc}')
                finally:
                    inputs=outputs=decoded=None
                record['elapsed_sec']=time.monotonic()-before
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
                rows[i]=record
                update_progress_state(progress, record)
                atomic_json(progress_file, progress)
                bar.set_postfix_str(progress_label(progress), refresh=False)
                bar.update(1)
                if record['status'] in ('oom','error','decode_fail'):
                    gc.collect(); torch.cuda.empty_cache()
                if not managed and record['error']:
                    tqdm.write(f'[{args.split}] sample={i} {record["status"]}: {record["error"]}')
                if len(rows)%20==0: save_metrics()
    finally:
        bar.close()
        result=save_metrics()
    print_summary(result, args.split); return 0


if __name__=='__main__':
    raise SystemExit(main())
