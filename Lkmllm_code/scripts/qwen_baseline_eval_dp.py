"""Auto-detected GPU Qwen evaluators, disjoint shards, full-precision merge."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import qwen_baseline_eval as evaluation


def combined_progress(expected):
    """Weighted running mean over committed samples, including failure zeros."""
    result = dict(evaluated=0, iou_sum=0., failures=0)
    for path, key, total in expected:
        try:
            state = json.loads(path.read_text(encoding='utf-8'))
            count, value, failures = state['evaluated'], state['iou_sum'], state['failures']
            if (state.get('protocol_key') != key or state.get('total') != total
                    or not isinstance(count, int) or not 0 <= count <= total
                    or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or not 0 <= value <= count + 1e-8
                    or not isinstance(failures, int) or not 0 <= failures <= count):
                continue
        except (OSError, ValueError, KeyError, TypeError):
            continue
        result['evaluated'] += count
        result['iou_sum'] += value
        result['failures'] += failures
    return result


def merge_workers(root, args, samples, workers=2):
    rows={}; runtimes=[]
    for rank in range(workers):
        directory=root/f'worker{rank}'
        expected_args=evaluation.parser().parse_args(worker_arguments(args,directory,rank,workers))
        shard=samples[rank::workers]
        expected_protocol=evaluation.protocol_for(expected_args,shard,samples)
        actual=json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
        if actual!=expected_protocol:
            raise ValueError(f'worker{rank} protocol mismatch')
        # Do not truncate a worker journal here. Both processes must have exited
        # successfully before merge; incomplete final lines are a merge error.
        path=directory/f'details_{args.split}.jsonl'
        raw=path.read_bytes()
        if raw and not raw.endswith(b'\n'):
            raise ValueError(f'worker{rank} has an uncommitted final record')
        local_rows=evaluation.read_progress(path,shard,True)
        if len(local_rows)!=len(shard):
            raise ValueError(f'worker{rank} incomplete: {len(local_rows)}/{len(shard)}')
        for local_index,row in local_rows.items():
            global_index=rank+workers*local_index
            if row.get('global_index')!=global_index or global_index in rows:
                raise ValueError('Wrong/duplicate global sample index')
            if row['sample_key']!=evaluation.sample_key(samples[global_index]):
                raise ValueError('Global sample identity mismatch')
            row=dict(row,worker_index=local_index,worker_rank=rank,index=global_index)
            rows[global_index]=row
        summary=json.loads((directory/f'metrics_{args.split}.json').read_text(encoding='utf-8'))
        if not summary['complete']:
            raise ValueError(f'worker{rank} reported incomplete metrics')
        runtimes.append(summary['runtime'])
    if set(rows)!=set(range(len(samples))):
        raise ValueError('Merged records do not cover exactly the selected samples')
    return dict(sorted(rows.items())),runtimes


def worker_arguments(args, directory, rank, workers=2):
    values=dict(vars(args),output_dir=str(directory),shard_index=rank,shard_count=workers)
    result=[]
    for action in evaluation.parser()._actions:
        if action.dest=='help': continue
        value=values[action.dest]
        if isinstance(value,bool):
            if value: result.append(action.option_strings[0])
        elif value is not None:
            result.extend([action.option_strings[0],str(value)])
    return result


def main(argv=None):
    args=evaluation.parser().parse_args(argv); evaluation.validate(args)
    if args.shard_count!=1 or args.shard_index!=0:
        raise ValueError('Do not provide worker shard flags to the launcher')
    import torch
    from scripts import e_head_eval as base
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    devices=[s.strip() for s in visible.split(',') if s.strip()] if visible is not None else [str(i) for i in range(torch.cuda.device_count())]
    if not devices or len(set(devices))!=len(devices) or torch.cuda.device_count()!=len(devices):
        raise ValueError('Visible CUDA device list does not match detected GPU count')
    samples=base.load_samples(Path(args.anno_json).resolve(),Path(args.video_dir).resolve(),args.max_samples)
    if not samples: raise ValueError('No selected samples')
    devices = devices[:min(len(devices), len(samples))]
    workers = len(devices)
    out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    protocol=evaluation.protocol_for(args,samples,samples)
    protocol.update(parallel_mode='data',workers=workers,sharding=f'global index modulo {workers}',
                    launcher_sha256=evaluation.digest(__file__))
    manifest=out/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=protocol:
        raise ValueError('Top-level protocol differs; choose a new output directory')
    evaluation.atomic_json(manifest,protocol)
    # A previous aggregate must never claim completion while a fresh run failed.
    evaluation.atomic_json(out/f'metrics_{args.split}.json',dict(
        total=len(samples),complete=False,status='workers_running',protocol=protocol))
    processes=[]; logs=[]; started=time.monotonic()
    expected_progress=[]
    bar = None
    try:
        for rank,device in enumerate(devices):
            directory=out/f'worker{rank}'; directory.mkdir(exist_ok=True)
            log=(directory/'console.log').open('a',encoding='utf-8'); logs.append(log)
            worker_args = evaluation.parser().parse_args(worker_arguments(args, directory, rank, workers))
            worker_protocol = evaluation.protocol_for(worker_args, samples[rank::workers], samples)
            expected_progress.append((directory/f'progress_{args.split}.json',
                                      evaluation.protocol_key(worker_protocol), len(samples[rank::workers])))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=device,QWEN_EVAL_PARENT_PROGRESS='1')
            # ROCm/HCU may require HIP visibility in addition to CUDA visibility.
            if getattr(getattr(torch, 'version', None), 'hip', None):
                env['HIP_VISIBLE_DEVICES'] = device
            command=[sys.executable,'-u',str(Path(evaluation.__file__).resolve()),*worker_arguments(args,directory,rank,workers)]
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            processes.append(process)
            print(f'worker{rank} pid={process.pid}, GPU={device}, samples={len(samples[rank::workers])}',flush=True)
        from tqdm import tqdm
        bar = tqdm(total=len(samples), desc=args.split, unit='sample',
                   dynamic_ncols=True, mininterval=1.)
        def refresh_progress():
            state = combined_progress(expected_progress)
            bar.set_postfix_str(evaluation.progress_label(state), refresh=False)
            bar.update(state['evaluated'] - bar.n)
        while any(process.poll() is None for process in processes):
            refresh_progress()
            time.sleep(.5)
        refresh_progress()
        bar.refresh()
        bar.close()
        bar = None
        codes=[process.returncode for process in processes]
        if any(codes):
            evaluation.atomic_json(out/f'metrics_{args.split}.json',dict(
                total=len(samples),complete=False,status='worker_failed',worker_exit_codes=codes,protocol=protocol))
            raise RuntimeError(f'Worker failure {codes}; inspect {out}/worker*/console.log; '
                               'journals retained; rerun same directory with --resume')
    finally:
        if bar is not None: bar.close()
        for process in processes:
            if process.poll() is None: process.terminate()
        for process in processes:
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        for log in logs: log.close()
    rows,runtimes=merge_workers(out,args,samples,workers)
    target=out/f'details_{args.split}.jsonl'; tmp=target.with_suffix('.jsonl.tmp')
    with tmp.open('w',encoding='utf-8') as stream:
        for row in rows.values(): stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(target)
    metrics=evaluation.aggregate(rows,len(samples))
    metrics.update(arguments=vars(args),protocol=protocol,parallel_mode='data',num_gpus=workers,
                   worker_runtime=runtimes,elapsed_sec_current_session=time.monotonic()-started)
    evaluation.atomic_json(out/f'metrics_{args.split}.json',metrics)
    evaluation.print_summary(metrics, args.split); return 0


if __name__=='__main__': raise SystemExit(main())
