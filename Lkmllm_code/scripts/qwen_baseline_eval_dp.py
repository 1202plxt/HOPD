"""Use the original evaluator filename for disjoint single-GPU workers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter
import qwen_baseline_eval as evaluation
import probe_qwen32b_privileged_k100 as video
from e_head_eval import load_samples


def worker_visibility_environment(parent, device, hip):
    env = dict(parent)
    if hip:
        for name in ('CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'HSA_VISIBLE_DEVICES'):
            env.pop(name, None)
        env['HIP_VISIBLE_DEVICES'] = device
    else:
        env['CUDA_VISIBLE_DEVICES'] = device
    return env


def main():
    a = evaluation.parser().parse_args()
    if a.shard_index != 0 or a.shard_count != 1:
        raise ValueError('Do not set shard flags on the launcher')
    out = Path(a.output_dir)
    if not a.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new empty output directory')
    import torch
    hip = bool(getattr(torch.version, 'hip', None))
    visible = os.environ.get('HIP_VISIBLE_DEVICES' if hip else 'CUDA_VISIBLE_DEVICES')
    ids = visible.split(',') if visible else list(map(str,range(torch.cuda.device_count())))
    if not ids or len(set(ids)) != len(ids) or torch.cuda.device_count() != len(ids):
        raise ValueError('Invalid visible GPU configuration')
    selected = load_samples(Path(a.anno_json),Path(a.video_dir),a.max_samples)
    ids = ids[:len(selected)]
    if not ids:
        raise ValueError('No selected samples')
    out.mkdir(parents=True,exist_ok=True)
    processes, streams = [], []
    try:
        for rank, device in enumerate(ids):
            values = dict(vars(a),output_dir=str(out/f'worker{rank}'),shard_index=rank,shard_count=len(ids))
            args=[]
            for key,value in values.items():
                if isinstance(value,bool):
                    if value:args.append('--'+key.replace('_','-'))
                else:args.extend(['--'+key.replace('_','-'),str(value)])
            stream = (out/f'worker{rank}.log').open('w',encoding='utf-8')
            streams.append(stream)
            processes.append(subprocess.Popen([sys.executable,'-u',str(Path(evaluation.__file__)),*args],
                env=worker_visibility_environment(os.environ, device.strip(), hip),stdout=stream,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):
                raise RuntimeError('Evaluation worker failed; inspect worker logs')
            time.sleep(1)
        if any(p.returncode for p in processes):
            raise RuntimeError('Evaluation worker failed; inspect worker logs')
        rows = {}
        for rank in range(len(ids)):
            folder = out/f'worker{rank}'
            metrics = video.read(folder/f'metrics_{a.split}.json')
            if not metrics['complete'] or metrics['total_selected'] != len(selected):
                raise ValueError('Incomplete evaluation shard')
            for line in (folder/f'details_{a.split}.jsonl').read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                index = row['global_index']
                if index in rows or not 0 <= index < len(selected) or index % len(ids) != rank or row['sample'] != selected[index]:
                    raise ValueError('Wrong or duplicate sample identity')
                rows[index] = row
        if set(rows) != set(range(len(selected))):
            raise ValueError('Incomplete global sample coverage')
        ordered = [rows[i] for i in range(len(selected))]
        target=out/f'details_{a.split}.jsonl';tmp=target.with_suffix('.tmp')
        with tmp.open('w',encoding='utf-8') as f:
            for row in ordered:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        tmp.replace(target)
        summary = dict(video.metrics(ordered),complete=True,total=len(selected),evaluated=len(ordered),
            statuses=dict(Counter(r['status'] for r in ordered)))
        video.write(out/f'metrics_{a.split}.json',summary)
        video.write(out/'protocol.json',dict(arguments=vars(a),workers=len(ids),worker_protocols=[
            video.read(out/f'worker{r}'/'protocol.json') for r in range(len(ids))]))
        print(summary,flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    main()

