"""Single-GPU normal-input evaluation; GT is used only for final metrics."""
import argparse
from collections import Counter
import gc
from pathlib import Path

import qwen_decode_worker as decoder
import probe_qwen32b_privileged_k100 as video
from e_head_eval import load_samples, resolve_video_path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model-path', 'anno-json', 'video-dir', 'output-dir'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--model-family', choices=('qwen3vl',), default='qwen3vl')
    p.add_argument('--split', default='Qvhighlights')
    p.add_argument('--total-tokens', type=int, default=14336)
    p.add_argument('--min-tokens', type=int, default=4)
    p.add_argument('--max-frames', type=int, default=1024)
    p.add_argument('--fps', type=float, default=2.)
    p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--max-samples', type=int, default=0)
    p.add_argument('--prompt-mode', choices=('official', 'decimal_seconds'), default='decimal_seconds',
                   help='decimal_seconds adds the same seconds-only instruction as Top10 training')
    p.add_argument('--decode-timeout', type=float, default=600.)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    return p


def main():
    a = parser().parse_args()
    from qwen_token_local_gate import SECONDS_INSTRUCTION
    prompt_suffix = SECONDS_INSTRUCTION if a.prompt_mode == 'decimal_seconds' else ''
    if '/' in a.split or '\\' in a.split or a.split in ('.', '..'):
        raise ValueError('split must be a filename label')
    if min(a.total_tokens, a.min_tokens, a.max_new_tokens) < 1 or a.max_samples < 0:
        raise ValueError('Invalid counts')
    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new empty evaluation directory')
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if torch.cuda.device_count() != 1:
        raise ValueError('Set CUDA_VISIBLE_DEVICES to exactly one GPU')
    samples = load_samples(Path(a.anno_json), Path(a.video_dir), a.max_samples)
    if not samples:
        raise ValueError('No evaluation samples')
    total_selected = len(samples)
    if not 0 <= a.shard_index < a.shard_count:
        raise ValueError('Invalid shard')
    indexed = list(enumerate(samples))[a.shard_index::a.shard_count]
    if not indexed:
        raise ValueError('Empty shard; use fewer workers')
    model = AutoModelForImageTextToText.from_pretrained(a.model_path, local_files_only=True,
        trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={'':0}, attn_implementation='sdpa').eval()
    if model.config.model_type != 'qwen3_vl':
        raise ValueError('Expected Qwen3-VL')
    processor = AutoProcessor.from_pretrained(a.model_path, local_files_only=True,
        trust_remote_code=True, padding_side='left', do_resize=False)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    out.mkdir(parents=True, exist_ok=True)
    video.write(out/'protocol.json', dict(arguments=vars(a), prompt='Normal full-source video/query only', prompt_suffix=prompt_suffix,
        model=video.checkpoint_identity(a.model_path), annotation_sha256=video.sha(a.anno_json),
        failures='All selected rows remain in denominator; failures count zero', runtime=video.runtime_identity(),
        source_sha256={p.name:video.sha(p) for p in [Path(__file__), Path(video.__file__)]}))
    decoder.configure_decoder(timeout=a.decode_timeout, attempts=3)
    rows, statuses = [], Counter()
    try:
        for i, (global_index, sample) in enumerate(indexed):
            row = dict(index=i, global_index=global_index, sample=sample, valid=False, prediction=None, iou=0., status='error')
            inputs = None
            try:
                path = resolve_video_path(Path(a.video_dir), sample)
                if path is None:
                    row['status'] = 'no_video'
                else:
                    # Only video/query reach the model; GT remains outside input construction.
                    inputs, audit = video.normal_input(processor, dict(video_path=str(path), query=sample['query']),
                        a.total_tokens, a.min_tokens, a.fps, a.max_frames, prompt_suffix=prompt_suffix)
                    _, result = video.generate(model, processor, inputs, sample, audit['duration'], a)
                    row.update(result, video=audit, status='ok' if result['valid'] else 'parse_fail')
            except torch.OutOfMemoryError as error:
                row.update(status='oom', error=str(error))
            except decoder.VideoDecodeError as error:
                row.update(status='decode_fail', error=str(error))
            except Exception as error:
                row.update(status='error', error=repr(error))
            finally:
                del inputs
                gc.collect()
                torch.cuda.empty_cache()
            rows.append(row)
            statuses[row['status']] += 1
            video.append(out/f'details_{a.split}.jsonl', row)
            if (i+1) % 20 == 0 or i+1 == len(indexed):
                print(f'{a.split}: {i+1}/{len(indexed)} mIoU={sum(r["iou"] for r in rows)/len(rows):.6f}', flush=True)
        summary = dict(video.metrics(rows), complete=True, total=len(indexed), total_selected=total_selected, evaluated=len(rows), statuses=dict(statuses))
        video.write(out/f'metrics_{a.split}.json', summary)
        print(summary, flush=True)
    finally:
        if decoder._decoder is not None:
            decoder._decoder.close()


if __name__ == '__main__':
    main()

