"""Export one round's student-adapter on one GPU, without modifying its run."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    snap, out = Path(a.snapshot).resolve(), Path(a.output_dir).resolve()
    cfg = json.loads((snap/'adapter_config.json').read_text())
    summary = json.loads((snap/'summary.json').read_text())
    base = Path(cfg['arguments']['student_path']).resolve()
    if (out.exists() or out == base or base in out.parents or snap.parent in out.parents
            or out == snap.parent):
        raise ValueError('Use a new export directory outside the base and snapshot round')
    if not summary.get('target_reached'):
        raise ValueError('Require a completed round student snapshot')
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from train_timelens_three_group_opsd import attach_head_lora
    model = AutoModelForImageTextToText.from_pretrained(base, local_files_only=True,
        trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa').eval()
    processor = AutoProcessor.from_pretrained(base, local_files_only=True, trust_remote_code=True)
    train = cfg['arguments']
    adapters = attach_head_lora(model, cfg['selected_heads'], train['rank'], train['alpha'], ['q_proj','o_proj'])
    saved = torch.load(snap/'adapter.pt', map_location='cpu', weights_only=True)
    if set(saved) != set(adapters):
        raise ValueError('Adapter modules mismatch')
    with torch.no_grad():
        for name, adapter in adapters.items():
            row = saved[name]
            if set(row) != {'A','B','indices'} or not torch.equal(row['indices'].cpu(), adapter.indices.cpu()):
                raise ValueError('Adapter indices mismatch')
            for key in ('A','B'):
                source, target = row[key], getattr(adapter, key)
                if source.shape != target.shape or not torch.isfinite(source).all():
                    raise ValueError('Invalid adapter values')
                target.copy_(source.to(target.device))
            adapter.merge()
            layer, target = name.split('.')
            setattr(model.model.language_model.layers[int(layer)].self_attn, target, adapter.base)
    out.mkdir(parents=True, exist_ok=False)
    model.config.use_cache = True
    model.save_pretrained(out, safe_serialization=True, max_shard_size='4GB')
    processor.save_pretrained(out)
    (out/'round_merge_provenance.json').write_text(json.dumps(dict(snapshot=str(snap),
        base_model=str(base), round_summary=summary), ensure_ascii=False, indent=2), encoding='utf-8')
    print('Exported:', out, flush=True)


if __name__ == '__main__':
    main()
