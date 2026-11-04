"""Export a selected-head adapter snapshot as a standalone student model."""
import argparse
from pathlib import Path
import probe_qwen32b_privileged_k100 as video
import train_timelens_three_group_opsd as modeling


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    snap, out = Path(a.snapshot).resolve(), Path(a.output_dir).resolve()
    cfg = video.read(snap/'adapter_config.json')
    base = Path(cfg['arguments']['student_path']).resolve()
    if out.exists() or out == base or base in out.parents or out == snap.parent or snap.parent in out.parents:
        raise ValueError('Use a new export directory outside the base and original training run')
    if video.checkpoint_identity(base) != cfg['student_checkpoint']:
        raise ValueError('Base checkpoint config/index/size identity changed')
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one GPU')
    model = AutoModelForImageTextToText.from_pretrained(base, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={'':0}, attn_implementation='sdpa').eval()
    processor = AutoProcessor.from_pretrained(base, local_files_only=True, trust_remote_code=True)
    args = cfg['arguments']
    adapters = modeling.attach_head_lora(model, cfg['selected_heads'], args['rank'], args['alpha'], ['q_proj','o_proj'])
    saved = torch.load(snap/'adapter.pt', map_location='cpu', weights_only=True)
    if set(saved) != set(adapters):
        raise ValueError('Adapter module mismatch')
    with torch.no_grad():
        for name, adapter in adapters.items():
            row = saved[name]
            if set(row) != {'A','B','indices'} or not torch.equal(row['indices'].cpu(), adapter.indices.cpu()):
                raise ValueError('Adapter index mismatch')
            for key in ('A','B'):
                value, target = row[key], getattr(adapter,key)
                if value.shape != target.shape or not torch.isfinite(value).all():
                    raise ValueError('Invalid adapter tensor')
                target.copy_(value.to(target.device))
            adapter.merge()
            layer, target = name.split('.')
            setattr(model.model.language_model.layers[int(layer)].self_attn, target, adapter.base)
    out.mkdir(parents=True, exist_ok=False)
    model.config.use_cache = True
    model.save_pretrained(out, safe_serialization=True, max_shard_size='4GB')
    processor.save_pretrained(out)
    video.write(out/'merge_provenance.json', dict(snapshot=str(snap), student_checkpoint=cfg['student_checkpoint'],
        adapter_sha256=video.sha(snap/'adapter.pt'), adapter_config_sha256=video.sha(snap/'adapter_config.json')))


if __name__ == '__main__':
    main()
