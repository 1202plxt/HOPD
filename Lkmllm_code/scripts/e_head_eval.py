"""QVHighlights/Charades/ActivityNet annotation expansion and video resolution."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional
_VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.webm')
_VIDEO_INDEX_CACHE = {}

def _video_index(video_dir: Path) -> dict[str, Path]:
    """Create one recursive basename index for datasets with nested video dirs."""
    key = str(video_dir.resolve())
    if key not in _VIDEO_INDEX_CACHE:
        index: dict[str, Path] = {}
        for path in sorted(video_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in _VIDEO_EXTENSIONS:
                index.setdefault(path.name.casefold(), path)
        _VIDEO_INDEX_CACHE[key] = index
        print(f"  视频索引: {len(index)} 个文件（递归扫描 {video_dir}）")
    return _VIDEO_INDEX_CACHE[key]


def resolve_video_path(video_dir: Path, s: dict) -> Optional[Path]:
    video_id = str(s["video_id"])
    rel = str(s.get("video_rel_path", "") or "")

    # Exact paths first. QVHighlights is often stored one directory below the
    # path passed through --video-dir.
    candidates = []
    if rel:
        candidates.append(video_dir / rel)
    for ext in _VIDEO_EXTENSIONS:
        filename = f"{video_id}{ext}"
        candidates.extend([
            video_dir / filename,
            video_dir / "videos" / filename,
            video_dir / "video" / filename,
            video_dir / "qvhighlights" / filename,
        ])
    for path in candidates:
        if path.is_file():
            return path

    # Do not strip a suffix such as _210.0_360.0: that suffix identifies the
    # temporal clip and loading the untrimmed source video would invalidate GT.
    index = _video_index(video_dir)
    if rel:
        found = index.get(Path(rel).name.casefold())
        if found is not None:
            return found
    for ext in _VIDEO_EXTENSIONS:
        found = index.get(f"{video_id}{ext}".casefold())
        if found is not None:
            return found
    return None


def load_samples(anno_json: Path, video_dir: Path,
                 max_samples: int = 500) -> list[dict]:
    """
    加载样本，自动适配多种 JSON 格式：
      A. timelens 统一格式 / Charades 格式: {"video_id": {"duration":..., "queries":[...]}}
      B. ActivityNet 原始格式: {video_id: {timestamps: [...], sentences: [...]}}
      C. 嵌套格式: {video_id: [{...}, {...}]}

    当样本总数超过 max_samples 时，按照 annotation 展开后的稳定顺序，
    在 [0, n-1] 范围内等间隔选取 max_samples 个索引。该过程不打乱、
    不使用随机数；base 与 LoRA 使用同一 annotation 时子集严格一致。
    """
    import numpy as np
    raw = json.loads(anno_json.read_text(encoding="utf-8"))
    raw.pop("_meta", None)

    if "数据" in raw:
        raw = raw["数据"]

    samples = []
    for vid, info in raw.items():
        if not isinstance(info, dict):
            continue

        # 格式 B: ActivityNet {timestamps, sentences}
        if "timestamps" in info and "sentences" in info:
            anns = info["timestamps"]
            sents = info["sentences"]
            duration = info.get("duration", 0.0)
            for i, (span, sent) in enumerate(zip(anns, sents)):
                if len(span) < 2:
                    continue
                samples.append({
                    "video_id": vid,
                    "query": sent,
                    "duration": duration if duration > 0 else span[-1] + 10.0,
                    "gt_start": float(span[0]),
                    "gt_end": float(span[1]),
                })
            continue

        # 格式 A / C
        duration = info.get("duration", 0.0)
        queries = info.get("queries", [])

        if queries:
            for query_idx, q in enumerate(queries):
                if isinstance(q, dict):
                    query_text = q.get("q_pos", q.get("query", ""))
                    if "spans" in q and q["spans"]:
                        gt_start = float(q["spans"][0][0])
                        gt_end = float(q["spans"][0][1])
                    elif "span" in q:
                        span = q["span"]
                        if isinstance(span, (list, tuple)) and len(span) >= 2:
                            gt_start = float(span[0])
                            gt_end = float(span[1])
                        else:
                            continue
                    else:
                        continue
                elif isinstance(q, str):
                    query_text = q
                    spans = info.get("spans", [])
                    if query_idx < len(spans) and len(spans[query_idx]) >= 2:
                        gt_start = float(spans[query_idx][0])
                        gt_end = float(spans[query_idx][1])
                    else:
                        continue
                else:
                    continue

                if not query_text:
                    continue

                samples.append({
                    "video_id": vid,
                    "query": query_text,
                    "duration": duration if duration > 0 else 120.0,
                    "gt_start": gt_start,
                    "gt_end": gt_end,
                })
        else:
            items = info if isinstance(info, list) else info.get("items", [])
            for item in items:
                samples.append({
                    "video_id": vid,
                    "query": item.get("sentence", item.get("query", "")),
                    "duration": item.get("duration", duration or 120.0),
                    "gt_start": float(item.get("gt_start", item.get("start", 0))),
                    "gt_end": float(item.get("gt_end", item.get("end", 0))),
                })

    # 官方 loader 会压缩空白并去掉 query 末尾句点。
    for sample in samples:
        sample["query"] = re.sub(r"\s+", " ", sample["query"]).strip().strip(".").strip()

    if max_samples and len(samples) > max_samples:
        total_before_sampling = len(samples)
        uniform_indices = np.linspace(
            0, total_before_sampling - 1,
            num=max_samples,
            dtype=np.int64,
        )
        samples = [samples[int(i)] for i in uniform_indices]
        print(f"  均匀采样: {total_before_sampling} → {len(samples)} 条 "
              f"(index {int(uniform_indices[0])}..{int(uniform_indices[-1])})")

    print(f"  加载 {len(samples)} 条样本")
    if samples:
        print(f"  示例: [{samples[0]['video_id']}] "
              f"[{samples[0]['gt_start']:.1f}, {samples[0]['gt_end']:.1f}] "
              f"'{samples[0]['query'][:50]}'")

    return samples
