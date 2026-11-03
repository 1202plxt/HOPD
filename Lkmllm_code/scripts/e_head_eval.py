"""
e_head_eval.py — HeadLoRA 微调模型的 Temporal Grounding 评估
                 (TimeLens prompt 对齐版，纯 generate 路径，多卡模型分片)

⚠️ 采样/编码参数必须和 heads_finetune.py 训练时完全一致，否则模型看到的
   "视频"和训练时不是一回事（帧数、每帧分辨率、时间戳节奏、attention 数值
   实现都会变），IoU 会直接掉到接近随机猜的水平。默认值全部对齐训练：

     历史默认: --fps 2.0 --min-tokens 64 --total-tokens 14336 --max-frames 0
     新预算实验: --fps 2 --min-tokens 4 --total-tokens 4096 --max-frames 1024
              --attn-implementation flash_attention_2
     评测默认: 同上

   如果训练时用的是不同参数，评测时也要对应传同样的值。用 --model-path
   同时评测 base model (baseline)、top-K HeadLoRA、random-head 对照，
   三者的采样参数必须相同才有可比性。

相对上一版评测脚本改了这几处 (为解决 IoU 从训练时期望的 ~0.5 掉到 0.09 的问题):
  1. attn_implementation 默认改成 flash_attention_2，和训练一致；
     可以用 --attn-implementation eager 切回旧行为 (但没必要)。
  2. 去掉 processor.video_processor.size['shortest_edge'] = 128 这行硬编码，
     训练时没有做这一步，评测强行压到 128 会让视频细节大量丢失。
  3. build_inputs_timelens 里的 video_content 现在带上 min_pixels / total_pixels，
     和训练时 collate_fn_with_processor 走的是同一条 pixel-budget 路径；
     sample_fps 也不再硬编码 1.0，改用外部传入的 --fps 值。
  4. CLI 默认 --min-tokens 64 / --total-tokens 14336；可显式传 --max-frames。
  5. EvalDataset 里当 total_tokens > 0 时不再做 max-size resize (和训练一致，
     让 processor 内部按 pixel budget 处理，避免我们在外面提前缩放搞乱)。

用法:
    # 微调后 (top-K HeadLoRA 主实验)
    python scripts/e_head_eval.py \
        --model-path  ../Lkmllm_data/checkpoints/lora_layer \
        --anno-json   ../Lkmllm_data/datasets/Test/Charades_sta/charades-timelens.json \
        --video-dir   ../Lkmllm_data/datasets/Test/Charades_sta/charades \
        --output-dir  ../Lkmllm_data/outputs/eval_results \
        --split       charades_topk

    # base baseline (⚠️ 采样参数必须传一样的)
    python scripts/e_head_eval.py \
        --model-path  ../shared_models/Qwen3-VL-8B-Instruct \
        --anno-json   ../Lkmllm_data/datasets/Test/Charades_sta/charades-timelens.json \
        --video-dir   ../Lkmllm_data/datasets/Test/Charades_sta/charades \
        --output-dir  ../Lkmllm_data/outputs/eval_results \
        --split       charades_base

    # 官方 TimeLens-8B（开启专用 processor / Qwen3-VL / generation 路径）
    python scripts/e_head_eval.py \
        --model-path  ../shared_models/TimeLens-8B \
        --anno-json   ../Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights-timelens.json \
        --video-dir   ../Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights \
        --output-dir  ../Lkmllm_data/outputs/eval_results \
        --split       Qvhighlights_TimeLens8B \
        --timelens-model

    # 随机 head 对照
    python scripts/e_head_eval.py \
        --model-path  ../Lkmllm_data/checkpoints/lora_head_random \
        --anno-json   ../Lkmllm_data/datasets/Test/Charades_sta/charades-timelens.json \
        --video-dir   ../Lkmllm_data/datasets/Test/Charades_sta/charades \
        --output-dir  ../Lkmllm_data/outputs/eval_results \
        --split       charades_random

双卡并行评测：加 --parallel-mode data --num-gpus 2，每张卡加载一份完整模型并处理一半样本。
显存不足以单卡容纳模型时使用 --parallel-mode model，通过 device_map 跨卡分片。
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]   # 项目根目录 = Lkmllm_code
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _install_project_path_api(module, project_root):
    """Accept older project_paths exports in both evaluators and DP workers.

    Only this process's module namespace is supplemented; the shared paths
    file is not rewritten. Explicit CLI paths and video/metric logic stay intact.
    """
    project_root = Path(project_root).resolve()
    workspace_root = project_root.parent
    if not hasattr(module, "A_DATA_ROOT"):
        module.A_DATA_ROOT = workspace_root / "Lkmllm_data"
    if not hasattr(module, "SHARED_MODELS_ROOT"):
        module.SHARED_MODELS_ROOT = workspace_root / "shared_models"

    def ensure_directory(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolve_path(value, *, default=None, must_exist=False):
        if value is None:
            result = Path(default) if default is not None else project_root
        else:
            raw = Path(str(value)).expanduser()
            result = raw if raw.is_absolute() else (Path.cwd() / raw).resolve()
            if not raw.is_absolute() and must_exist and not result.exists():
                result = (workspace_root / raw).resolve()
        if must_exist and not result.exists():
            raise FileNotFoundError(f"Required path does not exist: {result}")
        return result

    module.ensure_directory = ensure_directory
    module.resolve_path = resolve_path


import src.project_paths as _project_paths
_install_project_path_api(_project_paths, REPO_ROOT)
from src.project_paths import A_DATA_ROOT, ensure_directory, resolve_path, SHARED_MODELS_ROOT

# Qwen3-VL/TimeLens-8B 都使用 16x16 vision patch 和 2x2 spatial merge，
# 因此一个合并后的 visual token 对应 32x32 输入像素。base、LoRA 和
# TimeLens 评测必须统一使用同一视觉预算，才能进行公平比较。
PATCH_PIXELS = 32 * 32

from qwen_decode_worker import configure_decoder, process_vision_info, VideoDecodeError


# ═══════════════════════════════════════════════════════════════════════════════════
# TimeLens Prompt (与 heads_finetune.py 完全一致，改这里会破坏推理分布匹配)
# ═══════════════════════════════════════════════════════════════════════════════════

TIMELENS_SYSTEM = (
    "You are a video time analysis assistant. "
    "Given a video and a query, you must locate the exact time segment "
    "where the described event occurs."
)

TIMELENS_USER_TEMPLATE = (
    "You are given a video with multiple frames. The numbers before each video "
    "frame indicate its sampling timestamp (in seconds). Please find the visual "
    "event described by the sentence '{query}', determining its starting and "
    "ending times. The format should be: "
    "'The event happens in <start time> - <end time> seconds'."
)

TIMELENS8_OFFICIAL_PROMPT = (
    "Please find the visual event described by the sentence '{}', determining its "
    "starting and ending times. The format should be: 'The event happens in "
    "<start time> - <end time> seconds'."
)


# ═══════════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════════

def nonnegative_even_frames(value):
    value = int(value)
    if value < 0 or value % 2:
        raise argparse.ArgumentTypeError("--max-frames must be 0 or a positive even integer")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HeadLoRA 微调模型 Temporal Grounding 评估 (纯 generate 路径)"
    )
    p.add_argument("--model-path", required=True,
                   help="HeadLoRA 合并后的模型目录 (heads_finetune.py 里 save_pretrained "
                        "的输出) 或原始 base model 目录 (base baseline)")
    p.add_argument("--processor-path", default=None,
                   help="模型目录缺少 processor 文件时使用的基座模型目录；TimeLens 可自动回退到本地 TimeLens-8B")
    p.add_argument("--anno-json",  required=True)
    p.add_argument("--video-dir",  required=True)
    p.add_argument("--output-dir", default=str(A_DATA_ROOT / "outputs" / "eval_results"))
    p.add_argument("--split",      default="eval",
                   help="本次评估的标识名，写入输出文件名 "
                        "(例如 charades_topk / charades_base / charades_random 等)")
    p.add_argument("--decode-timeout", type=float, default=120.0,
                   help="每次实际 Qwen 解码的超时秒数（默认 120）")
    p.add_argument("--decode-max-attempts", type=int, default=3,
                   help="实际解码最多尝试 3 次（含首次），全部失败则跳过")
    p.add_argument("--decode-chunk-size", type=int, default=64,
                   help="兼容旧命令；不再预检查，此参数不参与采样")

    # 采样控制 - 默认全部对齐训练 (heads_finetune.py 的默认 & 你实际用的值)
    p.add_argument("--max-samples", type=int, default=0,
                   help="从 annotation 展开的全部样本中等间隔采样；"
                        "0 表示使用全部样本。相同 annotation 会得到完全相同的子集")
    p.add_argument("--sample-seed", type=int, default=42,
                   help="兼容旧命令保留；当前等间隔采样不使用随机种子")
    p.add_argument("--fps",         type=float, default=2.0,
                   help="和训练 --fps 保持一致 (训练默认 2.0)")
    p.add_argument("--max-frames", type=nonnegative_even_frames, default=0,
                   help="视频采样帧数上限；0沿用解码器默认，4096预算实验显式传1024")
    p.add_argument("--max-size",    type=int, default=256,
                   help="仅当 --total-tokens<=0 时才用；训练默认走 total-tokens 路径，"
                        "这个值就用不上")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="官方 TimeLens 评测默认 512")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 表示 greedy decoding (确定性、可复现)；> 0 会走采样")
    p.add_argument(
        "--timelens-model", action="store_true",
        help="评测官方 TimeLens-8B 时开启：只切换官方 prompt/processor 和 greedy "
             "generation；输入预算仍与本地 base/LoRA 统一为 fps=2、"
             "min_tokens=64、total_tokens=14336，且不限制帧数",
    )
    p.add_argument("--prompt-mode", choices=["official", "training"],
                   default="official",
                   help="official 使用 TimeLens 官方 Qwen/TimeLens-8B 短 prompt；"
                        "training 使用本项目微调时的 system + 长 prompt")

    # Token 预算 (和训练一致，是决定 IoU 的关键路径)
    p.add_argument("--min-tokens",   type=int, default=64,
                   help="和训练 --min-tokens 保持一致 (训练默认 64，TimeLens 标准)")
    p.add_argument("--total-tokens", type=int, default=14336,
                   help="视觉 token 总上限（默认 14336）。"
                        "≤0 时降级到旧的 max-size resize 路径 (不推荐)。")

    # attention 实现 - 默认对齐训练 (flash_attention_2)，
    # 上一版硬编码 eager 会引入不必要的数值差异
    p.add_argument("--attn-implementation", type=str, default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"],
                   help="和训练 --attn-implementation 保持一致 (训练默认 flash_attention_2)")

    # 多卡模型分片（单进程、一份模型）
    p.add_argument("--num-gpus", type=int, default=0,
                   help="参与模型分片的可见 GPU 数量；0 表示全部可见 GPU。")
    p.add_argument("--device-map", default="balanced",
                   choices=["auto", "balanced", "balanced_low_0", "sequential"],
                   help="Transformers/Accelerate 模型分片策略。")
    p.add_argument("--gpu-mem-gib", type=float, default=0.0,
                   help="每张 GPU 的权重显存上限（GiB）；0 表示按空闲显存计算。")
    p.add_argument("--gpu-reserve-gib", type=float, default=4.0,
                   help="自动计算时每张 GPU 为激活/KV cache 预留的显存（GiB）。")
    p.add_argument("--cpu-mem-gib", type=float, default=0.0,
                   help="CPU offload 上限（GiB）；0 表示禁止 CPU offload。")
    p.add_argument("--parallel-mode", choices=["data", "model"], default="data",
                   help="data=每卡一份模型并行评测不同样本；model=单模型跨卡分片")

    # 指标
    p.add_argument("--iou-thresholds", type=float, nargs="+",
                   default=[0.3, 0.5, 0.7])
    return p


# ═══════════════════════════════════════════════════════════════════════════════════
# 视频采帧
# ═══════════════════════════════════════════════════════════════════════════════════

def sample_frames(video_path: Path, fps: float) -> list:
    """从视频中提取帧，返回 PIL Image 列表。"""
    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
        native_fps = vr.get_avg_fps()
        step = max(1, int(native_fps / fps))
        indices = list(range(0, len(vr), step))
        frames_t = vr.get_batch(indices)
        from PIL import Image
        return [Image.fromarray(frames_t[i].numpy()) for i in range(len(indices))]
    except ImportError:
        pass
    try:
        import cv2
        from PIL import Image
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频：{video_path}")
        native_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        step = max(1, int(native_fps / fps))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = list(range(0, n_frames, step))
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames
    except ImportError:
        pass
    sys.exit("ERROR 请安装 decord 或 opencv-python")


# ═══════════════════════════════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════════════════════════════

def _require_full_model(model_dir: Path) -> None:
    if ((model_dir / "model.safetensors").is_file() or
            any(model_dir.glob("model-*.safetensors")) or
            (model_dir / "pytorch_model.bin").is_file()):
        return
    if (model_dir / "adapter_model.safetensors").is_file():
        raise ValueError(
            f"{model_dir} 只有 PEFT adapter；请先用 scripts/merge_timelens_peft_adapter.py 合并完整模型。"
        )
    if (model_dir / "adapter.pt").is_file() or any(model_dir.glob("sample-*/adapter.pt")):
        raise ValueError(
            f"{model_dir} 只有 HeadLoRA adapter，没有合并后的完整模型权重。"
            "请先用 scripts/merge_gt_head_opsd_adapter.py 合并 adapter，"
            "再把合并目录传给 --model-path。"
        )
    raise FileNotFoundError(f"{model_dir} 没有完整模型权重（model*.safetensors 或 pytorch_model.bin）")


def _load_eval_processor(model_dir: Path, timelens_model: bool,
                         processor_path: Optional[str] = None):
    from transformers import AutoProcessor

    processor_kwargs = dict(trust_remote_code=True, local_files_only=True,
                            padding_side="left", do_resize=False)
    candidates = []
    if processor_path:
        candidates.append(resolve_path(processor_path, must_exist=True))
    candidates.append(model_dir)
    failures = []
    for metadata_name in ("training_metadata.json", "merge_metadata.json"):
        metadata_file = model_dir / metadata_name
        if metadata_file.is_file():
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            base_path = metadata.get("arguments", {}).get("model_path") or metadata.get("base_model")
            if base_path:
                try:
                    candidates.append(resolve_path(base_path, must_exist=True))
                except FileNotFoundError:
                    failures.append(f"{base_path}: metadata 中的基座目录不存在")
    if timelens_model:
        candidates.append(SHARED_MODELS_ROOT / "TimeLens-8B")

    for source in dict.fromkeys(candidates):
        try:
            processor = AutoProcessor.from_pretrained(str(source), **processor_kwargs)
        except (OSError, ValueError) as exc:
            failures.append(f"{source}: {type(exc).__name__}: {exc}")
            continue
        if source != model_dir:
            print(f"  [processor] 模型目录缺少可加载的 processor；改用 {source}", flush=True)
        return processor
    raise ValueError("无法加载 processor。请用 --processor-path 指定训练基座目录。尝试记录：\n" +
                     "\n".join(failures))

def load_model_and_processor(model_dir: Path, num_gpus: int,
                               gpu_mem_gib: float = 0.0,
                               gpu_reserve_gib: float = 4.0,
                               cpu_mem_gib: float = 0.0,
                               device_map: str = "balanced",
                               attn_implementation: str = "flash_attention_2",
                               timelens_model: bool = False,
                               processor_path: Optional[str] = None):
    """加载模型 (base 或 HeadLoRA 合并后的 checkpoint)，多卡均衡分配。"""
    from transformers import AutoModelForImageTextToText

    _require_full_model(model_dir)
    processor = _load_eval_processor(model_dir, timelens_model, processor_path)
    # ⚠️ 上一版这里硬编码 shortest_edge=128，会把视频细节压得远低于训练时；
    # 训练时 collate_fn_with_processor 里没有做这一步 —— 保持 processor 默认。

    # flash-attn 环境检查 (跟 heads_finetune.py 里的降级逻辑保持一致)
    if attn_implementation == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
            print(f"  [attn] flash_attn={flash_attn.__version__}")
        except ImportError:
            print("  [attn] flash_attn 不可用，降级为 sdpa")
            attn_implementation = "sdpa"

    n_gpus = min(num_gpus, torch.cuda.device_count())
    max_memory = {}
    if n_gpus >= 1:
        for i in range(n_gpus):
            if gpu_mem_gib > 0:
                budget_gib = gpu_mem_gib
            else:
                with torch.cuda.device(i):
                    free_bytes, _ = torch.cuda.mem_get_info()
                budget_gib = free_bytes / 1024**3 - gpu_reserve_gib
                if budget_gib < 1:
                    raise RuntimeError(f"GPU{i} 预留显存后不足 1GiB")
            max_memory[i] = f"{budget_gib:.2f}GiB"
        if cpu_mem_gib > 0:
            max_memory["cpu"] = f"{cpu_mem_gib:g}GiB"
        gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
        print(f"  [model] {n_gpus} GPUs: {gpu_names}")
        print(f"  [model] max_memory={max_memory}")

    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map=device_map,
        max_memory=max_memory if max_memory else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    print(f"  [model] attn_implementation={attn_implementation}")
    n_layers = len(model.model.language_model.layers)

    from collections import Counter
    layer_devices = Counter()
    for layer in model.model.language_model.layers:
        layer_devices[str(layer.self_attn.o_proj.weight.device)] += 1
    print(f"  [model] {n_layers} layers 分布: {dict(layer_devices)}")
    print(f"  [model] hf_device_map={getattr(model, 'hf_device_map', {})}")

    if cpu_mem_gib <= 0:
        non_cuda = [(name, str(param.device)) for name, param in model.named_parameters()
                    if param.device.type != "cuda"]
        if non_cuda:
            raise RuntimeError("模型发生 CPU/disk offload；请增加 GPU 预算或显式设置 "
                               f"--cpu-mem-gib。示例：{non_cuda[:3]}")

    return model, processor, n_layers


def load_model_and_processor_single_gpu(model_dir: Path, gpu_id: int,
                                        attn_implementation: str = "flash_attention_2",
                                        timelens_model: bool = False,
                                        processor_path: Optional[str] = None):
    """在指定单张 GPU 上加载完整模型（多卡数据并行时每个进程调用一次）。
    不用 device_map='auto' 的跨卡分片，而是整个模型放到一张卡上。"""
    from transformers import AutoModelForImageTextToText

    _require_full_model(model_dir)
    processor = _load_eval_processor(model_dir, timelens_model, processor_path)
    # 不手动改 processor.video_processor.size（和训练 collate 保持一致）

    if attn_implementation == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
            print(f"  [attn] flash_attn={flash_attn.__version__}")
        except ImportError:
            print("  [attn] flash_attn 不可用，降级为 sdpa")
            attn_implementation = "sdpa"

    device = torch.device(f"cuda:{gpu_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map={"": device},
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    print(f"  [GPU{gpu_id}] 模型加载完成 device={device} "
          f"attn={attn_implementation}")
    return model, processor, len(model.model.language_model.layers)


# ═══════════════════════════════════════════════════════════════════════════════════
# 输入构建 (TimeLens 格式，视频参数和 heads_finetune.py 的 collate_fn_with_processor
# 完全一致：sample_fps 用外部传入的 fps；开启 total-tokens 时走 pixel-budget 路径
# 而不是自己在外面 resize)。
#
# 这里 add_generation_prompt=True 是关键：让 chat_template 在 assistant 位置留出
# 生成起点，模型才知道该"开始回答了"，而不是在补全 user turn。
# ═══════════════════════════════════════════════════════════════════════════════════

def build_inputs_timelens(processor, video_path: Path, query: str, device,
                          sample_fps: float,
                          min_tokens: int, total_tokens: int,
                          prompt_mode: str = "official",
                          max_frames: int = 0,
                          ) -> Optional[dict]:
    """按 TimeLens 官方路径直接从视频文件构建 generate 输入。"""

    if video_path is None or not Path(video_path).is_file():
        return None

    user_text = (TIMELENS8_OFFICIAL_PROMPT.format(query)
                 if prompt_mode == "official"
                 else TIMELENS_USER_TEMPLATE.format(query=query))

    # 和训练时 collate_fn_with_processor 里的 video_content 走同一条路径
    video_content = {"type": "video", "video": str(video_path), "fps": sample_fps}
    if max_frames:
        video_content["max_frames"] = max_frames
    if total_tokens and total_tokens > 0:
        video_content["min_pixels"]   = min_tokens * PATCH_PIXELS
        video_content["total_pixels"] = total_tokens * PATCH_PIXELS

    if prompt_mode == "official":
        messages = [{"role": "user", "content": [
            video_content,
            {"type": "text", "text": user_text},
        ]}]
    else:
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": TIMELENS_SYSTEM}
            ]},
            {"role": "user", "content": [
                video_content,
                {"type": "text", "text": user_text},
            ]},
        ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    # The input geometry is shared by Qwen3-VL base, its LoRA checkpoints and
    # TimeLens-8B. The timelens_model flag only changes prompt/generation logic.
    vision_kwargs = dict(
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, **vision_kwargs,
    )
    if video_inputs:
        video_tensors, video_metadatas = zip(*video_inputs)
        video_tensors, video_metadatas = list(video_tensors), list(video_metadatas)
    else:
        video_tensors, video_metadatas = video_inputs, None

    scalar_video_kwargs = {
        k: (v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v)
        for k, v in video_kwargs.items()
    }

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_tensors,
        video_metadata=video_metadatas,
        padding=True,
        return_tensors="pt",
        **scalar_video_kwargs,
    )
    return {k: v.to(device) if hasattr(v, "to") else v
            for k, v in inputs.items()}


# ═══════════════════════════════════════════════════════════════════════════════════
# 时间戳解析 & IoU
# ═══════════════════════════════════════════════════════════════════════════════════

# 模式 1: TimeLens 格式 "The event happens in X - Y seconds."
_RE_TIMELENS = re.compile(
    r'(?:event\s+happens\s+in|event\s+occurs\s+in|starts\s+at|begins\s+at)'
    r'\s+([\d.]+)\s*[-–—to]+\s*([\d.]+)\s*seconds?',
    re.IGNORECASE,
)

# 模式 2: "start: X, end: Y"
_RE_STANDARD = re.compile(
    r"start\s*[::：]\s*([\d.]+)[,，\s]+end\s*[::：]\s*([\d.]+)",
    re.IGNORECASE,
)

# 模式 3: 纯数字
_RE_NUMS = re.compile(r"[\d.]+")


def parse_pred(text: str, duration: float = 120.0) -> Tuple[float, float]:
    """复现 TimeLens 官方 ``extract_time``，多组预测只取第一组。"""
    paragraph = text.lower()
    time_pattern = re.compile(
        r"\b(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?)\b")
    matches = time_pattern.findall(paragraph)
    matches = matches[:len(matches) // 2 * 2]
    timestamps = []
    if matches:
        converted = []
        for value in matches:
            parts = value.split(":")
            if len(parts) == 3:
                converted.append(int(parts[0]) * 3600 + int(parts[1]) * 60
                                 + float(parts[2]))
            else:
                converted.append(int(parts[0]) * 60 + float(parts[1]))
        timestamps = list(zip(converted[::2], converted[1::2]))
    if not timestamps:
        for pattern in (r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",
                        r"(\d+\.?\d*)\s+to\s+(\d+\.?\d*)"):
            found = re.findall(pattern, paragraph)
            if found:
                timestamps = [(float(start), float(end)) for start, end in found]
                break
    if not timestamps:
        numbers = re.findall(r"\b(\d+\.\d+|\d+)\b", paragraph)
        numbers = numbers[:len(numbers) // 2 * 2]
        timestamps = [(float(numbers[i]), float(numbers[i + 1]))
                      for i in range(0, len(numbers), 2)]
    return timestamps[0] if timestamps else (0.0, 0.0)


def temporal_iou(ps: float, pe: float, gs: float, ge: float) -> float:
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = (pe - ps) + (ge - gs) - inter
    return inter / union if union > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════════
# 推理：Generate 模式 (唯一路径)
# ═══════════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def infer_generate(model, processor, video_path: Path, query: str, device,
                   duration: float, sample_fps: float,
                   min_tokens: int, total_tokens: int,
                   max_new_tokens: int = 64, temperature: float = 0.0,
                   prompt_mode: str = "official",
                   max_frames: int = 0,
                   ) -> Tuple[Tuple[float, float], str]:
    """
    让模型生成 TimeLens 格式的答案，解析时间戳。
    返回 ((start, end), raw_text)。
    """
    inputs = build_inputs_timelens(
        processor, video_path, query, device,
        sample_fps=sample_fps,
        min_tokens=min_tokens, total_tokens=total_tokens,
        prompt_mode=prompt_mode, max_frames=max_frames,
    )
    if inputs is None:
        return (0.0, 0.0), ""

    # TimeLens 官方评测统一使用 greedy，并清空 checkpoint 采样参数。
    gkw = dict(max_new_tokens=max_new_tokens, do_sample=False,
               temperature=None, top_p=None, top_k=None)

    gen_ids = model.generate(
        **inputs,
        pad_token_id=(processor.tokenizer.pad_token_id
                      or processor.tokenizer.eos_token_id),
        **gkw,
    )
    prompt_len = inputs["input_ids"].shape[1]
    text = processor.tokenizer.decode(
        gen_ids[0, prompt_len:], skip_special_tokens=True,
    )
    pred = parse_pred(text, duration)
    return pred, text


# ═══════════════════════════════════════════════════════════════════════════════════
# 指标累积器
# ═══════════════════════════════════════════════════════════════════════════════════

class Metrics:
    def __init__(self, thresholds: List[float]):
        self.th = thresholds
        self.ious: List[float] = []
        self.hits: Dict[float, int] = {t: 0 for t in thresholds}
        self.total = 0
        self.no_vid = 0
        self.fail = 0
        self.num_oom = 0

    def update(self, ps: float, pe: float, gs: float, ge: float,
               has_video: bool, parse_ok: bool):
        self.total += 1
        if not has_video:
            self.no_vid += 1
            return
        if not parse_ok:
            self.fail += 1
            self.ious.append(0.0)
            return
        iou = temporal_iou(ps, pe, gs, ge)
        self.ious.append(iou)
        for t in self.th:
            if iou >= t:
                self.hits[t] += 1

    def summary(self) -> dict:
        n = max(len(self.ious), 1)
        d = {
            "total": self.total,
            "evaluated": len(self.ious),
            "no_video": self.no_vid,
            "parse_fail": self.fail,
            "num_oom": self.num_oom,
            "mIoU": round(sum(self.ious) / n, 4),
        }
        for t in self.th:
            d[f"R@1_IoU{t}"] = round(self.hits[t] / n, 4)
        return d

    def print_table(self, title: str = "") -> dict:
        d = self.summary()
        n = d["evaluated"]
        print(f"\n{'='*64}")
        print(f"  {title}")
        print(f"  样本: {d['total']}  有效推理: {n}  "
              f"(无视频: {d['no_video']}  解析失败: {d['parse_fail']}"
              f"  OOM: {d['num_oom']})")
        print(f"  {'-'*46}")
        print(f"  {'指标':<22} {'值':>8}  {'命中/总数':>12}")
        print(f"  {'-'*46}")
        print(f"  {'mIoU':<22} {d['mIoU']:>8.4f}")
        for t in self.th:
            k = f"R@1_IoU{t}"
            print(f"  {k:<22} {d[k]:>8.4f}  {self.hits[t]:>6}/{n}")
        print(f"{'='*64}\n")
        return d

    @classmethod
    def merge(cls, metrics_list):
        """合并多个 worker 的 Metrics（多卡数据并行时主进程调用）。"""
        if not metrics_list:
            return cls([0.3, 0.5, 0.7])
        m = cls(metrics_list[0].th)
        for o in metrics_list:
            m.ious.extend(o.ious)
            m.total += o.total
            m.no_vid += o.no_vid
            m.fail += o.fail
            m.num_oom += o.num_oom
            for t in m.th:
                m.hits[t] += o.hits[t]
        return m


# ═══════════════════════════════════════════════════════════════════════════════════
# 样本加载 (兼容多种数据集格式，跟微调路径无关，原样保留)
# ═══════════════════════════════════════════════════════════════════════════════════

_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".webm"}
_VIDEO_INDEX_CACHE: dict[str, dict[str, Path]] = {}


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


def get_video_duration(video_path: Path) -> float:
    from qwen_decode_worker import get_video_duration as isolated_duration
    return isolated_duration(video_path)


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


# ═══════════════════════════════════════════════════════════════════════════════════
# Dataset
#
# 帧处理和外部 resize 策略：
#   - 如果 total_tokens > 0: 不限制帧数、不在外面手动 resize，
#     交给 processor 内部按 min_pixels / total_pixels 处理
#   - 如果 total_tokens <= 0: 用 max_size 手动 resize (旧路径，不推荐)
# ═══════════════════════════════════════════════════════════════════════════════════

class EvalDataset(Dataset):
    def __init__(self, samples, video_dir: Path, fps, max_size,
                 total_tokens: int):
        self.samples = samples
        self.video_dir = video_dir
        self.fps = fps
        self.max_size = max_size
        self.total_tokens = total_tokens

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        vp = resolve_video_path(self.video_dir, s)
        duration = float(s.get("duration", 0.0) or 0.0)
        if vp is not None:
            try:
                if duration <= 0:
                    duration = get_video_duration(vp)
            except Exception as e:
                print(f"  [WARN] {vp.name}: {e}")
        duration = max(duration, float(s["gt_end"]))
        return {
            **{k: s[k] for k in ("video_id", "query", "gt_start", "gt_end")},
            "duration": duration,
            "has_video": vp is not None and vp.is_file(),
            "video_path": str(vp) if vp else None,
            "fps": self.fps,
        }


# ═══════════════════════════════════════════════════════════════════════════════════
# 评估循环（单卡主进程 / 多卡 worker 共用）
# ═══════════════════════════════════════════════════════════════════════════════════

def evaluate_dataset(model, processor, dataset, device, args, gpu_id=None):
    """在给定 model/processor/dataset 上跑完整评估，返回 (acc, records)。
    gpu_id=None 表示主进程（单卡）；否则为多卡 worker 的 gpu 编号。"""
    configure_decoder(args.decode_timeout, args.decode_max_attempts)
    acc = Metrics(args.iou_thresholds)
    records: List[dict] = []

    desc = f"[{args.split}]" if gpu_id is None else f"[{args.split} gpu{gpu_id}]"
    pbar = tqdm(range(len(dataset)), desc=desc, leave=(gpu_id is None))

    for idx in pbar:
        item = dataset[idx]
        gs, ge = float(item["gt_start"]), float(item["gt_end"])
        duration = float(item["duration"])

        if not item["has_video"]:
            acc.update(0, 0, gs, ge, has_video=False, parse_ok=False)
            records.append({
                "video_id": item["video_id"], "query": item["query"],
                "gt_start": gs, "gt_end": ge,
                "pred_start": 0.0, "pred_end": 0.0,
                "iou": 0.0, "status": "no_video", "pred_raw": "",
            })
            continue

        ps, pe = 0.0, 0.0
        pred_raw = ""

        try:
            (ps, pe), pred_raw = infer_generate(
                model, processor, Path(item["video_path"]), item["query"],
                device, duration,
                sample_fps=item.get("fps", args.fps),
                min_tokens=args.min_tokens, total_tokens=args.total_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                prompt_mode=args.prompt_mode, max_frames=args.max_frames,
            )
            # TimeLens 官方 eval_dataloader 按数据集 UNIT 取整；三个基准默认 1 秒。
            ps, pe = round(ps), round(pe)
            parse_ok = not (pred_raw.strip() and ps == 0.0 and pe == 0.0)

        except VideoDecodeError as exc:
            acc.update(0, 0, gs, ge, has_video=False, parse_ok=False)
            records.append({
                "video_id": item["video_id"], "query": item["query"],
                "gt_start": gs, "gt_end": ge,
                "pred_start": 0.0, "pred_end": 0.0, "iou": 0.0,
                "status": "decode_skipped", "pred_raw": "",
                "decode_error": str(exc),
            })
            continue

        except torch.cuda.OutOfMemoryError:
            acc.num_oom += 1
            torch.cuda.empty_cache()
            gc.collect()
            records.append({
                "video_id": item["video_id"], "query": item["query"],
                "gt_start": gs, "gt_end": ge,
                "pred_start": 0.0, "pred_end": 0.0,
                "iou": 0.0, "status": "oom", "pred_raw": "",
            })
            continue
        except Exception as ex:
            # 旧实现静默吞掉异常，会表现为进度增长但始终没有阶段 mIoU。
            # 至少打印前 5 个，便于定位 processor/generate 兼容问题。
            if sum(1 for r in records if str(r.get("status", "")).startswith("error:")) < 5:
                pbar.write(
                    f"[ERROR] idx={idx} video={item['video_id']} "
                    f"{type(ex).__name__}: {ex}"
                )
            records.append({
                "video_id": item["video_id"], "query": item["query"],
                "gt_start": gs, "gt_end": ge,
                "pred_start": 0.0, "pred_end": 0.0,
                "iou": 0.0, "status": f"error:{ex}", "pred_raw": "",
            })
            continue

        iou = temporal_iou(ps, pe, gs, ge)
        acc.update(ps, pe, gs, ge, has_video=True, parse_ok=parse_ok)

        status = "ok"
        if not parse_ok:
            status = "parse_fail"
        elif iou == 0.0:
            status = "zero_iou"

        records.append({
            "video_id": item["video_id"],
            "query": item["query"],
            "gt_start": gs,
            "gt_end": ge,
            "pred_start": round(ps, 2),
            "pred_end": round(pe, 2),
            "iou": round(iou, 4),
            "status": status,
            "pred_raw": pred_raw,
        })

        if (idx + 1) % 20 == 0:
            d = acc.summary()
            pbar.set_postfix(mIoU=f"{d['mIoU']:.3f}",
                             R05=f"{d.get('R@1_IoU0.5', 0):.3f}")

        if (idx + 1) % 50 == 0:
            if gpu_id is not None:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
            else:
                for gi in range(torch.cuda.device_count()):
                    with torch.cuda.device(gi):
                        torch.cuda.empty_cache()

    pbar.close()
    return acc, records


def _worker_entry(gpu_id: int, model_dir: Path, samples: list, video_dir: Path,
                  args, out_dir: Path):
    """单 GPU worker：加载完整模型并把本卡结果写入独立分片文件。"""
    torch.cuda.set_device(gpu_id)

    print(f"\n[GPU{gpu_id}] 加载模型: {model_dir}")
    model, processor, _ = load_model_and_processor_single_gpu(
        model_dir, gpu_id, attn_implementation=args.attn_implementation,
        timelens_model=args.timelens_model, processor_path=args.processor_path)
    device = torch.device(f"cuda:{gpu_id}")

    dataset = EvalDataset(samples, video_dir, args.fps, args.max_size,
                          total_tokens=args.total_tokens)
    print(f"[GPU{gpu_id}] 样本数: {len(dataset)}")

    acc, records = evaluate_dataset(model, processor, dataset, device, args,
                                    gpu_id=gpu_id)
    if len(records) != len(samples):
        raise RuntimeError(f"GPU{gpu_id} 结果数 {len(records)} != 样本数 {len(samples)}")
    for record, sample in zip(records, samples):
        record["_eval_index"] = int(sample["_eval_index"])
    shard = out_dir / f"details_{args.split}.rank{gpu_id}.jsonl"
    with shard.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    state = {
        "thresholds": acc.th, "ious": acc.ious, "hits": acc.hits,
        "total": acc.total, "no_video": acc.no_vid,
        "parse_fail": acc.fail, "num_oom": acc.num_oom,
    }
    (out_dir / f"metrics_{args.split}.rank{gpu_id}.json").write_text(
        json.dumps(state, ensure_ascii=False), encoding="utf-8")
    print(f"[GPU{gpu_id}] 完成 {len(records)} 条，分片={shard}", flush=True)


def _load_worker_metrics(path: Path) -> Metrics:
    state = json.loads(path.read_text(encoding="utf-8"))
    metrics = Metrics([float(value) for value in state["thresholds"]])
    metrics.ious = [float(value) for value in state["ious"]]
    metrics.hits = {float(key): int(value) for key, value in state["hits"].items()}
    metrics.total = int(state["total"])
    metrics.no_vid = int(state["no_video"])
    metrics.fail = int(state["parse_fail"])
    metrics.num_oom = int(state["num_oom"])
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════════

def main(argv=None):
    args = build_parser().parse_args(argv)
    # --split is a filename tag, not a path. Normalize separators before the
    # expensive evaluation so every worker writes into --output-dir itself.
    safe_split = re.sub(r"[/\\]+", "_", args.split).strip("_")
    if not safe_split or safe_split in (".", ".."):
        build_parser().error("--split 必须包含有效的文件名标识")
    if safe_split != args.split:
        print(f"[split] {args.split!r} -> {safe_split!r}", flush=True)
        args.split = safe_split
    if int(__import__("os").environ.get("WORLD_SIZE", "1")) > 1:
        sys.exit("ERROR: 本脚本自行创建每卡 worker，请用单进程 python，不能使用 torchrun/DDP")
    # --timelens-model only changes model-specific prompt/processor/generation.
    # It must not silently replace the local 14336-token comparison protocol.
    out_dir = ensure_directory(Path(args.output_dir))

    # ── 环境 ──
    total_gpus = torch.cuda.device_count()
    num_gpus = args.num_gpus if args.num_gpus > 0 else total_gpus
    num_gpus = min(num_gpus, total_gpus)
    if num_gpus == 0:
        sys.exit("ERROR: 没有可用的 GPU")
    print(f"CUDA: {torch.cuda.is_available()}  |  总 GPU: {total_gpus}  |  使用: {num_gpus}")
    for gi in range(total_gpus):
        print(f"    GPU{gi}: {torch.cuda.get_device_name(gi)}")

    # ── 打印对齐配置 ──
    mode_name = ("Qwen/TimeLens短提示词评测（视频预算按显式参数）"
                 if args.prompt_mode == "official" else "训练 prompt 横向对比协议")
    print(f"\n[{mode_name}]")
    print(f"  fps            = {args.fps}")
    print(f"  max_frames     = {args.max_frames or 'decoder default'}")
    print(f"  min_tokens     = {args.min_tokens}")
    print(f"  total_tokens   = {args.total_tokens}")
    print(f"  max_new_tokens = {args.max_new_tokens}")
    print(f"  prompt_mode    = {args.prompt_mode}")
    patch_side = int(PATCH_PIXELS ** 0.5)
    print(f"  patch_budget   = {patch_side}x{patch_side}")
    print(f"  attn_impl      = {args.attn_implementation}")
    if args.total_tokens <= 0:
        print(f"  [WARN] total_tokens<=0，走 max-size={args.max_size} 的旧路径，"
              f"和训练默认路径不一致，IoU 可能会差")

    # ── 解析路径 ──
    model_dir = resolve_path(
        args.model_path,
        default=SHARED_MODELS_ROOT / "Qwen3-VL-8B-Instruct",
        must_exist=True,
    )
    anno_json = resolve_path(args.anno_json, must_exist=True)
    video_dir = resolve_path(args.video_dir, must_exist=True)
    print(f"\n模型: {model_dir}")
    print(f"标注: {anno_json}")
    print(f"视频: {video_dir}")

    # ── 加载数据 ──
    samples = load_samples(anno_json, video_dir, args.max_samples)
    if not samples:
        print("[ERROR] 未加载到样本")
        return 1

    # ═══════════════════════════════════════════════════════════════════════════
    # 评估：数据并行（每卡一份模型）或单模型跨卡分片
    # ═══════════════════════════════════════════════════════════════════════════
    t0 = time.time()
    model = None
    if args.parallel_mode == "data" and num_gpus > 1:
        print(f"\n{'─'*64}")
        print(f"  模式: generate (数据并行 ×{num_gpus}，每卡一份完整模型)")
        print(f"{'─'*64}")
        indexed = [dict(sample, _eval_index=index) for index, sample in enumerate(samples)]
        shards = [indexed[gpu_id::num_gpus] for gpu_id in range(num_gpus)]
        import multiprocessing as mp
        context = mp.get_context("spawn")
        processes = []
        for gpu_id, shard in enumerate(shards):
            process = context.Process(
                target=_worker_entry,
                args=(gpu_id, model_dir, shard, video_dir, args, out_dir),
                name=f"eval-gpu-{gpu_id}",
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        failed = [(process.name, process.exitcode) for process in processes
                  if process.exitcode != 0]
        if failed:
            raise RuntimeError(f"数据并行 worker 失败：{failed}")
        worker_metrics, records = [], []
        for gpu_id in range(num_gpus):
            worker_metrics.append(_load_worker_metrics(
                out_dir / f"metrics_{args.split}.rank{gpu_id}.json"))
            shard_path = out_dir / f"details_{args.split}.rank{gpu_id}.jsonl"
            for line in shard_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
        records.sort(key=lambda record: record["_eval_index"])
        if len(records) != len(samples) or [r["_eval_index"] for r in records] != list(range(len(samples))):
            raise RuntimeError("数据并行结果缺失、重复或顺序异常")
        for record in records:
            record.pop("_eval_index", None)
        acc = Metrics.merge(worker_metrics)
    else:
        print(f"\n{'─'*64}")
        print(f"  模式: generate (模型分片 ×{num_gpus}, device_map={args.device_map})")
        print(f"{'─'*64}")
        model, processor, _ = load_model_and_processor(
            model_dir, num_gpus=num_gpus, gpu_mem_gib=args.gpu_mem_gib,
            gpu_reserve_gib=args.gpu_reserve_gib, cpu_mem_gib=args.cpu_mem_gib,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
            timelens_model=args.timelens_model,
            processor_path=args.processor_path)
        input_device = model.get_input_embeddings().weight.device
        dataset = EvalDataset(samples, video_dir, args.fps, args.max_size,
                              total_tokens=args.total_tokens)
        print(f"\n推理模式: generate  |  {len(dataset)} 条样本")
        acc, records = evaluate_dataset(model, processor, dataset, input_device, args)

    elapsed = time.time() - t0

    # ── 打印指标 ──
    tag = f"[{args.split}]  {anno_json.name}"
    metrics = acc.print_table(title=tag)

    # ── IoU 分布直方图 ──
    if acc.ious:
        bins = 10
        step = 1.0 / bins
        counts = [0] * bins
        for v in acc.ious:
            counts[min(int(v / step), bins - 1)] += 1
        total = len(acc.ious)
        print("  IoU 分布:")
        for i, c in enumerate(counts):
            lo, hi = i * step, (i + 1) * step
            bar = "█" * int(c / total * 36) if total > 0 else ""
            print(f"    [{lo:.1f},{hi:.1f}) |{bar:<36}| {c:4d} ({c/total*100:5.1f}%)")
        print()

    metrics.update({
        "model_path": str(model_dir),
        "anno_json": str(anno_json),
        "split": args.split,
        "elapsed_sec": round(elapsed, 1),
        "avg_sec_per_sample": round(elapsed / max(len(records), 1), 3),
        "num_gpus": num_gpus,
        "parallel_mode": args.parallel_mode,
        "device_map_strategy": args.device_map if args.parallel_mode == "model" else None,
        "prompt_mode": args.prompt_mode,
        "hf_device_map": ({str(k): str(v) for k, v in
                           getattr(model, "hf_device_map", {}).items()}
                          if model is not None else {}),
        "config": {
            "fps": args.fps, "max_frames": args.max_frames or None,
            "min_tokens": args.min_tokens, "total_tokens": args.total_tokens,
            "attn_implementation": args.attn_implementation,
        },
    })

    # ── 保存 ──
    sp = args.split
    (out_dir / f"metrics_{sp}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2))

    with open(out_dir / f"details_{sp}.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    fails = [r for r in records
             if r["iou"] == 0.0 and r["status"] not in ("no_video", "oom")]
    with open(out_dir / f"failures_{sp}.jsonl", "w", encoding="utf-8") as f:
        for r in fails:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  → metrics_{sp}.json  |  details_{sp}.jsonl  "
          f"|  failures_{sp}.jsonl ({len(fails)} 零 IoU)")

    print(f"\n输出目录: {out_dir}")
    return 0


