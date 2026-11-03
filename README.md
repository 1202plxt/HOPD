# Qwen3-VL-8B + 32B-GRPO Head-OPD：NVIDIA 自动多卡，14336

唯一训练入口是 `scripts/train_qwen3_32b_direct_head_opd.py`，直接读取指定 head。内部依赖中沿用历史名称的文件无需单独运行。无需执行 probe 或 GT-mask。

## 先配置并激活环境

以下步骤用于 Linux + NVIDIA GPU，从零创建 Python 3.11 环境。先确认 `nvidia-smi` 正常，并且驱动支持 CUDA 12.4。这里安装 PyTorch 的 CUDA 12.4 wheel，无需为 SDPA 另外编译 FlashAttention。

```bash
# 1. 创建 Python 3.11 环境
conda create -n qwen3vl python=3.11 -y
conda activate qwen3vl
python -m pip install --upgrade pip setuptools wheel

# 2. 安装 NVIDIA CUDA 版 PyTorch 和配套 torchvision
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

# 3. 安装模型、分片加载、视频处理及其他依赖
python -m pip install \
  transformers==4.57.1 accelerate safetensors huggingface_hub \
  "qwen-vl-utils[decord]==0.0.14" decord av \
  "numpy<2" pillow tqdm einops sentencepiece protobuf psutil

# 4. 清除其他平台的设备可见性设置
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES

# 5. 检查依赖关系
python -m pip check

# 6. 检查导入、GPU 数量以及 forward 接口
python - <<'PY'
import sys, inspect
import torch, torchvision, transformers, accelerate
import numpy, PIL, tqdm, decord, av, qwen_vl_utils
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import Qwen3VLForConditionalGeneration

print('Python:', sys.version)
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('Transformers:', transformers.__version__)
print('Accelerate:', accelerate.__version__)
assert torch.cuda.is_available(), 'CUDA unavailable: check NVIDIA driver and PyTorch wheel'
print('Visible GPUs:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    info = torch.cuda.get_device_properties(i)
    print(i, info.name, f'{info.total_memory / 1024**3:.2f} GiB')
assert 'logits_to_keep' in inspect.signature(Qwen3VLForConditionalGeneration.forward).parameters
print('Environment imports and forward interface: OK')
PY
```

上述命令创建环境后，后续进入服务器只需 `conda activate qwen3vl`。训练至少需要两张可见 GPU；依赖检查通过后继续下面的目录和运行步骤。这里列出的是安装配置，尚未在接收者 NVIDIA 机器上完成真实模型训练验证。[Qwen3-VL 官方模型说明](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)

## 放置目录并进入代码目录

将本包的 `Lkmllm_code` 放在项目当前根目录，和 `shared_models`、`Lkmllm_data` 并列：

```text
LK_OPD/
├── Lkmllm_code/（本包代码、heads、scripts、src）
├── shared_models/
│   ├── Qwen3-VL-8B-Instruct/
│   └── Video-opd-teacher-Qwen3-VL-32B-GRPO/
└── Lkmllm_data/
    ├── datasets/Train/timelens-100k/
    ├── datasets/Test/
    ├── checkpoints/
    └── outputs/eval_results/
```

从项目根目录进入代码目录：

```bash
cd Lkmllm_code
```

训练自动识别可见 GPU，至少两卡。教师优先在其余卡分片，先加载教师再加载学生；默认学生及其前后向在逻辑 GPU0。其余卡不足时会尝试使用学生卡预留外的空间。4 卡或 8 卡使用同一命令，可用 `CUDA_VISIBLE_DEVICES` 限定卡数。实际显存峰值需在目标机器验证。

## 训练

只提供 14336 训练。学生正常视频上限 14336；教师全局视频与 GT 参考片段共用 14336 上限，其中 GT 片段最多 4096，并附参考片段引导。GT 用于教师片段选取和训练 IoU 门控。实际 token 数写入日志。

head 列表在 `heads/qwen3_gtmask_top10.json`，坐标从 0 开始；所附坐标来自此前 Qwen3 的 4096 筛选，在此直接复用。仅训练指定 head 的 Q/O 低秩参数。

学生一次采样回答同时用于 IoU 和 OPD rollout；教师贪心预测，只有教师有效且 IoU 严格超过当前学生才蒸馏，无额外阈值和差值门槛。损失为 rollout 上的全词表反向 KL。梯度累积 4，每 25 步保存 adapter，即每 100 个有效样本保存。目标为 1000 个有效样本、250 步。

以下命令均在 `Lkmllm_code` 目录中执行：

```bash
OUT="../Lkmllm_data/checkpoints/qwen3_headopd_32b_gtclip_1k_14336_$(date +%Y%m%d_%H%M%S)"

TRAIN_ARGS=(
  --student-path ../shared_models/Qwen3-VL-8B-Instruct
  --teacher-path ../shared_models/Video-opd-teacher-Qwen3-VL-32B-GRPO
  --heads-json heads/qwen3_gtmask_top10.json
  --anno-json ../Lkmllm_data/datasets/Train/timelens-100k/timelens-100k.jsonl
  --video-dir ../Lkmllm_data/datasets/Train/timelens-100k
  --output-dir "$OUT"
  --teacher-mode gt_clip --gate teacher-better
  --max-samples 1000 --max-attempts 10000 --epochs 1
  --student-total-tokens 14336 --student-min-tokens 64
  --teacher-total-tokens 14336 --teacher-min-tokens 64
  --total-tokens 14336 --min-tokens 64 --fps 2 --max-frames 0
  --pool-factor 2 --max-pool-frames 1536
  --clip-total-tokens 4096 --clip-fps 4 --clip-max-frames 128
  --clip-padding-seconds 0
  --lr 2e-5 --rank 8 --alpha 16 --grad-accum 4
  --max-new-tokens 80 --max-grad-norm 1
  --student-device 0 --student-reserve-gib 12 --teacher-reserve-gib 4
  --offload-saved-tensors --save-every-steps 25 --seed 42
)

# 文件和配置预检，不加载模型
python -u scripts/train_qwen3_32b_direct_head_opd.py "${TRAIN_ARGS[@]}" --plan-only

# 正式训练
python -u scripts/train_qwen3_32b_direct_head_opd.py "${TRAIN_ARGS[@]}"
```

JSONL 每条样本包含 `video_path` 和 `events`；event 包含 `query` 和秒为单位的 `span: [start, end]`。相对视频路径以 `--video-dir` 为基准。输出目录自动创建。

训练结束自动合并权重；检查 `summary.json` 的 `training_complete=true`、`merged=true`。候选耗尽可能未达到有效样本目标，以最终日志为准。`step-*` 是 weights-only 快照，不是完整优化器恢复点。

## 自动多卡测评三个数据集

同一终端使用 `MODEL="$OUT"`。另开终端时改为相应的相对路径，例如 `MODEL=../Lkmllm_data/checkpoints/具体训练目录名`。每张可见卡加载完整学生，按样本分片并汇总。三个数据集依次运行。

```bash
MODEL="$OUT"
EVAL_ROOT="../Lkmllm_data/outputs/eval_results/qwen3_headopd_14336_$(date +%Y%m%d_%H%M%S)"
TEST_ROOT=../Lkmllm_data/datasets/Test

eval_one () {
  local split="$1" anno="$2" videos="$3"
  python -u scripts/qwen_baseline_eval_dp.py \
    --model-family qwen3vl --model-path "$MODEL" \
    --anno-json "$anno" --video-dir "$videos" \
    --output-dir "$EVAL_ROOT/$split" --split "$split" \
    --fps 2 --min-tokens 4 --total-tokens 14336 --max-frames 1024 \
    --max-new-tokens 1024 --repetition-penalty 1.0 --max-samples 0 \
    --attn-implementation sdpa
}

eval_one Qvhighlights "$TEST_ROOT/Qvhighlights/qvhighlights-timelens.json" "$TEST_ROOT/Qvhighlights/qvhighlights" && \
eval_one Charades_sta "$TEST_ROOT/Charades_sta/charades-timelens.json" "$TEST_ROOT/Charades_sta/charades" && \
eval_one Activitynet "$TEST_ROOT/Activitynet/activitynet-timelens.json" "$TEST_ROOT/Activitynet/activitynet"
```

测评保持此前 Qwen3 协议：14336 上限、min4、最多 1024 帧；训练为 min64。学生测评只接收正常视频和查询。结果保存于 `$EVAL_ROOT/<split>/metrics_<split>.json`，报告 mIoU、R@0.3/0.5/0.7。测原始学生时，将 `MODEL` 改为 `../shared_models/Qwen3-VL-8B-Instruct`。

## 检查范围

本次交付完成 Python 静态语法和入口参数检查，未在 NVIDIA GPU 上运行真实训练与测评。无测试脚本。文件校验值在 `MANIFEST.sha256`。
