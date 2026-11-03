# Head-OPD：Qwen3-VL-8B + 32B-GRPO 教师（NVIDIA 多卡自动分配）

本包提供 NVIDIA 多卡自动分配训练和按可见显卡数量分样本测评，支持按环境提供的 4 卡或 8 卡使用。默认直接使用预先选好的 10 个 Qwen3-VL-8B head；**不需要运行 probe、GT-mask 筛选，也不需要提供诊断 `summary.json` / `protocol.json`**。训练中的 GT 参考视频仍然需要 GT 区间标注。

## 交付范围

本包只有一个训练入口：`scripts/train_qwen3_32b_direct_head_opd.py`，用于 Qwen3-VL-8B 学生和 32B-GRPO 教师。测评入口是 `scripts/qwen_baseline_eval_dp.py`；`qwen_baseline_eval.py` 和 `qwen_decode_worker.py` 供其调用。`scripts` 内其他 Python 文件是沿用原实现的内部依赖，历史名称不代表额外的训练任务，不需要单独运行。不会执行 head 探测或 GT-mask 筛选。

## 1. 方法和 head

head 列表：`Lkmllm_code/heads/qwen3_gtmask_top10.json`。`layer`、`head` 均从 0 开始：

```
(16,21) (19,31) (18,19) (18,15) (16,23)
(20,23) (20,29) (17,4) (22,4) (20,14)
```

这些坐标来自此前报告的 Qwen3-VL-8B、4096 预算筛选，在 8192/14336 训练中直接复用。入口会验证学生模型架构和坐标范围，但**不会因维度一致就声称不同权重具有相同身份或效果**。要更换 head，可修改 JSON 中 `heads` 列表，或使用 `--heads-json /path/to/heads.json`；本包仅支持所校验的 Qwen3-VL-8B 架构布局。

只训练所选 head 的 Q/O 低秩参数，教师冻结；学生采样一次回答，该回答同时用于学生 IoU 门控和蒸馏。教师贪心预测与 GT 比较，仅教师有效且 IoU 严格高于学生时接受。损失为全回答、全词表 `KL(student || teacher)`；tokenizer 兼容性由代码检查。GT 用于教师参考片段和训练筛选，学生测评时只接收正常视频和查询。

训练和评测分别自动识别**当前进程可见**的 NVIDIA GPU 数量；如果集群已设置 `CUDA_VISIBLE_DEVICES`，不会越过该分配。`--student-device` 使用可见列表内的逻辑编号。教师优先在其余卡上自动分片，再加载学生；如果其余卡容量不足，会尝试在学生卡的预留空间之外加载教师分片。学生的视频输入及训练前后向需放在学生所在卡。学生权重以分片文件大小估算，实际峰值请用真实视频检查；显存不足会报错，不会自动量化或将教师权重放到 CPU。

两种输入模式：

- **`extra8192`（默认）**：学生完整视频至多 8192 token；教师看到相同完整视频，另加最多 1024 token 的 GT 片段和文本引导。教师总视觉 token 可超过 8192。
- **`matched14336`**：学生完整视频至多 14336；教师全局视频与 GT 片段共用 14336 预算，需要为参考片段减少教师全局视频采样。这不是在 14336 之外追加。

两个模式都是从命令中指定的学生完整权重开启**新一轮**；优化器和 adapter 重新初始化。weights-only `step-*` 不是优化器/RNG 的完整断点恢复。

## 2. 环境和文件

- 准备至少 **两张 NVIDIA GPU**；4 卡或 8 卡时会自动识别实际可见显卡数量与空闲显存。教师约 33B 参数采用 BF16，先自动分片加载教师，再加载学生。默认学生在逻辑 GPU0；完整模型、视频 token 对应的前向与反向在同一学生 GPU 上。教师优先分配在其他 GPU，如果空间不足才使用学生 GPU 的预留外空间。模型权重不会自动量化或回退到 CPU；命令中的 `--offload-saved-tensors` 会将部分学生反向所需张量转移到 CPU 内存。具体显存和峰值请以设备实际值为准。
- 使用已配置 PyTorch CUDA、Transformers（包含 Qwen3-VL / `AutoModelForImageTextToText`）、Accelerate、NumPy、tqdm、Pillow、torchvision、`qwen-vl-utils` 和 decord 的 Conda 环境。模型的 forward 还需支持 `logits_to_keep`；环境应与先前可运行模型的版本一致。
- 单独准备原始 `Qwen3-VL-8B-Instruct`、`Video-opd-teacher-Qwen3-VL-32B-GRPO` 的完整权重和 processor、训练及测试标注与视频。本包不包含这些大文件。
- 训练标注的文件路径中必须含名为 `Train` 的目录（大小写不敏感）。训练 JSONL 每行包含 `video_path` 与 `events`；event 包含 `query` 与秒为单位的 `span: [start, end]`。相对视频路径以 `--video-dir` 为基准。
- 解压后保留 `HeadOPD_Qwen3_32B_NVIDIA_autoGPU/Lkmllm_code` 目录名与内部结构；辅助脚本文件名带 `_k100` 是历史命名，入口使用 NVIDIA CUDA，不需要 HCU 环境。

激活示例（环境名替换为自己机器实际名称）：

```bash
conda activate qwen3vl
cd /解压目录/HeadOPD_Qwen3_32B_NVIDIA_autoGPU/Lkmllm_code

unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
# 可选：指定当前任务可用的显卡；不设置则使用环境中所有可见显卡。
# 例如 4 卡：export CUDA_VISIBLE_DEVICES=0,1,2,3
# 例如 8 卡：export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python - <<'PY'
import sys, torch, transformers, accelerate
from transformers import AutoModelForImageTextToText, AutoProcessor
import qwen_vl_utils, decord, numpy, tqdm
print('Python:', sys.executable)
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('Transformers:', transformers.__version__)
print('Accelerate:', accelerate.__version__)
print('GPU 数:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

## 3. 训练：8192 + 额外 GT 帧

在代码根目录执行，先把路径替换为接收者机器上的实际绝对路径：

```bash
export STUDENT=/path/to/Qwen3-VL-8B-Instruct
export TEACHER=/path/to/Video-opd-teacher-Qwen3-VL-32B-GRPO
export TRAIN_JSON=/path/to/Train/timelens-100k/timelens-100k.jsonl
export TRAIN_VIDEO=/path/to/Train/timelens-100k
export OUT=/path/to/checkpoints/qwen3_extra8192_1k

TRAIN_ARGS=(
  --student-path "$STUDENT" --teacher-path "$TEACHER"
  --anno-json "$TRAIN_JSON" --video-dir "$TRAIN_VIDEO" --output-dir "$OUT"
  --input-scheme extra8192 --teacher-mode gt_clip --gate teacher-better
  --max-samples 1000 --max-attempts 10000 --epochs 1
  --student-total-tokens 8192 --student-min-tokens 4
  --teacher-total-tokens 8192 --teacher-min-tokens 4
  --total-tokens 8192 --min-tokens 4 --fps 2 --max-frames 768
  --pool-factor 2 --max-pool-frames 1536
  --clip-total-tokens 1024 --clip-fps 4 --clip-max-frames 128
  --clip-padding-seconds 0 --lr 2e-5 --rank 8 --alpha 16 --grad-accum 4
  --max-new-tokens 80 --max-grad-norm 1
  --student-device 0 --student-reserve-gib 12 --teacher-reserve-gib 4
  --offload-saved-tensors
  --save-every-steps 25 --seed 42
)

# 仅检查文件和配置，不加载模型或训练：
python -u scripts/train_qwen3_32b_direct_head_opd.py "${TRAIN_ARGS[@]}" --plan-only

# 预检通过后，在同一终端正式训练：
python -u scripts/train_qwen3_32b_direct_head_opd.py "${TRAIN_ARGS[@]}"
```

先在独立的新输出目录中，把 `--max-samples` 改为 4、`--max-attempts` 改为 100、`--save-every-steps` 改为 1 并添加 `--skip-merge`，检查真实 GPU 与视频前后向；再更换新目录执行完整训练。`--plan-only` 不验证模型加载、视频处理或显存。

有效样本目标为 1000；每 4 个有效样本更新一次，每 25 步保存一次 adapter，即每 100 个有效样本保存。实际目标完成并正常合并后，`summary.json` 的 `training_complete` 与 `merged` 才为 true；模型输出目录可直接用于下方测评。`train.jsonl`、`training_metadata.json` 和 `accepted_samples.jsonl` 记录 head 列表、实际输入、门控和训练情况。

## 4. 可选：历史 14336 匹配预算训练

变量沿用上面，但使用新的 `$OUT`：

```bash
export OUT=/path/to/checkpoints/qwen3_matched14336_1k

python -u scripts/train_qwen3_32b_direct_head_opd.py \
  --student-path "$STUDENT" --teacher-path "$TEACHER" \
  --anno-json "$TRAIN_JSON" --video-dir "$TRAIN_VIDEO" --output-dir "$OUT" \
  --input-scheme matched14336 --teacher-mode gt_clip --gate teacher-better \
  --max-samples 1000 --max-attempts 10000 --epochs 1 \
  --student-total-tokens 14336 --student-min-tokens 64 \
  --teacher-total-tokens 14336 --teacher-min-tokens 64 \
  --total-tokens 14336 --min-tokens 64 --fps 2 --max-frames 0 \
  --pool-factor 2 --max-pool-frames 1536 \
  --clip-total-tokens 4096 --clip-fps 4 --clip-max-frames 128 \
  --clip-padding-seconds 0 --lr 2e-5 --rank 8 --alpha 16 --grad-accum 4 \
  --max-new-tokens 80 --max-grad-norm 1 \
  --student-device 0 --student-reserve-gib 12 --teacher-reserve-gib 4 \
  --offload-saved-tensors \
  --save-every-steps 25 --seed 42
```

添加 `--plan-only` 可先检查；需注意此方案中的 GT 参考片段属于教师 14336 总预算。

## 5. 自动多卡测评三个数据集

评测自动识别所有可见 GPU，**每张卡加载一份完整学生模型**，按样本分片，结束后合并。先确保训练 `summary.json` 中 `training_complete=true`、`merged=true`；以下 `MODEL` 可指向该完整模型目录，也可指向待比较的原始模型。

```bash
export MODEL=/path/to/completed/merged/model
export TEST_ROOT=/path/to/datasets/Test
export EVAL_ROOT=/path/to/evaluation/new_output

unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
# 可选：指定当前任务可用的显卡；不设置则使用环境中所有可见显卡。
# 例如 4 卡：export CUDA_VISIBLE_DEVICES=0,1,2,3
# 例如 8 卡：export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

eval_one () {
  local split="$1" anno="$2" videos="$3"
  python -u scripts/qwen_baseline_eval_dp.py \
    --model-family qwen3vl --model-path "$MODEL" \
    --anno-json "$anno" --video-dir "$videos" \
    --output-dir "$EVAL_ROOT/$split" --split "$split" \
    --fps 2 --min-tokens 4 --total-tokens 8192 --max-frames 768 \
    --max-new-tokens 1024 --repetition-penalty 1.0 --max-samples 0 \
    --attn-implementation sdpa
}

eval_one Qvhighlights "$TEST_ROOT/Qvhighlights/qvhighlights-timelens.json" "$TEST_ROOT/Qvhighlights/qvhighlights" && \
eval_one Charades_sta "$TEST_ROOT/Charades_sta/charades-timelens.json" "$TEST_ROOT/Charades_sta/charades" && \
eval_one Activitynet "$TEST_ROOT/Activitynet/activitynet-timelens.json" "$TEST_ROOT/Activitynet/activitynet"
```

如果本机数据结构不同，修改路径。使用 14336 评测时，把 `--total-tokens` 改为 `14336`、`--max-frames` 改为 `1024`。结果在 `$EVAL_ROOT/<split>/metrics_<split>.json`，检查 `complete`、总样本和失败数；训练完成不意味着测评提升。

## 6. 本包检查范围

文件清单和 SHA256 在 `MANIFEST.sha256`。打包前已进行 Python 语法检查、模拟多卡分配和结果合并检查；**未在 NVIDIA GPU 上完成真实模型加载、前后向及完整评测**。接收者需先按上述步骤小规模验证。请勿只替换配套测评脚本中的一个文件，否则可能出现接口版本不匹配。
