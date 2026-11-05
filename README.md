# Qwen3-8B / 冻结32B：边界选择性OPD

2026-11-05更新：本版本的学生与教师均采用贪心生成（`do_sample=False`、`num_beams=1`，清除temperature/top_p/top_k采样参数）。学生每次优化器更新后，以当前HeadLoRA生成后续回答；同一条学生回答用于门控和逐token全词表reverse KL评分，不另采第二条。训练metadata记录`rollout_generation.policy_version=greedy_v1_20261105`。这是新的生成协议，须使用新OUT，不将历史随机采样checkpoint记作贪心训练。OPD也可使用随机采样，本次改动用于检验贪心生成对定位训练的影响，不保证提升。

使用原文件名。训练自动检测当前环境可见的NVIDIA GPU，默认**每两卡一组**，组间分样本、同步学生梯度；学习率2e-5。

| 可见GPU数 | 自动分组 | 训练方式 |
|---|---|---|
| 2 | 1组，每组2卡 | 单组处理样本 |
| 4 | 2组，每组2卡 | 两组并行处理不同样本 |
| 6 | 3组，每组2卡 | 三组并行处理不同样本 |
| 8 | 4组，每组2卡 | 四组并行处理不同样本 |

单卡会提示至少需要两卡。奇数卡按完整双卡组使用，剩余一张闲置，启动日志明确显示分组和未使用卡。只检测环境可见的卡，不自动占用被CUDA_VISIBLE_DEVICES隐藏的卡。文件名 `run_head_opd_8gpu.sh` 保留，但已不再限制八卡，也不再写死四进程。

## 1. 配置环境

Linux + NVIDIA CUDA + Python 3.11。已有原qwen3vl环境时，直接激活即可；首次安装在本README所在目录执行：

```bash
conda create -n qwen3vl python=3.11 -y
conda activate qwen3vl
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip check
```

CUDA wheel需要与服务器驱动兼容。目录按下面放置；模型和数据不包含在本包中：

```text
LK_OPD/
├── Lkmllm_code/
│   ├── scripts/
│   └── heads/qwen3_gtmask_top10.json
├── shared_models/
│   ├── Qwen3-VL-8B-Instruct/
│   └── Video-opd-teacher-Qwen3-VL-32B-GRPO/
└── Lkmllm_data/
    ├── datasets/Train/timelens-100k/
    │   └── timelens-100k.jsonl
    ├── datasets/Test/
    ├── checkpoints/
    └── outputs/
```

将本包的Lkmllm_code放到上面的对应位置。以下命令全部在这个Lkmllm_code目录运行。

## 2. 预检并跑小样本

```bash
conda activate qwen3vl
cd Lkmllm_code

# 配置预检：自动检测卡数并分组，不加载模型、不创建训练目录
bash scripts/run_head_opd_8gpu.sh --plan-only

# 小样本：全局8个有效样本、2次优化
OUT="../Lkmllm_data/checkpoints/boundary_smoke_$(date +%Y%m%d_%H%M%S)" \
  bash scripts/run_head_opd_8gpu.sh \
  --max-samples 8 --max-attempts 100 --save-every-steps 1
```

例如4卡机器会显示 `visible GPUs=4; groups=2; GPUs/group=2`，两组处理不同候选。预检只验证配置、模型文件身份和数据路径，不验证显存适配或真实前后向。

小样本成功后，summary.json应有 `accepted_events=8`、`optimizer_steps=2`、`training_complete=true`、`merged=true`。候选不足以通过门控时会保存partial-adapter并报错停止。

## 3. 正式训练：相同命令适配2/4/6/8卡

```bash
export OUT="../Lkmllm_data/checkpoints/qwen3_boundary_opd_$(date +%Y%m%d_%H%M%S)"
bash scripts/run_head_opd_8gpu.sh
```

无需手动设置进程数。父进程通过PyTorch检测可见卡数，自动启动 `GPU数÷2` 个训练进程。每组学生放第一卡，冻结32B教师自动放置或分片到该组两卡。各组共享同一版学生参数，不训练多个独立学生。

如需只使用指定卡：

```bash
# 使用指定4张卡，自动分成(0,1)和(2,3)两组
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_head_opd_8gpu.sh

# 使用指定2张卡，一组
CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_head_opd_8gpu.sh
```

默认全局1000个接受样本、有效batch=4、250步，**不会因为组数增加而乘以组数**。每25步保存adapter，结束自动合并学生权重到OUT。默认14336视觉token/min64/fps2/maxframes0，GT片段最多4096视觉token/fps4/128帧，包含在教师总上限内。仅更新预选head的Q/O LoRA，rank8/alpha16/lr2e-5。

不同组门控接受数可以不同。每个有效batch补足实际接受样本，梯度以CPU float32求和除以全局接受数；拒绝组贡献零。补样本时部分组可能闲置，不保证按卡数线性加速。教师始终冻结，不做多轮自教师刷新。每组必须有足够显存容纳师生权重与激活；程序不会把模型权重卸载到CPU/磁盘，默认仅将学生保存的反向激活放到CPU。

## 4. 自定义模型、数据和参数

默认路径不符合服务器时，改下面五个实际路径即可：

```bash
STUDENT=/models/Qwen3-VL-8B-Instruct \
TEACHER=/models/Video-opd-teacher-Qwen3-VL-32B-GRPO \
ANNO=/data/Train/timelens-100k.jsonl \
VIDEOS=/data/Train/timelens-100k \
OUT=/data/checkpoints/qwen3_boundary_run1 \
bash scripts/run_head_opd_8gpu.sh --lr 2e-5 --boundary-epsilon 0
```

ANNO必须是训练集。标注格式示例：

```json
{"video_path":"example.mp4","events":[{"query":"A person opens the door","span":[10.0,20.0]}]}
```

相对video_path以VIDEOS为基准。预选head名单为heads/qwen3_gtmask_top10.json，可通过 `--heads-json` 替换。输出必须是新目录；快照为weights-only，不支持优化器/RNG精确恢复。

## 5. QV测评

在同一终端，正式训练时export的OUT就是完整学生权重目录，训练成功后可直接运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/qwen_baseline_eval.py \
  --model-family qwen3vl --model-path "$OUT" \
  --anno-json ../Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights-timelens.json \
  --video-dir ../Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights \
  --output-dir "${OUT}_qv_eval" \
  --split Qvhighlights --fps 2 --min-tokens 4 \
  --total-tokens 14336 --max-frames 1024 --max-new-tokens 1024
```

另开终端时先将OUT设置为实际训练输出目录。多卡测评将脚本换成 `scripts/qwen_baseline_eval_dp.py` 并设置多张可见卡；每卡一份完整学生、按样本分片汇总。GT只用于模型生成后的指标，失败样本计零并保留在分母。报告mIoU和R@0.3/0.5/0.7，测评min4与训练min64的差异需如实说明。

## 方法

学生输入全局视频和查询，贪心生成一条回答。冻结32B教师输入全局视频、查询及带原视频时间戳的GT参考片段，独立贪心生成回答。教师不接收精确GT答案文本。两路视觉token总数匹配，教师全局画面和参考片段共用视觉预算。

设GT、学生、教师区间分别为 `[sg, eg]`、`[ss, es]`、`[st, et]`：

```text
start_gain = abs(ss - sg) - abs(st - sg)
end_gain   = abs(es - eg) - abs(et - eg)
select_start = start_gain > epsilon
select_end   = end_gain > epsilon
```

`epsilon` 单位为秒，默认0，误差相等不选中。教师预测点落在GT区间内部不等于边界更准。整体IoU只记入日志，**不是边界模式的接受门槛**；教师整体IoU更低时也可能有一个边界值得蒸馏。

| 教师起点更准确 | 教师终点更准确 | 直接损失范围 |
|---|---|---|
| 是 | 是 | 两个边界数字token |
| 是 | 否 | 起点数字token |
| 否 | 是 | 终点数字token |
| 否 | 否 | 跳过样本 |

例如GT `[10,20]`、学生 `[9,24]`、教师 `[12,19]`：起点收益−1秒、终点收益3秒，因此仅蒸馏终点。

教师重新评分的是**学生实际生成的同一条token前缀**，不是教师自己生成的答案，也不是GT teacher-forcing。每个选中位置计算全词表 `KL(student || stopgrad(teacher))`，先平均该样本所选数字token，再在全局有效batch内平均样本损失。其余答案token及未选中边界的直接损失为0；参数共享意味着仍可能间接改变另一边界。

当前仅可靠处理 `The event happens in <start> - <end> seconds.` 格式，支持小数和数字跨多个token。通过原始生成token的逐前缀解码映射起止数字，不重新编码规范答案替代实际rollout。预测无效、格式不支持、映射不稳定、同一token跨越两个数字或混入其他非空白答案文字时跳过并记录原因。教师在独立生成中更准，不保证其在学生前缀下的分布一定有益，需独立评测验证。

## 代码入口和验证范围

Python文件名保持原样。`scripts/train_qwen3_32b_direct_head_opd.py` 也会自动检测卡数；直接调用需传模型、数据和输出参数，并用 `--boundary-selective --boundary-epsilon 0` 启用本方法。不传边界开关则执行整体IoU门控、整段回答KL的对照。推荐使用上面的启动脚本，它已默认启用边界模式。

`train_qwen3_32b_head_opd_dp4_mp2.py` 内部支持任意数量双卡组，dp4只是保留的历史文件名。`merge_self_opsd_round.py` 保留原名用于导出当前方法的adapter，不执行自蒸馏。旧自教师、多轮训练逻辑已移除。

本地检查：18项CPU测试通过，包含师生贪心调用、清除采样参数、2/4/6/8卡分组、奇数卡处理、自动启动命令和全局样本计数；3项真实PyTorch梯度/因果位置/候选反向测试因本机无PyTorch而跳过。真实NVIDIA GPU训练、显存峰值、分布式通信和下游效果仍需服务器验证。

```bash
python -m unittest discover -s scripts -p 'test_*.py' -v
```

训练日志记录师生预测、IoU、边界收益、选择结果、数字token索引、跳过原因和全局计数。上传GitHub时上传本文件夹内容，不上传权重、视频或运行产物；校验清单为MANIFEST.sha256。
