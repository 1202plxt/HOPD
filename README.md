# Qwen3-8B / 冻结32B：Top1 局部门控 OPD

2026-11-06 更新：默认训练目标改为全局 **2000 条有效学生回答**，有效 batch=4，共500次更新。采用今天讨论的 **一次教师评分、Top1 局部替换门控、通过位置的全词表 reverse KL**；不再默认使用教师独立回答的 IoU/边界误差作为训练门槛，也不逐位置生成教师补全。

## 方法

1. 当前 Qwen3-8B 学生根据正常全局视频和查询，贪心生成一次回答。
2. 冻结的32B教师接收全局视频、带原视频时间戳的GT参考片段和精确GT文本，对**同一学生回答前缀**前向评分一次，同时得到所有位置的 logits。教师不独立生成答案。
3. start/end 数字及各自第一个数字结束位置，取教师该位置的 Top1 token，替换学生当前位置。前缀和剩余后缀均保持学生原始 token；前一位置的建议不写回 rollout。
4. 解析替换后的完整回答，对该边界单独比较绝对误差：

```text
local_gain = abs(student_boundary - GT_boundary)
             - abs(replaced_boundary - GT_boundary)
distill_this_position = local_gain > boundary_epsilon
```

默认 epsilon=0 秒，误差相等也屏蔽。非法替换、特殊token、不可靠字符映射、共享两个边界的token、改变其他回答内容、非有限数值或产生无效区间的替换均不放行。实际token可以含多个数字或融合标点，不假设一位数字对应一个token。

5. 普通回答位置正常参与全词表 `KL(student || stopgrad(teacher))`；时间戳和数字结束位置只在门控通过时参与。先对每条回答选中位置平均，再按全局有效batch平均样本梯度。**Top1只用于门控，KL不是Top1/Top100截断分布。**教师logits同时用于门控和KL，没有额外的模型调用。

例如学生 `[14,45]`、GT `[24,80]`：第一位替成2得到24，第二位在真实前缀1下替成8得到18，都可放行；end第一位替成7得到75可放行，第二位替成3得到43应屏蔽。教师建议与原token相同不会减少误差，因此数字位置屏蔽；普通格式token仍可蒸馏。

**局限**：替换会保留学生后缀，不代表教师沿自己的路径最终会输出该数字。若学生为6、GT为16，教师单个token为1，替换结果为1，应拒绝；只有词表里存在单token的16等可解析替换时，才能识别这种位数纠偏。全词表KL仍会传递教师其他token的概率，局部收益不能保证训练或最终IoU单调改善。GT文本和参考片段属于训练期特权信息，学生独立测评时不接收它们。

教师固定为独立32B，学生仅训练预选head的Q/O HeadLoRA，rank8/alpha16/lr2e-5。start、end按各自GT边界判断，整体IoU仅作学生日志指标，不作为新模式门控。学生生成策略仍为贪心：`do_sample=False,num_beams=1`，不增加第二条学生轨迹。此协议区别于历史边界门控和随机rollout实验，必须使用新输出目录。

## 文件与环境

目录沿用：

```text
LK_OPD/
├── Lkmllm_code/
│   ├── scripts/
│   └── heads/qwen3_gtmask_top10.json
├── shared_models/
│   ├── Qwen3-VL-8B-Instruct/
│   └── Video-opd-teacher-Qwen3-VL-32B-GRPO/
└── Lkmllm_data/
    ├── datasets/Train/timelens-100k/timelens-100k.jsonl
    ├── datasets/Test/
    ├── checkpoints/
    └── outputs/
```

模型、视频和权重不包含在仓库中。在已有兼容 Qwen3-VL 的环境中运行；需要 PyTorch、Transformers（Qwen3-VL及视频metadata接口）、Accelerate、qwen-vl-utils、视频解码库。K100使用平台提供的HIP版PyTorch，勿用NVIDIA CUDA wheel替换。环境和硬件适配需实际验证。

## 启动：默认2k，当前终端输出

在 `Lkmllm_code` 目录执行，脚本保留原文件名：

```bash
export OUT="../Lkmllm_data/checkpoints/qwen3_top1gate_opd_2k_$(date +%Y%m%d_%H%M%S)"
bash scripts/run_head_opd_8gpu.sh
```

自动检测可见GPU，默认每两卡一组。2/4/6/8卡分别为1/2/3/4组；多组分候选样本并同步同一个学生的梯度，不是训练多个独立学生。全局2k不会乘以组数，组数增加也不保证线性加速。每组学生在第一卡，教师自动分片/放置到该组GPU，拒绝CPU/disk权重卸载，学生反向激活默认可卸载CPU。每25次更新保存一次adapter，成功后合并完整学生模型到OUT。

NVIDIA指定可见设备：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_head_opd_8gpu.sh
```

双K100指定可见设备（先激活服务器已有HIP环境）：

```bash
env -u CUDA_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES -u HSA_VISIBLE_DEVICES \
  HIP_VISIBLE_DEVICES=0,1 bash scripts/run_head_opd_8gpu.sh
```

HIP分组只设置HIP_VISIBLE_DEVICES，并清除冲突的CUDA/ROCR/HSA隔离变量；NVIDIA分组使用CUDA_VISIBLE_DEVICES。单卡不适合这份8B学生+32B教师配置。训练日志直接输出到当前终端，不自动nohup或重定向。

### 先做小样本验证

```bash
bash scripts/run_head_opd_8gpu.sh --plan-only

OUT="../Lkmllm_data/checkpoints/top1gate_smoke_$(date +%Y%m%d_%H%M%S)" \
  bash scripts/run_head_opd_8gpu.sh \
  --max-samples 8 --max-attempts 100 --save-every-steps 1
```

在K100上给这两条命令同样设置前述HIP隔离变量。plan-only不加载模型、不训练；不能证明显存适配或前后向成功。小样本目标是全局8条有效回答、2次优化，成功摘要应有 `training_complete=true`、`accepted_events=8`、`optimizer_steps=2`、`merged=true`。候选耗尽仍未达到目标会保存partial-adapter并报错。

“有效回答”表示学生预测有效、token映射可靠，普通回答位置可训练；**即使没有任何边界位置改善，该回答仍会训练普通位置并计入2k**。2k不是“2000条至少一个数字改善”的样本。

### 自定义路径

```bash
STUDENT=/models/Qwen3-VL-8B-Instruct \
TEACHER=/models/Video-opd-teacher-Qwen3-VL-32B-GRPO \
ANNO=/data/Train/timelens-100k.jsonl \
VIDEOS=/data/Train/timelens-100k \
OUT=/data/checkpoints/qwen3_top1gate_2k \
bash scripts/run_head_opd_8gpu.sh --lr 2e-5 --boundary-epsilon 0
```

训练ANNO必须位于Train目录。标注支持TimeLens格式：

```json
{"video_path":"example.mp4","events":[{"query":"A person opens the door","span":[10.0,20.0]}]}
```

默认14336视觉token/min64/fps2/maxframes0；GT参考片段最多4096视觉token、fps4、128帧，包含在教师的14336上限内。精确GT文本只写入教师提示，不写入学生提示。head名单为 `heads/qwen3_gtmask_top10.json`，可用 `--heads-json` 替换。默认候选上限10000，用 `--max-attempts` 调整。快照只保存权重，不支持完整优化器/RNG恢复。

## 终端和记录

```text
[token-start] pos=... '1'->'2' value=14->24 error=10->0s DISTILL
[token-end] pos=... '5'->'3' value=45->43 error=35->37s SKIP
[scope] ordinary=... boundary=... total=... full-vocab RKL
[progress] accepted=.../2000 (...) elapsed=...h ETA~...h
```

token替换只是门控建议，不会改写学生答案。`DISTILL`位置匹配完整分布。多组日志可能交错，以每组`[answer]`和训练JSON中的worker_rank对应；`train.jsonl`保存完整逐token门控，`progress.jsonl`保存全局墙钟进度及ETA。ETA基于加载结束后累计墙钟时间/有效样本数，初期不稳定，不是性能承诺。

## 测评

学生独立正常输入；14336预算示例：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/qwen_baseline_eval.py \
  --model-family qwen3vl --model-path "$OUT" \
  --anno-json ../Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights-timelens.json \
  --video-dir ../Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights \
  --output-dir "${OUT}_qv_eval" --split Qvhighlights \
  --fps 2 --min-tokens 64 --total-tokens 14336 --max-frames 0 --max-new-tokens 1024
```

该示例为NVIDIA单卡；K100设置HIP_VISIBLE_DEVICES并清除冲突隔离变量。报告mIoU及R@0.3/0.5/0.7，核对视频处理、标注、模型及输出目录，不能混用历史预算结果。

## 历史对照与验证范围

`--token-local-gate` 是启动脚本默认的新方法。Python入口保留历史 `--boundary-selective`（教师独立生成、按整边界误差筛选、只蒸馏选中数字）以及无这两个开关时的整体IoU门控基线，两个门控开关不能同时传。历史包装入口继续沿用原名；推荐统一使用 `run_head_opd_8gpu.sh`。不要把新旧输出归为同一协议。

本地29项测试：26项CPU测试通过；3项需要真实PyTorch的梯度/因果对齐/反向测试因本机无PyTorch跳过。新增测试覆盖一次学生生成/一次教师评分、Top1门控但分布KL、普通位置与结束位置、数字位数局限、2k默认配置及HIP隔离。GPU训练、HIP性能、显存峰值和测试集提升均未在本地验证。

```bash
python -m unittest discover -s scripts -p 'test_*.py' -v
```

MANIFEST.sha256只核对源码内容，不代表模型训练或指标正确。
