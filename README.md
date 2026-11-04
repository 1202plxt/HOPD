# Qwen3-8B / 冻结32B：边界选择性OPD（保留原文件名）

本包在原 `HeadOPD_Qwen3_32B_NVIDIA_autoGPU` 目录内替换现有Python文件内容，原文件名不变。当前仅训练Qwen3-VL-8B学生，教师为独立、始终冻结的Qwen3-VL-32B；默认学习率2e-5。原自教师、多轮和教师刷新逻辑已移除。

| 保留的文件名 | 当前作用 |
|---|---|
| `train_qwen3_32b_direct_head_opd.py` | 原名训练入口，支持边界选择开关及整段对照 |
| `train_qwen3_32b_head_opd_dp4_mp2.py` | 分组分样本、梯度同步、保存和导出 |
| `train_qwen3_boundary_selective_opd.py` | 边界门控、生成数字token映射、选择性KL；直接执行默认启用边界模式 |
| `probe_qwen32b_privileged_k100.py` | 冻结32B加载、正常视频与GT片段构造；当前不做probe |
| `train_timelens_three_group_opsd.py` | Head-LoRA与因果位置工具；不训练TimeLens |
| `qwen_baseline_eval.py` / `qwen_baseline_eval_dp.py` | 单卡/按可见卡分样本普通输入测评 |
| `merge_self_opsd_round.py` | 导出当前方法的adapter快照；不执行自蒸馏 |

其他历史Python文件名保留；历史训练入口转到同一冻结32B训练逻辑，不再支持rounds等旧参数。不相关实验工具的旧实现已清空，执行时提示使用当前入口。名称只为兼容原目录组织，不代表继续保留旧方法。

## 方法

学生输入全局视频和查询，采样生成一条回答。冻结32B教师输入全局视频、查询及带原视频时间戳的GT参考片段，独立贪心生成回答。教师不接收精确GT答案文本。两路视觉token总数匹配，教师全局画面和参考片段共用视觉预算。

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

## 环境与路径

Linux + NVIDIA CUDA + Python 3.11，可使用原qwen3vl环境。新建环境可沿用：

```bash
conda create -n qwen3vl python=3.11 -y
conda activate qwen3vl
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

CUDA wheel需兼容实际驱动。将包内 `Lkmllm_code` 与 `shared_models`、`Lkmllm_data` 并列。之后所有运行命令在 `Lkmllm_code` 目录中执行：

```bash
cd Lkmllm_code
conda activate qwen3vl
```

模型默认路径为 `../shared_models/Qwen3-VL-8B-Instruct` 和 `../shared_models/Video-opd-teacher-Qwen3-VL-32B-GRPO`。训练标注默认 `../Lkmllm_data/datasets/Train/timelens-100k/timelens-100k.jsonl`，视频目录为同名数据目录；JSONL记录格式：

```json
{"video_path":"example.mp4","events":[{"query":"A person opens the door","span":[10.0,20.0]}]}
```

相对video_path以video-dir为基准。预选名单沿用 `heads/qwen3_gtmask_top10.json`，可用 `--heads-json` 指定其他Qwen3名单；不做在线筛选或跨模型迁移。

## 八卡四组运行

```bash
# CPU逻辑与配置测试
python -m unittest discover -s scripts -p 'test_*.py' -v

# 配置预检（不加载模型、不检查显存）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/run_head_opd_8gpu.sh --plan-only

# 8个全局有效样本、2次优化的真实冒烟
OUT="../Lkmllm_data/checkpoints/boundary_smoke_$(date +%Y%m%d_%H%M%S)" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/run_head_opd_8gpu.sh \
  --max-samples 8 --max-attempts 100 --save-every-steps 1

# 正式训练
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/run_head_opd_8gpu.sh
```

默认启用 `--boundary-selective --boundary-epsilon 0`，全局1000个接受样本、有效batch=4、250步，每25步保存adapter。默认14336视觉token/min64/fps2/maxframes0，GT片段最多4096视觉token/fps4/128帧，包含在教师总预算中。仅训练预选head的Q/O LoRA，rank8/alpha16/lr2e-5。

每组两卡，学生在组内第一卡，教师自动放置或分片到组内卡，预留学生权重和激活空间。模型权重不卸载到CPU或磁盘；启动脚本默认将学生保存的反向激活放到CPU。实际显存需目标机器验证。不同组处理不同候选，所有组共享同步的学生参数，教师始终冻结。梯度以CPU float32求和除以全局实际接受数，拒绝组贡献零。补有效batch时可能不足四组同时工作，不保证四倍加速。

可通过 `STUDENT`、`TEACHER`、`ANNO`、`VIDEOS`、`OUT` 环境变量覆盖路径，通过额外参数覆盖 `--lr`、`--max-samples`、`--boundary-epsilon` 等。历史 `run_self_opsd_rounds_8gpu.sh` 文件名也保留并转到当前启动器，但不再支持多轮。

## 双卡一组 / 整段对照

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u scripts/train_qwen3_32b_direct_head_opd.py \
  --student-path ../shared_models/Qwen3-VL-8B-Instruct \
  --teacher-path ../shared_models/Video-opd-teacher-Qwen3-VL-32B-GRPO \
  --anno-json ../Lkmllm_data/datasets/Train/timelens-100k/timelens-100k.jsonl \
  --video-dir ../Lkmllm_data/datasets/Train/timelens-100k \
  --output-dir "../Lkmllm_data/checkpoints/boundary_2gpu_$(date +%Y%m%d_%H%M%S)" \
  --gpus-per-worker 2 --boundary-selective --boundary-epsilon 0 \
  --lr 2e-5 --max-samples 1000 --max-attempts 10000 --grad-accum 4 \
  --offload-saved-tensors
```

先冒烟时再加 `--max-samples 8 --max-attempts 100 --save-every-steps 1`。不传 `--boundary-selective` 时沿用整体IoU严格teacher-better门控、整段学生回答KL的基线。原单卡自教师启动器保留文件名但停止执行并给出提示；当前8B+32B方案至少需要两卡，不承诺单卡显存适配。

## 导出与QV测评

达到接受目标后自动导出完整学生到输出根目录。中间adapter也可单独合并，名称沿用原文件：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/merge_self_opsd_round.py \
  --snapshot /data/checkpoints/boundary_run/step-000025 \
  --output-dir /data/checkpoints/boundary_step25_merged

CUDA_VISIBLE_DEVICES=0 python -u scripts/qwen_baseline_eval.py \
  --model-family qwen3vl --model-path /data/checkpoints/boundary_run \
  --anno-json ../Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights-timelens.json \
  --video-dir ../Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights \
  --output-dir ../Lkmllm_data/outputs/boundary_qv_run \
  --split Qvhighlights --fps 2 --min-tokens 4 \
  --total-tokens 14336 --max-frames 1024 --max-new-tokens 1024
```

多卡测评时将文件改为 `scripts/qwen_baseline_eval_dp.py` 并设置多张可见卡，每张卡加载完整学生并分样本，汇总保留全部样本分母。学生测评只输入普通视频与查询，GT仅用于生成后的指标；贪心、repetition_penalty=1.0、SDPA，缺视频/解码/解析/OOM等失败计零。结果含mIoU和R@0.3/0.5/0.7；训练min64与测评min4的差异需报告。

## 日志与验证范围

train.jsonl记录师生预测、IoU、各边界误差/收益、选择结果、数字token索引及跳过原因。training_metadata.json记录模型config/index摘要和shard大小、head/标注/源码摘要及GPU布局；未计算完整基座权重摘要。summary.json记录全局接受/候选数、步数及单边界/双边界接受计数。快照为weights-only，不支持完整优化器/RNG恢复。输出必须是新目录。

本地14项CPU测试通过；3项真实PyTorch梯度/因果位置/候选反向测试因无PyTorch而跳过。语法与原名入口help通过。真实tokenizer、GPU前后向、分布式通信、显存、导出和下游指标仍需服务器验证，不声明已有性能提升。先冒烟再比较原模型、整段蒸馏与边界蒸馏；固定基础权重、head、预算、学习率和测评协议，并报告两种门控接受的事件可能不同。

上传GitHub时上传本文件夹内容即可；不上传模型、数据或运行产物。文件校验在MANIFEST.sha256。
