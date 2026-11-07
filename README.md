# Qwen3-8B / 冻结32B：Top10 独立边界门控 OPD

2026-11-07：将启动脚本使用的 token-local 模式从 Top1 升级为 **Top10 概率加权边界误差门控**，并移除假设替换的 `start < end` 约束。该版本与旧 Top1 训练属于不同协议，使用新输出目录；历史测评结果不归属于本次更新。

默认目标仍为全局 **2000 个有效样本**，有效 batch=4，达到目标时共 500 次更新。可用 `--max-samples 1000` 改为 1000 个有效样本、250 次更新。

## 门控与蒸馏

1. 学生接收正常视频和查询，贪心生成一次完整回答。师生提示均要求原视频时间轴上的纯秒数，禁止冒号和分秒表示；这是 prompt 约束，不是硬词表屏蔽。无法解析为规定纯秒数格式的学生回答会跳过。
2. 独立冻结的 32B 教师接收全局视频、GT 参考片段及精确 GT 文本，对同一学生轨迹评分一次；学生也对同一轨迹评分。没有教师独立答案生成或逐位置教师续写。
3. 对 start/end 数字及第一个数字结束位置，取教师该位置 Top10 的真实 token ID。每个 token 独立替换学生当前位置，保留原始前缀和后缀；任何建议均不写回学生轨迹。
4. 候选必须保留其他字段和回答内容，且目标边界是非负有限纯数字。特殊 token、冒号、下划线、修改另一个边界或格式的替换排除。**不要求假设替换后的 start < end**：已预测错误的另一个边界不能阻止本边界向 GT 移动。实际学生轨迹仍要求合法区间。
5. 在同一组有效教师 Top10 候选 `C` 上，分别将教师、学生概率重新归一化，计算目标边界的加权绝对误差：

```text
error(c) = abs(replaced_boundary(c) - GT_boundary)
E_teacher = sum(teacher_prob(c | c in C) * error(c))
E_student = sum(student_prob(c | c in C) * error(c))
E_original = abs(original_student_boundary - GT_boundary)

accept_position = (
    E_original - E_teacher > boundary_epsilon
    and E_student - E_teacher > boundary_epsilon
)
```

默认 epsilon=0，相等也不放行。Top1 自身改善不能代替整个候选集的概率加权比较；个别候选更差也不意味着整组必然拒绝。门控检查的是候选集内的期望误差。

例如学生 `1.0–3.0`、GT `75–78`，起点候选 `1→7` 对应起点误差由 74 降为 68 秒。虽然假设区间为 `7.0–3.0`，这个候选仍进入比较；是否放行由整个有效 Top10 的上述两项条件决定。学生的原始 end=3 及其评分前缀不被改写。

6. 至少一个边界位置通过，样本才执行反向传播、进入梯度累积并增加有效样本计数。没有通过位置时，普通 token 也不单独训练。
7. 有效样本继续在普通位置和通过的边界位置计算完整词表的 `KL(student || stopgrad(teacher))`，先按所选 token 平均，再按全局有效 batch 平均样本梯度。**Top10 用于门控，实际 KL 仍为全词表，未改为 Top10 截断 KL。**

教师完全冻结；仅训练预选 head 的 Q/O HeadLoRA，rank8、alpha16、lr2e-5。默认贪心生成、输出最多80 token。start/end 门控独立，整体 IoU 仅为日志指标。

### 能证明与不能证明的内容

门控通过只证明：在同一有效候选集合上，教师的加权边界误差小于学生分布和原始边界误差。日志保存每个候选误差、师生概率、有效概率质量、`gain_vs_sample_seconds`、`gain_vs_student_seconds` 以及假设区间是否有序。

被排除候选和 Top10 之外的概率没有获得此检验的保证；全词表 KL 仍可能传递它们的信号。共享参数更新也会影响后续起点、终点和其他样本，因此不保证训练更新后的误差或最终 IoU 单调改善。允许跨越另一边界不解决所有数字位数变化或多 token 联合修改问题。

## 文件与环境

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
    └── outputs/eval_results/
```

在服务器已有兼容 Qwen3-VL 的 PyTorch / Transformers / Accelerate / qwen-vl-utils 环境中运行。K100 使用平台 HIP 环境。模型、视频与权重不包含在仓库中。

预算保持现有构造：14336/min64/fps2/maxframes0；教师 GT 参考片段最多4096视觉 token、fps4、128帧，包含在教师总视觉上限中，不是本次额外增加4096。默认使用预选 Qwen3 GT-mask Top10，不重新排名。

## 启动与冒烟

在 `Lkmllm_code` 目录运行，保留历史启动脚本名：

```bash
bash scripts/run_head_opd_8gpu.sh --plan-only

OUT="../Lkmllm_data/checkpoints/top10_independent_smoke_$(date +%Y%m%d_%H%M%S)" \
  bash scripts/run_head_opd_8gpu.sh \
  --max-samples 8 --max-attempts 100 --save-every-steps 1
```

双 K100 的环境前缀为：

```bash
env -u CUDA_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES -u HSA_VISIBLE_DEVICES \
  HIP_VISIBLE_DEVICES=0,1 bash scripts/run_head_opd_8gpu.sh
```

NVIDIA 使用 `CUDA_VISIBLE_DEVICES=0,1`。每两卡一组学生/教师；2/4/6/8卡对应1/2/3/4个同步样本工作组。全局目标不乘以工作组数。学生在组内第一卡，32B 教师自动分片；默认开启学生中间张量 CPU 卸载。

自定义 `STUDENT`、`TEACHER`、`ANNO`、`VIDEOS`、`OUT` 环境变量可以指定实际路径。有效 target 耗尽前候选不足会保存 partial-adapter 并报错，不伪报完成。快照仅保存权重，不是完整优化器/RNG恢复。已运行的旧进程不会自动切换新协议。

## 测评

测评默认加入相同纯秒数指令；`--prompt-mode official` 可显式使用原始提示。GT 仅用于最后评分，不进入模型输入。

```bash
env -u CUDA_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES -u HSA_VISIBLE_DEVICES \
  HIP_VISIBLE_DEVICES=0,1 python -u scripts/qwen_baseline_eval_dp.py \
  --model-path "$OUT" \
  --anno-json ../Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights-timelens.json \
  --video-dir ../Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights \
  --output-dir "../Lkmllm_data/outputs/eval_results/$(basename "$OUT")/Qvhighlights" \
  --split Qvhighlights --prompt-mode decimal_seconds \
  --fps 2 --min-tokens 64 --total-tokens 14336 --max-frames 0 --max-new-tokens 1024
```

双卡按样本分片，每卡一份学生；K100 子进程清除冲突的 CUDA/ROCR/HSA 隔离变量。每个选中样本均计入指标分母，推理失败计零。比较模型时使用相同 prompt、预算和评测配置。

## 历史模式与验证

`--token-local-gate` 是启动脚本的新默认方法。Python 入口保留旧 `--boundary-selective` 和整体 IoU 门控控制模式；与 token-local 开关互斥。它们仍使用各自的历史逻辑，不与新 Top10 结果混写。

本次 CPU 检查覆盖 Top10 同候选集加权误差、Top1/整组结论不同、起止边界交叉、零有效位置跳过反向、一次教师评分、实际 token 前缀保持不变、纯秒数提示与 HIP 隔离：41 项通过；3 项 PyTorch 数值检查因本地没有 PyTorch 跳过。GPU 训练、K100 速度、显存和独立测试集效果待服务器验证。

```bash
python -m unittest discover -s scripts -p 'test_*.py' -v
```

MANIFEST.sha256 核对源码内容，不代表训练或测评已经完成。
