# Qwen3-8B / 冻结32B：Top10 独立边界门控 OPD

2026-11-07：将启动脚本使用的 token-local 模式从 Top1 升级为 **Top10 概率加权边界误差门控**，并移除假设替换的 `start < end` 约束。该版本与旧 Top1 训练属于不同协议，使用新输出目录；历史测评结果不归属于本次更新。

2026-11-08：双卡 NVIDIA 启动脚本默认 **RKL、每轮2000条有效样本、连续两轮**，共4000条、有效batch=4、1000次更新；训练完成并导出完整权重后自动进行三套双卡分样本测评。每轮GT时长配额为2∶4∶4。此前实验分数不归入此协议。

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

## 双卡 NVIDIA：训练两轮后自动测评

在已安装兼容 CUDA PyTorch 的环境中，从仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
LK_OPD_ROOT=/fs0/AI/sx2624011/LKOPD/LK_OPD \
bash Lkmllm_code/scripts/run_head_opd_2gpu.sh
```

`PY` 默认使用当前环境的 `python`；可通过 `PY=/path/to/environment/bin/python` 覆盖。脚本从代码向父目录查找包含 `shared_models` 与 `Lkmllm_data` 的位置，或使用 `LK_OPD_ROOT`。模型与数据默认路径均在脚本中，可以直接修改。原 `run_head_opd_8gpu.sh` 文件名保留，两者现在启动同一套**恰好两卡 NVIDIA**流水线。该 shell 不再自动扩展到其他卡数或 HIP。

默认学生从 `shared_models/Qwen3-VL-8B-Instruct` 起步，教师为冻结的 `Video-opd-teacher-Qwen3-VL-32B-GRPO`。通过 `STUDENT` 可明确覆盖起始完整权重；脚本不会自动载入之前的FKL检查点。双卡训练中学生在第一卡，教师在两卡间分片。

### 两轮与样本分配

- 每轮目标2000条**实际接受**样本：GT时长≤5秒400条，5秒<GT时长≤15秒800条，GT时长>15秒800条。
- 这是GT事件长度分桶，不是视频总时长。>15秒桶仍同时包含15–60秒和>60秒事件。
- 根据剩余有效配额随机选择分桶。拒绝不计数，继续同桶下一候选；候选预先分桶洗牌，取下一条无需再次扫描整个数据集。
- 轮间保留同一个学生、Head-LoRA、AdamW和梯度累积状态；教师保持冻结。不重建优化器，不重新挂载LoRA，不重放前轮候选。
- 两轮合计4000条新有效样本，短/中/长为800/1600/1600，默认1000次参数更新。配额不足或候选耗尽会报错并保留partial，不放宽配额，不进入测评。

参数已写在sh内：`--distill-kl reverse --rounds 2 --max-samples 2000 --duration-ratio 2 4 4 --max-attempts 0 --grad-accum 4`。Top10仍仅用于门控，普通位置与通过边界位置仍使用**全词表RKL**。

训练元数据、逐候选门控审计、轮次、duration_bucket、最终每轮配额与接受样本列表均保留。`--quiet --log-every 10`减少控制台输出，完整JSONL审计不删除。

### 自动测评

训练完成并合并完整学生权重后，依次测评QVHighlights、Charades-STA、ActivityNet。每套均由两张卡独立加载同一学生权重，按样本分片，每条只测一次，最后合并指标。测评进程在训练进程退出、模型释放后启动。

使用训练相同的纯秒数prompt，14336预算、min64、fps2、maxframes0、贪心生成、repetition_penalty=1；测评max_new_tokens=1024。GT只参与评分，不进入模型输入。失败样本计零，不从分母移除。结果放在：

```text
Lkmllm_data/outputs/eval_results/<训练目录名>_eval14336_dp2/
  Qvhighlights/
  Charades_sta/
  Activitynet/
```

三个测试集的annotation和视频路径可在sh中修改，或设置 `QV_ANNO/QV_VIDEOS`、`CHARADES_ANNO/CHARADES_VIDEOS`、`ACTIVITYNET_ANNO/ACTIVITYNET_VIDEOS`。脚本在训练前检查路径，防止训练完成后才发现缺少测试集。`OUT`指定新训练目录，`EVAL_OUT`指定新测评根目录。

### 冒烟与验证

较小目标测试（每轮10条，共20条；完成后仍会跑三套完整测评）：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash Lkmllm_code/scripts/run_head_opd_2gpu.sh \
  --max-samples 10 --save-every-steps 1
```

只测试训练可直接调用Python入口并使用同一组训练参数；多卡自动分组能力仍保留在Python入口中。快照仅保存权重，并非完整optimizer/RNG恢复。不要复用已有非空输出目录。

本地CPU检查涵盖两轮配额、拒绝重试、不重复、有效batch与1000步计数，以及实际Bash流水线顺序、训练失败/未合并时阻止测评、缺少测试集时提前失败。未进行真实NVIDIA训练、显存或速度验证。

```bash
cd Lkmllm_code
python -m unittest discover -s scripts -p 'test_*.py' -v
```
