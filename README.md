# Qwen3-8B / 冻结32B：Top10 独立边界门控 OPD

2026-11-07：将启动脚本使用的 token-local 模式从 Top1 升级为 **Top10 概率加权边界误差门控**，并移除假设替换的 `start < end` 约束。该版本与旧 Top1 训练属于不同协议，使用新输出目录；历史测评结果不归属于本次更新。

2026-11-09：默认RKL，GT时长四桶 **2∶3∶4∶1**；每轮2000次有效训练、连续两轮。自动按每两张GPU一组同步训练，支持完整状态恢复。旧实验成绩不归入新版本。

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


## 自动检测卡数：两卡一组

从仓库根目录执行，使用当前CUDA环境的python（可通过PY覆盖）：

```bash
LK_OPD_ROOT=/fs0/AI/sx2624011/LKOPD/LK_OPD \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash Lkmllm_code/scripts/run_head_opd_8gpu.sh
```

历史脚本名保留；run_head_opd_2gpu.sh也是同一入口。不设置CUDA_VISIBLE_DEVICES时检测全部可见NVIDIA GPU。2/4/6/8卡分别形成1/2/3/4个工作组；奇数卡最后一张闲置。四卡处理两部分不同样本，经CPU Gloo同步梯度，得到同一份学生，不是两套独立权重。全局有效batch始终为4，不乘工作组数。

学生是Qwen3-VL-8B-Instruct，独立冻结教师是Video-opd-teacher-Qwen3-VL-32B-GRPO。每组学生位于本组第一卡，32B教师跨组内两卡分片。支持卡数不代表显存一定足够；加载预检禁止CPU/disk教师回退。CPU激活卸载默认开启，CPU_OFFLOAD=0可关闭。

资产目录从代码向父目录发现，或用LK_OPD_ROOT指定。STUDENT/TEACHER/ANNO/VIDEOS/OUT可覆盖sh中默认路径。仅支持单节点。

## GT时长四桶：2∶3∶4∶1

按GT end-start分桶，不是视频长度或事件发生位置。

| GT持续长度 | 每轮2000次有效训练 | 两轮合计 |
|---|---:|---:|
| ≤5秒 | 400 | 800 |
| 5秒<长度≤15秒 | 600 | 1200 |
| 15秒<长度≤60秒 | 800 | 1600 |
| >60秒 | 200 | 400 |

按剩余有效配额随机选桶；门控拒绝不计数，同桶取下一候选。候选预先分桶，无需每步扫描全标注。两轮保留同一个学生、Head-LoRA、AdamW；教师持续冻结，达到4000次目标时1000次参数更新。Top10门控和全词表RKL不变。

此前有效池中GT>60秒仅367条。**仅该桶允许耗尽后洗牌循环采样**，其他桶不重复。两轮400次有效训练及门控拒绝可能需要重用事件；不能写成4000条独立样本。启动前记录实际桶大小；summary.duration_sampling包含accepted_exposures、unique_accepted、repeated_accepted、每轮配额及尝试数。空桶或不可重复桶耗尽会报错，不静默改变配额。

## 中断后继续：使用同一OUT

首次运行建议明确指定OUT：

```bash
OUT=/path/to/Lkmllm_data/checkpoints/my_opd_2341 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash Lkmllm_code/scripts/run_head_opd_8gpu.sh
```

中断后执行：

```bash
OUT=/path/to/Lkmllm_data/checkpoints/my_opd_2341 \
RESUME=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash Lkmllm_code/scripts/run_head_opd_8gpu.sh
```

同一OUT存在resume.pt时也自动恢复。RESUME=1必须指定OUT。恢复要求数据、代码、教师、学习率、配额、轮数、全局batch及工作组数不变，协议不匹配直接报错。排他锁拒绝同一OUT同时运行两套训练。

每一批并行候选处理完成后，原子提交一份滚动resume.pt，保存适配器、AdamW、各工作组待累积梯度、四桶顺序/游标/轮次、接受记录、门控计数、Python/NumPy/PyTorch/各GPU随机状态，以及日志提交偏移。重启从最近一次完整提交继续，未提交的批次可能重算；日志回退到同一提交点，避免重复计数。未提交导出目录改名保留，避免重放时目录冲突。

首份resume.pt在模型加载完成后生成，此前没有已完成训练进度。旧adapter.pt没有优化器/RNG/采样状态，无法当作完整恢复点。加载旧权重开启新阶段与无损恢复是不同实验。请固定运行环境和代码版本，不要在训练中git pull后要求旧状态通过校验。不承诺非确定GPU算子跨运行逐bit一致。

## 自动接双卡分样本测评

完成合并后依次测QVHighlights、Charades-STA、ActivityNet。无论训练用四卡或八卡，评测默认只用前两张可见卡（EVAL_DEVICES可指定），各卡加载同一完整学生并分配不同样本。训练进程退出后才启动测评。

同一OUT已训练并合并完成时，sh跳过训练并继续测评。评测--resume保存已完成记录（失败也保留计零），不重复写入汇总。模型/标注/提示不匹配则拒绝续测；不会自动重试失败样本。

使用纯秒数提示、14336/min64/fps2/maxframes0、贪心生成、repetition_penalty=1、输出最多1024token。GT只用于评分。三个测试集路径可在sh中修改，路径缺失在训练前报错。结果位于Lkmllm_data/outputs/eval_results/<训练名>_eval14336_dp2/，EVAL_OUT可覆盖。

## 验证范围

```bash
cd Lkmllm_code
python -m unittest discover -s scripts -p 'test_*.py' -v
```

CPU测试覆盖自动分组、全局batch、四桶比例、稀缺桶重复、状态序列恢复、原子提交中断、日志回退、四卡训练与双卡测评的环境隔离。PyTorch可用时另测真实AdamW与待累积梯度恢复的数值一致性。本地无PyTorch，相关数值检查跳过；尚未进行实际四卡训练、断电恢复或速度验证。
