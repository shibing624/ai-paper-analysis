---
title: 短到长蒸馏拯救剪枝LLM：ShortOPD如何让"复读机"重获生成能力
date: 2026-07-17
arxiv: 2607.13124
tags: [LLM压缩, 结构化剪枝, On-Policy Distillation, 训练效率, 复读问题]
---

# 短到长蒸馏拯救剪枝LLM：ShortOPD如何让"复读机"重获生成能力

你有没有过这种体验：跑通了一个看起来很优雅的结构化剪枝方案，PPL 涨了一点点，MMLU 几乎不掉，然后兴冲冲拿去做开放式生成——结果模型开始无限循环同一句话。

"好的好的好的好的好的好的好的。"

停不下来，generation quality 雪崩。这是当下结构化剪枝落地最大的暗坑：剪枝后的模型在选择题 benchmark 上看着还行，但一进真实部署就开始复读。

今天这篇 [arXiv:2607.13124](https://arxiv.org/abs/2607.13124) 想做的，就是把这个复读机救回来。

## 一句话看明白

**ShortOPD** 是 ByteDance 和中科院软件所合作的工作：先用 Block-Influence (BI) 把 Qwen3-4B-Instruct 砍掉 4 层（25% 参数），然后用**剪枝前的原始模型当冻结 teacher**，让剪枝后的 student 在**自己采样出的状态**上做 on-policy distillation。重点是，他们发现固定长 rollout 在恢复初期大量预算都被复读后缀烧掉了，于是搞了一套**short-to-long 调度器**：检测到 teacher 都跟着复读的尾巴就提前掐掉，把后续 rollout 预算逐渐扩展到 student 真正能用的有效长度。

效果：在 8 个数学/代码/开放式任务上把剪枝模型从 5.7 分拉回 **48.5 分**（teacher 是 75.2 分），恢复 9×；比 KD（30.5）高 18 分，比 SeqKD（28.6）高 20 分；用固定 8192 token 预算跑要 35.9 小时，他们用 8.5 小时就跑完了，仅差 1.7 分。

---

## 论文信息

| 项 | 内容 |
|---|---|
| **标题** | ShortOPD: Recovering Pruned LLMs with Short-to-Long On-Policy Distillation |
| **作者** | Qingyu Zhang, Qianhao Yuan, Hongyu Lin, Yaojie Lu, Xianpei Han, Le Sun, Xiang Li, Ming Xu, Jiarui Li, Xiuyin Zhao |
| **机构** | ByteDance; Chinese Information Processing Laboratory, Institute of Software, CAS; University of CAS |
| **日期** | 2026-07-14 |
| **链接** | https://arxiv.org/abs/2607.13124 |

---

## 为什么这件事值得做

结构化剪枝（structured pruning）一直是 LLM 部署端最被看好的压缩路径。它和非结构化稀疏不一样——后者把权重矩阵打成马蜂窝，理论上能砍 50% 但实际推理必须配专用 kernel；结构化剪枝直接整层整头砍掉，模型跑起来还是普通 dense transformer，GPU 友好、推理延迟实打实地降。

但问题是，过去几年结构化剪枝论文几乎都在两类 benchmark 上自证清白：语言建模的 PPL，和选择题类的 MMLU/HellaSwag/ARC。ShortGPT、LLM-Pruner、SliceGPT、FLAP、Shortened LLaMA、Minitron 一路下来，PPL 涨 0.1、MC 掉 2 分，数据漂亮得不像话。

可这些指标是 recognition 任务。生产环境里没人用模型做选择题，部署要看的是**自由生成**。作者做了个简单但杀伤力极大的实验：把 Qwen3-4B-Instruct 砍掉 36 层中的 4 层（约 25% 参数），然后看不同任务上的生成表现。

| Domain | Benchmark | Pruned greedy | Teacher greedy |
|---|---|---|---|
| Math | GSM8K | 4.9 | 88.1 |
| Math | MATH-500 | 17.7 | 89.4 |
| Code | HumanEval | 2.7 | 75.0 |
| Code | MBPP | 6.6 | 80.5 |
| Open | Alpaca | 2.9 | 5.7 |
| Open | QA | 0.33 | 4.38 |
| Open | Sum | 3.36 | 8.86 |
| Open | MT-Bench | 2.19 | 6.95 |

> 表 1 摘要：剪枝后各任务的 greedy pass@1 几乎归零，但 teacher greedy 分数都正常

从能用到基本不能用，**就只砍了 4 层**。

这就是 recognition 和 generation 之间那道一直被遮起来的 gap。作者说这是结构化剪枝"难落地"的根本原因——光在选择题上不掉分没意义，模型在生成时不会因为你会做选择题就不复读。

---

## 两个关键观察：能力没被抹掉，是被降级了

作者先抛了两个观察来解释这个 gap 的本质。

### 观察 1：pass@1 没了，但 pass@k 还在

被剪枝的模型在 greedy 解码下几乎答不对题。但当你给它多采几次样（best-of-k），分数会迅速涨回来。比如 GSM8K：k=1 是 4.9，k=64 涨到 9.1；HumanEval：k=1 是 2.7，k=64 涨到 20.1。**正确的轨迹其实还藏在压缩模型的采样分布里，只是被其他高概率但错误的路径压住了。**

更夸张的是 GSM8K 上一组我都没想到的数据：k=64 时模型能拿到 91.2%，**比未剪枝 teacher 的 greedy 88.1% 还高**。

这个发现直接推翻了"剪枝=能力被破坏"这个直觉。能力没被擦掉，只是被降级——就像一个会解微积分的人喝了酒，反应迟钝做错题，但他其实还会。

### 观察 2：恢复失败主要通过后缀复读

那为什么 pass@1 这么低？复读。

![图：随剪枝深度变化的后缀复读率](https://arxiv.org/html/2607.13124v1/x2.png)

*图 1：Qwen3-4B-Instruct 在不同剪枝深度下的 suffix-repetitive rollouts 占比。0 层约 10%，9 层（25% 压缩点）飙到 84%，论文标为 critical point。*

作者在 25% 剪枝点上观察到：

- **复读后缀 token 的平均 teacher-student JSD 只有 0.0014**，普通 token 是 0.051，差了大约 35 倍
- 复读 token 的 teacher NLL 几乎是 0（3×10⁻⁵），普通 token 是 0.68
- 固定 H=2048 跑前 100 步，**55%–75% 的 rollout 是以后缀循环结束的**

复读区域的 teacher-student 分布几乎完全一致，蒸馏梯度接近零。这就是为什么传统的恢复方法在剪枝模型上特别难——它能在最该发力的地方几乎使不上劲。

![图：训练早期浪费的 warm-up 期](https://arxiv.org/html/2607.13124v1/x3.png)

*图 2：固定 2048 预算下，前 100 步的训练动态。红色区是 wasted warm-up：蒸馏 loss 趴底但复读率从 60% 慢慢降，80 步之后才进入能涨 loss 的状态。*

直觉上 recovery 早期模型还没学会好好"说话"，长 rollout 几乎全部预算都花在复读尾巴上，真正能学到东西的前缀只占一小部分。**预算没花在刀刃上**。

---

## 方法：On-Policy Distillation + Short-to-Long 调度

### 底层框架：OPD 的本质

On-Policy Distillation (OPD) 这套范式这两年很火，逻辑也直白：

- Teacher = 剪枝前的原始冻结模型（不是更大的外部 teacher）
- Student = 剪枝后的 student 实时采样出 token 序列
- 训练信号 = 每个 token 位置上，teacher 分布和 student 分布的散度

形式化目标函数：

$$\mathcal{L}_{\mathrm{OPD}}(\theta) = \mathbb{E}_x \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} \left[\frac{1}{|y|} \sum_{t=1}^{|y|} D_\alpha\big(\pi_T(\cdot|x, y_{<t}), \pi_\theta(\cdot|x, y_{<t})\big)\right]$$

具体到 ShortOPD：

- 散度 $D_\alpha$ 选**广义 JSD**（α=0.5），这是 forward-KL 和 reverse-KL 的有界折中，避开 log-ratio 极端值
- 只蒸馏 top-100 logits + tail 聚合的分布，clip 到 2.0
- Teacher 完全冻结，**不需要标签、不需要 verifier、不需要外部 teacher**
- 一个 epoch 710 步，rollout group=8，train batch=64，rollout 温度 0.8

跟 SeqKD 比，OPD 训练信号是 student 自己的采样状态，跟 KD 比，不需要 ground-truth 回答。四个属性一起满足（on-policy + dense + label-free + verifier-free）这件事，作者专门用一张表对比了：

| 方法 | On-policy | Dense signal | Label-free | Verifier-free |
|---|:---:|:---:|:---:|:---:|
| SFT w/o KD | × | × | × | ✓ |
| SeqKD | × | × | ✓ | ✓ |
| KD | × | ✓ | × | ✓ |
| RLVR (PPO/GRPO) | ✓ | × | × | × |
| **ShortOPD** | **✓** | **✓** | **✓** | **✓** |

> 表 2：不同恢复方法的属性对比，ShortOPD 是唯一四项齐全的

### 关键设计：Short-to-Long 调度器

整个 ShortOPD 的真正贡献，就是**在 OPD 框架上加了一个动态 rollout 预算调度器**。

核心问题：固定长 rollout 浪费在复读尾巴上；固定短 rollout 又会截断后续合法的长生成。

解决方案：保持全局 response length 4096 作为静态 padding 和上下文上限，但设一个**逐步动态**的采样预算 $H_s$ ∈ [1024, 2048]。

![图：ShortOPD 整体框架](https://arxiv.org/html/2607.13124v1/x4.png)

*图 3：ShortOPD 流程图。Compress → Sample on-policy rollouts (Budget H) → Frozen teacher 给出 top-K JSD 蒸馏信号 → Suffix-Repetition Probe → Rollout Statistics (EMA) → Controller 给出 Next Budget H。复读率高就 Shrink H，低+截断率高就 Grow H。*

整个控制器每步观测三个 EMA 统计量（衰减 0.7）：

- $r_s$：以严重周期循环结束的 rollout 比例（复读率）
- $q_s$：无循环且填满当前预算的 rollout 比例（clean truncation rate）
- $L_s$：检测到循环前的平均可用前缀长度（effective length）

控制规则：

$$H_s^* = \begin{cases} \min(H_s, \lambda \bar{L}_s), & \bar{r}_s > \rho_{\mathrm{high}} \\ \gamma_\uparrow H_s, & \bar{r}_s < \rho_{\mathrm{low}} \land \bar{q}_s > \tau \\ H_s, & \text{otherwise} \end{cases}$$

超参：$\rho_{\mathrm{low}}=0.20$，$\rho_{\mathrm{high}}=0.45$，$\tau=0.10$，$\lambda=1.15$，$\gamma_\uparrow=1.25$，$\beta_H=0.7$。预算舍入到 16 的倍数。

作者特意强调两个设计选择：

1. **shrink 优先于 grow**。复读发生时立刻缩，不会等；增长需要同时满足复读低和当前 horizon 有约束力。
2. **loss 信号只精修边界，不决定 a_i**。Effective length 的结构起点由独立的复读模式探测给出，loss 只用来在结构起点附近找第一个 loss-confirmed 边界——这避免了 loss 信号同时被用作"判定复读"和"细化边界"两个角色时的不稳定。

这个分工会让人想起 PPO 里的 value function 和 reward shaping 也是分开的，作者显然吸取了这种思路。

### 复读探测器（Algorithm 2）

具体怎么判定一个尾巴是"复读"？只在 rollout 末尾 W=512 个有效 token 里找周期 p ∈ {1, ..., 10}，要求：

- Terminal anchor A=32，agreement 阈值 η=0.90
- 最小循环数 C=3，最小 tail L_min=64
- Severe loop：tail ≥128 tokens 或占 suffix 的 30%

用 shifted comparison，不要求 rollout 在循环边界结束。

---

## 实验：数字说话

### 主表：8 个任务上的恢复

| Model | Method | GSM8K | MATH | HumanEval | MBPP | Alpaca | QA | Sum | MT-B | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|
| Dense teacher | — | 88.10 | 89.40 | 75.00 | 80.54 | 6.64 | 4.38 | 8.86 | 6.95 | 75.17 |
| Pruned (-25%) | untrained | 1.14 | 1.60 | 0.00 | 0.00 | 1.14 | 1.00 | 1.06 | 1.09 | 5.71 |
| | SFT w/o KD | 25.93 | 9.40 | 26.83 | 29.18 | 1.91 | 1.19 | 3.12 | 1.59 | 21.19 |
| | SeqKD | 32.83 | 12.20 | 29.88 | 41.25 | 2.52 | 1.38 | 5.26 | 2.10 | 28.60 |
| | KD | 29.34 | 9.60 | 29.27 | 39.30 | 2.94 | 1.48 | 6.89 | 2.36 | 30.52 |
| | **ShortOPD (1 ep)** | **62.70** | **42.00** | **43.90** | **49.81** | **4.68** | **2.04** | **7.94** | **4.28** | **48.46** |
| | ShortOPD (2 ep) | 70.66 | 50.60 | 51.83 | 56.42 | 4.98 | 2.20 | 8.16 | 4.47 | 53.45 |
| | ShortOPD (3 ep) | 72.71 | 55.20 | 54.27 | 57.20 | 5.14 | 2.21 | 8.15 | 4.89 | 55.41 |

> 表 3：主实验，1 个 epoch 后 ShortOPD Avg 48.46，比最佳 off-policy baseline KD 高 17.94 分

数字看一眼就有感觉：剪枝后 Avg 从 75.17 砸到 5.71，ShortOPD 1 epoch 拉回 48.46（约未恢复值的 9×），3 epoch 还能再涨到 55.41（teacher 的 73.7%）。GSM8K 单项 3 epoch 后 72.71，距离 teacher 88.1 只差 15 分。

![图：主实验结果柱状图](https://arxiv.org/html/2607.13124v1/x1.png)

*图 4：4 种方法在 Math、Code、Open-ended、Avg 四个域上的 generation score 对比。橙色是 ShortOPD，每一项都远高于其他方法。*

### 短 vs 长的代价

主表说服力够强，但更狠的是**效率数据**。

| Schedule | Avg | Rollout tokens | Wall-clock |
|---|---|---|---|
| Vanilla OPD (fixed H=2048) | 49.6 | 337M | 11.1h |
| **ShortOPD (init 2048)** | **48.5** | **250M** | **8.5h** |
| Vanilla OPD (fixed H=8192) | 50.2 | 869M | 35.9h |

ShortOPD 用 1/4 训练时间、71% 更少 rollout token，**结果比固定 8192 只差 1.7 分**。

![图：效率对比](https://arxiv.org/html/2607.13124v1/x8.png)

*图 5：三种调度在 generated tokens 和 wall-clock 上的对比。ShortOPD 8.5h，fixed-2048 11.1h，fixed-8192 35.9h。Avg 几乎都在 50 附近。*

![图：调度动态对比](https://arxiv.org/html/2607.13124v1/x5.png)

*图 6：左图是 Vanilla OPD（蓝）和 ShortOPD（橙）的训练动态。右图是 wasted tokens 和 rollout budget 的演化：橙色虚线显示 ShortOPD 预算在高复读期从 2048 降到 1024，clean truncation 主导后回升到 2048，呈现 shrink → hold short → grow → hold long 的阶段切换。*

![图：每步生成时间](https://arxiv.org/html/2607.13124v1/x6.png)

*图 7：rollout 生成时间（秒/步）。ShortOPD 平均 25.1s/step，比 Vanilla 35.4s/step 低 29%。*

具体看训练动态：前 100 步，ShortOPD 的 wasted warm-up 期从 vanilla 的 ~80 步压缩到 ~40-50 步，预算从 2048 降到 1024，等复读率降下来再涨回去。

### RLVR 为什么不行

作者特意把 RLVR（PPO 和 GRPO）也拿来对比了——这是当下最火的后训练范式。

![图：GSM8K-only 对比 RLVR](https://arxiv.org/html/2607.13124v1/x9.png)

*图 8：GSM8K-only 训练后，各方法在 GSM8K 上的准确率。Pruned 1.1，PPO 0.23，GRPO 1.6，ShortOPD (GSM8K-only) 37.8，ShortOPD multi 62.7。*

PPO 0.23，GRPO 1.6，**比直接用剪枝模型还低**。ShortOPD 即使只在 GSM8K 上训，也能拿 37.8。

原因前面已经埋下了：剪枝模型在 GSM8K 上几乎采不出正确答案，RLVR 的稀疏奖励信号几乎全部为零，**冷启动问题在这里被放大到极致**。这也是清华同期那篇 OPD 论文（arXiv:2604.13016）反复强调的——"thinking-pattern 不一致"和"高分 teacher ≠ 新知识"在 RLVR 场景下同样成立。

### 域消融：恢复语料不是越多越好

作者把 45,447 个 prompt 按域（math/code/general）做消融。

| 恢复语料 | GSM8K | MATH | HumanEval | MBPP | Alpaca | QA | Sum | MT-B | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Full mixture | 62.70 | 42.00 | 43.90 | 49.81 | 4.68 | 2.04 | 7.94 | 4.28 | 48.46 |
| w/o code | 58.45 | 33.40 | **0.61** | **2.33** | 4.47 | 2.15 | 7.88 | 3.99 | 34.96 |
| w/o math | **8.04** | **6.00** | 28.66 | 43.19 | 3.96 | 1.73 | 7.19 | 2.94 | 30.51 |
| w/o general | 54.21 | 28.80 | 48.17 | 50.19 | 3.01 | 1.51 | 5.41 | 3.15 | 39.02 |

去掉 code → HumanEval 掉到 0.61，MBPP 掉到 2.33。
去掉 math → GSM8K 掉到 8.04，MATH 掉到 6.00。
去掉 general → open-ended 全面下滑，但 code 略升。

> 表 4：恢复语料库的领域覆盖是一等设计选择，不是细节。on-policy distillation 只修复 student 自己的搜索能访问到的状态

这跟很多人对"恢复语料"的认知不一样——大家倾向于觉得多就是好，但这儿恰恰证明**领域覆盖必须和评测任务对齐**。这个观察其实跟"对 student 真正在搜的分布做监督"是同一个逻辑。

### Multiple-Choice 反转

最有意思的一个数字是 multiple-choice 上的对比：

| Method | ARC-C | HellaSwag | MMLU | WinoG. | Avg |
|---|---|---|---|---|---|
| Dense teacher | 89.93 | 80.60 | 70.93 | 65.19 | 76.66 |
| SFT w/o KD | 89.33 | 76.89 | 69.09 | 64.09 | 74.85 |
| KD | 87.71 | 77.13 | 67.59 | 65.04 | 74.37 |
| ShortOPD (1 ep) | 79.52 | 68.89 | 60.70 | 61.33 | 67.61 |
| ShortOPD (3 ep) | 84.47 | 75.59 | 64.59 | 61.72 | 71.59 |
| **SFT-init ShortOPD** | **89.93** | **79.67** | **69.99** | **67.64** | **76.81** |

> 表 5：MC 任务的 mirror image——SFT/KD 在 off-policy teacher-forced MC 上最优（74.9/74.4），但在生成上远落后

MC 评分模式和生成恢复**呈镜像关系**：SFT/KD 训在 off-policy teacher-forced MC 上分数最高（74.9/74.4），但在生成上反而是 ShortOPD 远远领先。ShortOPD 1 epoch MC 67.6，3 epoch 涨到 71.6，**而 SFT-init ShortOPD（先用 SFT 初始化再跑 ShortOPD）能拿 76.81，匹配 teacher 的 76.66**。

这是全文最值得记住的发现之一：**recognition 和 generation 沿部分独立的轴恢复，两者互补而非冲突**。你不能用一个指标代替另一个。

---

## 我的判断

读完这篇论文我最大的感受是：ShortOPD 不是那种让人"哇这想法真巧"的工作，它是一个**对工程痛点想得很透、解决方案很扎实**的工程型工作。

### 真正值钱的地方

1. **指出了"recognition ≠ generation"这个被结构化剪枝领域长期忽视的问题**。过去几年这一领域几乎全在 MC 榜单上自嗨，剪完 25% 还能在 HellaSwag 拿 75 分，然后生产环境里复读给你看。ShortOPD 用一组简单的 pass@1 表格就把这个 gap 摆到台面上。Table 5 那个 MC mirror image 我觉得是全文最该被引用的发现——它直接告诉做剪枝的人：你的 MC 分数不能说明任何事。

2. **将"复读"作为可测量的现象**。复读不是玄学，他们用 JSD 0.0014 vs 0.051 把这个差距定量化了，然后用 84% 的复读率定位 critical point，最后用 shrunk rollout 预算解决。这意味着它可以被检测、被度量、被控制。

3. **短到长调度器本身简洁**。复读率高缩，截断率高+复读率低才涨，优先级排得很清楚。这个控制规则并不复杂，工程上很容易复现——我估计一个训练脚本 50 行内能加上。

4. **OPD 范式在剪枝恢复上的真正落地**。清华那篇 arXiv:2604.13016 之前把 OPD 失败模式扒得很细，但 ShortOPD 给出的是"在剪枝 student 自己的搜索分布上监督"这种恰好绕过失败模式的方法。

### 我也有点怀疑的地方

1. **单模型单剪枝比的实验设计**。Qwen3-4B-Instruct 砍 4 层得到 27 层，所有消融都在这一组上做。不同 size（7B/13B/70B）、不同架构（MoE）、不同剪枝比（30%、50%）的泛化性，作者没怎么碰。Table 7 给了更多模型尺寸的 ablation hint，但主表只有这一组。

2. **复读探测器的超参多且偏敏感**。W=512、A=32、η=0.90、C=3、L_min=64、ρ_low=0.20、ρ_high=0.45、τ=0.10、λ=1.15、γ_↑=1.25、β_H=0.7——光调度器就有 10 个以上超参。这些数字在 Qwen3-4B 上工作，换个模型可能就要重调。论文没有给出超参敏感性的系统性分析。

3. **"为什么是 α=0.5 的 JSD"凭直觉给**。附录里给了局部展开，梯度在 q=p 时为零，斜率是 forward-KL 的 1/4。但这个选择具体怎么影响复读区域的训练动态，论文没仔细展开。

4. **跟主流 LLM 蒸馏论文（OpenAI 蒸馏、DeepSeek R1 蒸馏、Qwen3 自身蒸馏）的对比缺失**。它的 OPD 配方看着是 VisionOPD 仓库的复现，但工业界蒸馏大模型基本都用更大的外部 teacher + 多阶段 SFT 起步，ShortOPD 用"自己当 teacher"这个角度虽然节省了 verifier 和 label，但在工业级生产里**能不能直接用**还不好说。

5. **架构 vs 范式的功劳分割不清晰**。Table 3 末尾 SFT-init ShortOPD 拿到了 MC 76.81，**比纯 ShortOPD 1 epoch 的 67.61 高 9 分**。这暗示 SFT 预训练是 MC 上的主力贡献，ShortOPD 主要是补生成侧的短板。到底多少提升来自 short-to-long，多少来自 OPD 本身？3 epoch 那个 71.59 相比 1 epoch 的 67.61 也只涨了 4 分，没有把变量分得很干净。

### 对工程实践的启发

如果你的团队也在做 LLM 部署端压缩，这篇论文给你三个可立刻用上的 takeaway：

1. **评估清单必须有开放式生成**。MC 和 PPL 不够，加 MT-Bench、AlpacaEval、HumanEval 这类。如果某次剪枝在 PPL 上 +0.05 但开放式生成里复读率从 5% 涨到 50%，那是不能上线的。

2. **OPD 是个被低估的剪枝恢复手段**。它不需要外部 teacher、不需要 label、不需要 verifier，工具链很轻。但跑前必须确认 teacher 和 student 在 thinking-pattern 上是一致的（清华那篇论文已经把这个先决条件讲清楚了）。

3. **训练 budget 调度本身是个独立的研究问题**。ShortOPD 的 short-to-long 控制器只针对复读，但同思路可以推广到其他浪费训练预算的现象上（探索-利用失衡、样本难度分布漂移等）。

---

## 写在最后

回到开头那个复读机问题。ShortOPD 的答案其实很简单——剪枝没有抹掉能力，剪枝是降级了采样分布；只要在 student 自己的分布上做密集监督，能力就能找回来。但找回来的过程中，**不该让 student 在复读尾巴上浪费 token**，于是需要一个知道什么时候该把 rollout 剪短的调度器。

这个思路很朴素，但配上扎实的工程实现，1/4 时间、71% 节省 token、9× 恢复幅度——这种数字拿出来就是给生产环境用的，不是给 benchmark 用的。

如果只能从这篇论文里带走一句话，我想是这个：

> **结构化剪枝的失败不是能力的消失，是采样的退化；恢复的过程必须贴着 student 真实的搜索分布走，别让训练预算白白烧在废 token 上。**

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我。*
