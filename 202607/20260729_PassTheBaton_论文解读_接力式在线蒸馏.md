---
title: 接力式在线蒸馏：把"半路崩掉"的学生轨迹从老师手里救回来
date: 2026-07-29
paper: Pass the Baton: Trajectory-Relayed On-Policy Distillation
arxiv: 2607.26057
authors: Haolei Xu, Xiaowen Xu, Haiwen Hong, Zixuan Ni, Hongxing Li, Yiwen Qiu, Weiming Lu, Yongliang Shen
affiliations: Zhejiang University; Yuvion Team, Alibaba Group
code: https://github.com/zju-real/Relay-OPD
project: https://zju-real.github.io/Relay-OPD
---

# 接力式在线蒸馏：把"半路崩掉"的学生轨迹从老师手里救回来

## 核心摘要

做 LLM 蒸馏的朋友，最近一年肯定被同一个问题反复折磨过：学生自己 roll out 的轨迹一旦早期走偏，后面整段推理就全废了，老师的 token 级监督信号也跟着失真。Qwen3 团队这篇论文（来自浙大 + 阿里 Yuvion）盯的就是这个**前缀失败（prefix failure）**。他们发现一个挺干净的现象：在失败的前缀上，老师和学生会出现一种"继续方向上的不对称"——老师倾向于停下来反思（"但是…"），学生则继续往错的路上冲。他们把这个不对称变成一个**零标签的交接触发器**，然后在检测到触发时让老师短暂接管一段（teacher leg），学生再接手生成。这种"接力式"的轨迹在 1.7B 学生上把标准 OPD 拉高了 5.73 个点，把更强的 FastOPD 也压下去了 1.49 个点，更关键的是**训练轨迹长度直接砍掉 50% 以上**。

工程上最让我觉得聪明的一点：整条接力 rollout 居然能塞进**一个 speculative decoding 引擎**里跑——学生当 draft，老师当 target，通过 state-switch 实现交接，没引入额外的调度开销。论文代码已开源，方向非常工程化。

## 论文信息

- **标题**：Pass the Baton: Trajectory-Relayed On-Policy Distillation
- **作者**：Haolei Xu, Xiaowen Xu, Haiwen Hong, Zixuan Ni, Hongxing Li, Yiwen Qiu, Weiming Lu, Yongliang Shen
- **机构**：Zhejiang University；Yuvion Team, Alibaba Group
- **arXiv**：2607.26057（2026-07-28）
- **代码**：https://github.com/zju-real/Relay-OPD
- **项目页**：https://zju-real.github.io/Relay-OPD

## 问题的来源：为什么 on-policy distillation 也救不了长链推理

先把背景摆出来。On-Policy Distillation（OPD）这两年从 Thinking Machines 那篇博客开始被工业界正式搬进 LLM post-training 流水线——Qwen3、MiMo-V2-Flash、GLM-5 都在用。它的精髓是：**学生自己 roll out，老师在学生实际走过的 prefix 上给稠密的 token 级监督**。这样比离线蒸馏（直接拿老师生成的文本做 SFT）少了一大截分布偏移，比纯 RL（GRPO、PPO）又稳定不少。

但凡是做过数学推理蒸馏的人，应该都撞过同一堵墙：**学生一旦早期推理方向走偏，后面的 token 就会沿着错误路径一路狂奔，老师的监督也变得越来越不靠谱。** 这就是 prefix failure。

![图1：Relay-OPD 的运行样例 + 与标准 OPD 的对比 + 主结果](https://arxiv.org/html/2607.26057v1/x1.png)

看图 1(a) 那个具体的例子特别直观——"苹果 3 RM、芒果 4 RM、木瓜 5 RM、花 50 RM 各买至少一个，最多能买几个水果"。

- 学生写到中间，已经买了 15 个苹果："还剩 5 RM，能买 1 个芒果和 1 个木瓜，刚好 17 个水果"——So (50.6%)，但它**忘了木瓜要 5 RM 已经不是 4+5=9 了**。
- 老师的 top-1 是 "But (74.4%)"，直接把反思 token 抛出来："等等，15 个苹果之后只剩 5 RM，但 1 芒果 + 1 木瓜是 9 RM，根本不够"。
- 老师接管一段，纠正上下文，然后学生 resume 回来自己写出正确答案 "12 + 1 + 2 = 15"。

注意这里老师并没有"把整条答案写完"，它只接管了一个**反射 token + 几个段落**（论文里平均 23.2 token），剩下的推理路径还是学生自己的。

### 现有方法的三种典型应对，以及为什么都不够

论文把现有做法归成三类，每一类都有结构性的硬伤：

1. **固定长度截断（ESR / FastOPD）**：到某个 token 数就强制截断，不看推理状态。简单粗暴，但**截断位置跟失败点没关系**，可能截到一半。
2. **离线重写（TRD）**：rollout 完了再让老师改写。问题是改写完的轨迹**经常带明显的重写痕迹**——不像正常解题过程。
3. **Token 级混合（SKD）**：根据分布分歧度动态切老师/学生，但**没有明确的"推理方向失败"信号**，只是 token 分布上的差异。

**这三种方法的共同盲点**：没有一个机制能在 prefix failure 发生**当时**根据 reasoning state 本身做判断和干预。

## Relay-OPD 的核心 idea

作者做了一件很干净的事：先在 128 个 DAPO-Math-17K 英文样本上做"轨迹干预实验"——人为在某些位置让老师接管，看**干预的时机和长度怎么影响效果**。

![图2：轨迹干预预实验](https://arxiv.org/html/2607.26057v1/x2.png)

![图2b：师生差距随位置变化](https://arxiv.org/html/2607.26057v1/x3.png)

两个发现直接决定了 Relay-OPD 的设计：

**发现 1：纠正可以是极度局部的。** L=0（只在触发点替换一个反射 token，老师 token 占比 0.35%）就能把准确率从 27.73% 拉到 34.96%，涨了 7.23 个点。L=3 之后曲线基本饱和，**再多 teacher token 收益也不大**。

**发现 2：干预要趁早。** 把 L 固定为 3 段，加一个延迟参数 r（跳过的有效触发数）——r=0（不延迟）41.99 分，r=1（延迟一次）直接掉到 33.98，r=3 进一步掉到 29.49。看图 2b 就更清楚：随着生成推进，老师和学生之间的 token 级 advantage 差距在迅速收窄，到 4K-8K 区间几乎重合。**越往后接管，老师越被学生的 context 牵着走，根本没机会 redirect。**

所以 Relay-OPD 的设计原则很自然：**早、局部、有预算**。

### 接力触发器（Handoff Trigger）

定义反射词集合 $\mathcal{R}$（Wait、But、However、Actually、Hold、… 以及它们的大小写、首空格变体，共 18 个左右）。在每个学生 prefix $h$ 上，定义触发条件：

$$
\phi(h) = \mathbf{1}\bigl[a^{T}(h) \in \mathcal{R}\bigr] \cdot \mathbf{1}\bigl[\mathcal{K}_S(h) \cap \mathcal{R} = \varnothing\bigr]
$$

用人话讲：**老师 top-1 想用一个反思词，而学生 top-K 里压根没有反思词**——这就是 reasoning direction 的分歧。$K$ 控制敏感度，论文设 $K=5$。

图 1(a) 里那个例子就是触发点：老师想要 "But"，学生 top-K 里只有 "So/Now/…"——但 $\mathcal{R}$ 里没有交集，触发条件成立。

### 接力轨迹的构造

设 relay budget $(M, L)$——最多 $M$ 次老师接管，每次老师生成 $L$ 个段落（按 `\n\n` 切分）。

- 学生按自己的策略生成，每步算 $\phi(h)$；
- 触发条件成立且 $j_t \lt M$ → 老师接管：以 trigger token 作为 leg 起点，再续 $L$ 段；
- 教师 leg 结束后如果预算还有 → 学生 resume；
- 第 $M$ 次 leg 结束 → 当前 rollout 终止。

L 用段落而不是 token 数——这一点细节挺重要，**每个 leg 在结构完整的推理单元处收尾**，不会断在半句话里。

### 优化目标

作者用 reverse-KL 风格的单样本 PPO 目标，advantage 就是 $A_t^{\text{Relay}} = \log \pi_T(z_t \mid h_t^z) - \log \pi_{\bar\theta}(z_t \mid h_t^z)$，对整条 relay trajectory $z$ 做 clip-PPO 更新。

注意：teacher leg 上的 token **也是 z 里的一部分，也直接参与 loss 计算**——这和后面的消融实验直接相关。

## 最让我觉得漂亮的工程实现：塞进一个 speculative decoding 引擎

我读到这里的第一反应是"这一段写得很克制，但是是全文最 engineering-savvy 的部分"。

如果按字面实现 Relay-OPD，需要：
- 维护两套生成管道（学生 + 老师）；
- 在触发点协调切换；
- 在 resume 点再切回来；
- 处理 schedule、logits 同步、KV cache 切换……

作者的解法是**全部塞进一个 speculative decoding 引擎**：

![图3：Relay-OPD 概览](https://arxiv.org/html/2607.26057v1/x4.png)

- **学生 = draft model，老师 = target model**；
- 每个 token 位置由一个 decoding state $s_t \in \{S, T, \perp\}$ 决定：学生 leg、老师 leg、还是终止；
- 状态机由 Eq. (10) 控制——S 状态下检测到 $\phi = 1$ 切到 T，T 状态下生成到 $L$ 段切回 S（或终止）；
- 学生 leg 上 $\pi^{\text{tgt}} = \pi_{\bar\theta}$，**draft 永远被接受**（相当于普通学生自回归生成）；
- 老师 leg 上做标准的 speculative rejection sampling against $\pi_T$；
- 老师 leg 的 leg-initial token（也就是 trigger token $a^T(h^z)$）**直接 emit，不做 verify**。

最妙的是这一句："teacher logits computed during verification simultaneously provide $a^T(h^z)$ and the trigger criterion $\phi(h^z)$ at no additional cost"——**触发器检测基本是免费的**，因为 verify 学生 draft 的时候本来就要算老师的 logits，从里面读出 top-1 和 top-K 交集判断即可。

训完一个 batch，把最新的学生权重同步到 draft model，老师全程冻结。整条 relay rollout 跑完和"两套管道来回切"在分布上严格等价（Leviathan et al. 的 speculative sampling 正确性直接保证），**但没有调度开销**。

这个工程上的小决定我认为是 Relay-OPD 能跑出 50%+ 训练长度下降的关键支撑——它把"在线干预"的开销压到了 speculative decoding 的开销水平。

## 实验结果

**设置**：老师 Qwen3-4B-Instruct-2507，学生 Qwen3-0.6B / 1.7B-Non-Thinking，训练数据 DAPO-Math-17K 英文子集，8 个数学 benchmark（AIME 24/25/26、MATH500、AMC23、OlympiadBench、HMMT Feb26、HMMT Nov25），8 张 H100 训 1 epoch，温度 1.0，top-p 1.0。

### 主实验表（Table 1）

**1.7B 学生**：

| 方法 | AIME24 | AIME25 | AIME26 | MATH | AMC23 | Olymp. | HMMT Feb26 | HMMT Nov25 | Avg | Train Len |
|------|--------|--------|--------|------|-------|--------|------------|------------|------|-----------|
| Student | 12.60 | 9.58 | 7.40 | 71.95 | 47.89 | 38.54 | 6.34 | 4.38 | 24.84 | — |
| SFT | 23.33 | 19.48 | 16.15 | 81.40 | 59.45 | 46.62 | 12.59 | 6.56 | 33.20 | 4262 |
| KD | 23.54 | 21.15 | 15.31 | 81.45 | 60.23 | 48.07 | 12.78 | 7.50 | 33.75 | 4262 |
| GRPO | 24.58 | 22.08 | 15.62 | 80.35 | 60.16 | 48.74 | 14.49 | 9.38 | 34.42 | 2558 |
| OPD | 35.83 | 25.52 | 23.33 | 85.70 | 70.08 | 55.27 | 20.08 | 14.06 | 41.23 | 4658 |
| TRD | 19.27 | 19.69 | 12.71 | 77.70 | 55.47 | 44.18 | 11.93 | 4.58 | 30.69 | 2785 |
| FastOPD | 42.29 | 30.42 | 26.35 | 87.95 | 74.30 | 58.16 | 23.58 | 20.73 | 45.47 | 2709 |
| SKD | 33.12 | 30.73 | 28.85 | 87.35 | 72.42 | 54.41 | 20.08 | 11.88 | 42.35 | 4753 |
| **Relay-OPD** | **42.71** | **32.81** | **30.52** | **89.50** | **76.88** | **58.79** | 24.72 | 19.79 | **46.96** | 2296 |
| Δ vs OPD | +6.88 | +7.29 | +7.19 | +3.80 | +6.80 | +3.52 | +4.64 | +5.73 | **涨 5.73 个点** | **降 50.7 个点** |

**0.6B 学生**（节选关键行）：

| 方法 | Avg | Δ vs OPD | Train Len |
|------|------|----------|-----------|
| Student | 11.80 | — | — |
| OPD | 28.03 | — | 6900 |
| FastOPD | 30.42 | +2.39 | 3302 |
| **Relay-OPD** | **31.04** | **涨 3.01 个点** | 2490 |

几个关键观察：

- **1.7B 平均 46.96**，8 个 benchmark 上 7 个第一、1 个第二（HMMT Feb26 24.72 略低于 FastOPD 的 23.58——等等，FastOPD 是 23.58，Relay-OPD 24.72 还更高，我看错了，**实际是 8 个 benchmark 上 Relay-OPD 全部最佳**）；
- 0.6B 也全部第一/第二，趋势一致；
- 训练 token 效率：1.7B 从 OPD 的 4658 token → 2296 token，**降 50.7 个点**；0.6B 从 6900 → 2490，**降 63.9 个点**；
- **最佳 checkpoint 步数也明显前移**——1.7B 的 Relay-OPD 在 step 35 就到最优，OPD 要 step 55，FastOPD 要 step 45。

### Pass@k（图 4）

![图4：Pass@k 性能](https://arxiv.org/html/2607.26057v1/x5.png)

在 HMMT Feb26 / Nov25 上，Relay-OPD 的 Pass@8/16/32/64/128 全部高于 OPD。Pass@128 时 Feb26 上 Relay-OPD 51.4 vs OPD 45.6，Nov25 上 66.5 vs 60.1。**这说明效果提升不是因为采样变窄了**——多样性、覆盖度都涨了。

### 推理长度（图 5）

![图5：推理响应长度对比](https://arxiv.org/html/2607.26057v1/x6.png)

AIME25 上从 18.1k token 砍到 14.9k（-17.9%），AIME26 砍 14.2%，HMMT Feb26 砍 28.3%。**学生学到了"在错的路上及时回头"的能力**，而不是单纯抄到更短的答案。

### 训练动态（图 6）

![图6：训练动态](https://arxiv.org/html/2607.26057v1/x7.png)

- **左**：耗尽 relay budget 的轨迹比例从 75-85% 降到 50-60%——学生越来越不需要老师帮忙；
- **中**：teacher token 占比从 ~13% 跌到 2-3%——同理；
- **右**：策略熵 Relay-OPD 始终高于 OPD 和 FastOPD——老师干预增加了学生探索。

这三点串起来一句话：**学生不是被"教成老师"，而是被"教会自己跳出来"，这点挺关键的**。

## 消融实验（Table 2）

1.7B 学生上，关键的两个消融：

**教师腿的价值**（vs 直接在 trigger 点终止）：

| 变体 | Avg |
|------|------|
| Trigger-Stop（M=1，不生成 teacher leg） | 43.48 |
| Relay-OPD（M=1, L=3） | **46.25** (+2.77) |

**教师腿上的训练目标**：

| 变体 | Avg |
|------|------|
| Student Draft Token（用学生 draft 的 token 做 advantage） | 44.56 |
| Teacher FKL（k=128，正向 KL 全 teacher 分布） | 44.08 |
| **Relay Token（用实际生成的 teacher token 做 reverse-KL）** | **46.96** |

两个消融都讲得通：

- 教师腿不只是"截断时机优化"，它**真的提供了校正过的上下文 + 局部的推理示范**；
- reverse-KL 风格的单样本目标**比 forward-KL 更 mode-seeking**，让学生选择性地吸收老师的纠正信号；而不是像 forward-KL 那样被迫匹配老师完整分布，把不可靠的监督也吞下去。

## 超参敏感性（图 7）

![图7：超参敏感性](https://arxiv.org/html/2607.26057v1/x8.png)

- **L 段数**：0 → 1 → 2 → 3 → 4 → 5，准确率 44.31 → 45.56 → 45.81 → 46.96 → 47.10 → 46.47。**L=3/4 最好**，L=5 略降；
- **M 次数**：1/2/3/4 = 46.25 / 46.96 / 45.94 / 44.01，**M=2 最优**；
- **handoff top-K K**：K=1 / 5 / 10 / |V| = 44.27 / 46.96 / 43.14 / 41.23。K=5 最佳，K=1 触发太敏感，K=10 漏检严重。

**主干配置：$K=5$, $(M, L) = (2, 3)$，8 张 H100 训 1 epoch。**

## 我的判断

把"工程整合型创新"这顶帽子扣给这篇论文，idea 的零件其实都在文献里这个判断我认。

几个让我印象深的点：

1. **触发器定义是真的简洁**。18 个反射词 + top-K 交集判定，不依赖任何 verifier、process label、reward model，**完全 label-free**。这意味着它可以无缝嫁接到任何现有 OPD pipeline 上，不需要额外的标注数据。
2. **Speculative decoding 引擎复用**这一手很值得抄作业。我之前看到的 OPD 工程实现基本都是"学生自己 roll → 老师离线打分"，Relay-OPD 这个 schema 至少从 paper 看是第一个把教师干预 inline 进去的——而且从 Eq.(10) 的状态机描述看，控制逻辑很轻。
3. **训练 token 砍 50% 这个数其实比 5.73 个点的提升更有杀伤力**。在工业界，蒸馏的训练算力成本是按 token × FLOPs 算的，这个数字意味着你可以在同样的预算内训更多次或者训更多学生。

**我持保留意见的地方**：

- 评估的 8 个 benchmark 全部是数学推理，**没有覆盖 code / agent / general chat**。反射词集合 $\mathcal{R}$ 是为推理任务精心构造的（"Wait"、"Actually"、"However"），迁移到其他领域时这个 trigger 还能不能 work 是问号。论文没给这部分实验。
- 标准 OPD 的 1.7B 41.23 这个 baseline 跟同期的 MiMo、DeepSeek 报告的数比偏低，**可能是 DAPO-Math-17K 英文子集 + 1 epoch 这个相对轻量的训练配置导致的**——所以 Relay-OPD 的 5.73 个点提升，是在"轻量配置"下打出来的，重训练配置下增益可能更小。
- K=5 这个 magic number 没在更多学生尺寸上验证（0.6B 和 1.7B 都用 K=5），不同模型 / 不同 tokenizer 可能需要重选。

**和同期工作的位置**：和 2603.25562（CASIA 那边提的 "teacher top-K local support matching"）是 2026 年 OPD 圈子两个比较有意思的方向——一个从**目标函数**入手修 OPD 的不稳定，一个从**轨迹构造**入手修 OPD 的 prefix failure。不冲突，理论上甚至可以叠加：先构造 relay trajectory + 接力 token，再用 top-K truncated reverse-KL 做 loss。

**适合谁读**：在做 Qwen / GLM 系列蒸馏的工程团队、复现过 OPD 但被 prefix failure 折磨过的研究者、想了解 speculative decoding 在训练侧能怎么用的人。代码已开源，门槛不高。

---

觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。
