---
title: "用国际象棋测出预训练到RL的缩放律：当预训练和RL在同一台显微镜下"
date: 2026-07-21
arxiv: 2607.16097
tags: [论文解读, 强化学习, 预训练, 缩放律, 国际象棋, GRPO]
---

# 用国际象棋测出预训练到RL的缩放律

你有没有过这种困惑：RLHF 跑了一通，模型 pass@1 涨了 5 个点，但 pass@16 几乎没动。强化学习到底是在让模型"学会新的推理路径"，还是只是把基座本来就藏着的好答案"挑出来"？同样的故事，Yue 等人（清华 LeapLab，NeurIPS 2025 满分论文）说 RL 只是 sharpen base model，Yuan 等人说 RL 能 compose 出新技能。这场争论之所以一直没定论，是因为一个根本的实验困难——主流 LLM 的 pretraining 语料是上万亿 token 的 Common Crawl，没人能在受控条件下把"pretraining 选择"和"RL 训练"两条曲线干净地拆开看。

NYU + Modal Labs + UCLA + UIUC + Columbia 联合团队（arXiv:2607.16097）干了一件挺狠的事：把"pretraining → SFT → RL"这条标准流水线**搬到了国际象棋上**。他们用 54B token 的人类棋谱做 pretraining，用合成推理轨迹做 SFT，用 GRPO 在可验证奖励的 chess puzzle 上跑 RL，模型规模 5M 到 1B 一字排开。整篇文章就是用这个"受控显微镜"回答两个问题：

1. 固定总 compute 时，pretraining 和 RL 怎么分？**多少给基座，多少给 RL？**
2. RL 到底对策略做了什么——是 sharpen，还是能发现新东西？

答案给出了一个**联合 pretraining–RL 缩放律**：

$$R(C_{RL}, N, T) = f(L_{pt}) + g(N, T) \cdot \log_{10}\frac{C_{RL}}{C_{ref}}$$

其中 $L_{pt}$ 是 pretraining 的 validation loss。这条式子的意思是，给定 RL 算力时，post-RL 性能**由 pretraining loss 决定**；而 RL 曲线的**斜率**则主要由 pretraining token 数 $T$ 驱动，模型大小 $N$ 是次要修正。compute-optimal frontier 上，RL 应分到的比例从 50M 模型的 ~20% 升到 680M 模型的 ~28%——总 compute 越大，RL 的相对权重越高。

更值得玩味的是机制发现：在简单棋题上 RL 主要放大 SFT 已经偏好的正确动作（"sharpen"），在难题上 RL 既能挖出原本几乎不存在的正确动作（"discovery"），但同时也会强化错误动作。这三件事同时发生，正好解释了为什么 pass@1 涨、pass@16 几乎不涨——RL 在收窄分布。

最后他们把同样的"pretraining token 数 ↔ RL 收益"模式在 1B 的 OLMo-2 数学模型上复现了一遍。chess 上的结论在文本任务上不只是一个偶然的巧合。

---

## 核心摘要

**痛点**：pretraining 和 RL 的缩放律各自被研究得很透，但**两者如何互动**没人能说清。pre-LLM 时代 OpenAI Five / AlphaZero 这类工作能完全在 RL 框架下做，但 LLM 时代不行——随机初始化直接上 RL 几乎拿不到任何 reward，必须有 pretraining prior，所以"pretraining 得多好才有 RL 的发挥空间"成了悬而未决的问题。同时，"RL 是 sharpen 还是 discover"是 2024-2025 推理模型之争的焦点，Yue 2025 / Yuan 2025 / Sun 2025 各执一词。

**方案**：把"pretraining → SFT → RL"全流程搬到国际象棋上。pretraining 语料是 54B token 的人类 Lichess 棋谱（可按 Elo 切片），SFT 用合成推理轨迹（policy 自己采样多个 continuation 拼成 prefix tree），RL 用 GRPO 在 156K 个可验证的 chess puzzle 上做。模型从 5M 到 1B 共 10 个尺寸。

**关键效果**：
- 36 组 pretraining-RL 组合上拟合出联合缩放律：post-RL 性能 $R^{ref}_{N,T}$ 与 pretraining loss 的 Spearman 相关 $|\rho|$ 从 0.93（$\log_{10} C = 16$）到 0.99（$\log_{10} C = 20$）；RL 斜率与 $\log_{10} T$ 的 Spearman 相关 +0.84。
- compute-optimal frontier 上 RL 占比从 50M 模型的 ~20% 涨到 680M 模型的 ~28%。
- 机制分析：简单题上 GT amplification 占比 80%→35%（B1→B5），难题上 wrong-mode amplification 从 ~1% 升到 ~20%。

**我的判断**：这是一篇典型的"工程显微镜型"工作——它没有发明新算法，而是把"pretraining 到 RL"这段黑箱打开给你看。论文的真正价值是**给业界提供了可证伪的预测**：以后做 RL 后训练前，可以先量一下 pretraining loss 估一估天花板在哪；做 RL 时如果发现 pass@1 涨但 pass@16 不涨，应该怀疑 wrong-mode amplification 而不是急着调 learning rate。值得拿出来认真读。

---

## 论文信息

| 字段 | 内容 |
|---|---|
| 标题 | Understanding Reasoning from Pretraining to Post-Training |
| 作者 | Jingyan Shen*, Ang Li*, Salman Rahman, Yifan Sun, Micah Goldblum, Matus Telgarsky, Pavel Izmailov（* equal contribution） |
| 机构 | New York University; Modal Labs; UCLA; UIUC; Columbia University |
| arXiv | [2607.16097](https://arxiv.org/abs/2607.16097) |
| 提交日期 | 2026-07-17 |
| 模型 & 数据 | [huggingface.co/pavelslab-nyu/pre2post-chess](https://huggingface.co/pavelslab-nyu/pre2post-chess) |
| 代码 | [github.com/pavelslab-nyu/pre2post-chess](https://github.com/pavelslab-nyu/pre2post-chess) |

---

## 为什么需要这篇论文

### pretraining 和 RL 的研究是分裂的

过去 5 年 scaling law 走得很快。Kaplan 2020 给出"loss 随模型大小、数据、算力按幂律下降"，Hoffmann 2022 给出 Chinchilla 规则（Gopher 用 D/N=20）。但这些都**只看 pretraining**。

到了 RL 时代，Khatri 2025 的 The Art of Scaling RL 用 sigmoid 拟合 RL 的 compute–performance 曲线。但这篇文章的 pretraining 初始化是**固定的**——它假设你已经有一个模型，然后问 RL 怎么加。

两个独立的缩放律中间存在一个巨大的空隙：给定固定总 compute，**怎么在 pretraining 和 RL 之间分**？给定一个已有的 pretraining，**RL 能跑多远**取决于什么？这两个问题没有量化答案。

### "RL 是 sharpen 还是 discover" 是 2024-2025 推理模型之争的焦点

说 sharpen 的人（Yue 2025，清华 LeapLab，NeurIPS 2025 满分）：基座模型在 pass@1024 上**反超** RL 模型，RL 训练后的推理路径其实都在基座的采样分布里。RL 只是把已有路径的"中奖率"调高了。

说 discover 的人（Yuan 2025）：RL 能 compose 预训练技能形成新能力。

说"分情况"的人（Sun 2025, Zhang 2025）：看具体问题，有些题目上"顿悟"出现，有些题目 RL 完全失败。

三方都有实验数据，但实验设计有共同缺陷：pretraining 语料是 Common Crawl、训练语料是数学/代码，没法逐状态归因 RL 到底改了哪一步的概率。这篇论文说，行，那我们去一个**每一步都有 ground-truth** 的领域做这件事。

### 为什么用国际象棋

作者选 chess 不是因为它对 LLM 来说新，Ruoss 2024（Google DeepMind）和 zhang2024human 已经把 Transformer 训练到 grandmaster level 了。他们选 chess 的真正理由是：

1. **action space 紧凑且明确**：每步是合法走子，token vocabulary 只有 81 个，比自然语言小几个数量级。
2. **每步有 ground-truth**：Stockfish 可以判定每步走法的好坏，puzzle 的解法序列是唯一正确的。
3. **数据可控**：Lichess 有 10 年+ 的人类棋谱，可以按 Elo、长度、风格切片。比 Common Crawl 干净太多。
4. **小模型也能产生非平凡 scaling 信号**：5M-1B 的模型在 chess puzzle 上能跑出从 ~5% 到 ~60% 的 pass@1 区间，足以看出 scaling 规律。

---

## 怎么做的：chess testbed 的设计

### 整体流水线

![图1：方法概览](https://arxiv.org/html/2607.16097v1/x1.png)

*图1：chess testbed 的三阶段流水线。左边是 (a) pretraining → SFT → RL 的完整训练流，词表是 81 个 chess token（每个走法 4 个 token：piece / source / destination / flag），SFT 阶段让模型生成 prefix tree 形式的推理轨迹，RL 阶段对每个 prompt 跑两次 rollout 由 Stockfish 判对错；右边 (b) 给出联合 pretraining-RL 缩放律的公式 $R = f(L_{pt}) + g(N,T)\log(C_{RL}/C_{ref})$，左下紫色曲线是 reference reward 随 pretraining loss 单调下降，右下绿色是斜率随 $\log T$ 线性上升；(c) 是机制分析的两张棋盘对比——简单题上 SFT 已经把正确走法排第一，RL 只是再加把劲；难题上 SFT 把 95% 概率压在错误走法上，RL 能从尾部把正确走法挖出来。*

训练流水线完整镜像标准 LLM pipeline：

1. **Pretraining**：54B token 的人类 Lichess 棋谱（Blitz + Rapid 2022），用 Qwen3 架构在 8 块 H200 上训。模型规模：5M、10M、20M、32M、50M、100M、200M、410M、680M、1B 共 10 个尺寸。
2. **SFT**：合成 reasoning trace。给定一个 puzzle 初始局面 $s_0$，用 proposal policy 采样 $K$ 个 continuation，把它们按公共前缀 merge 成一棵 prefix tree，深度优先遍历序列化进 `<T>...<sep>...</T>` 标签里。然后让模型从 trace 中挑出 best 走法作为监督目标。**关键的实验技巧**：trace 本身是由 policy 自己生成的，所以 SFT 教的是"怎么在自己的分布里挑最好的"，不是"模仿外部强模型"。
3. **RL**：在 Lichess puzzle 上用 GRPO 训练。puzzle 有 5 个难度 bin（B1: Elo 800-1200 到 B5: Elo >2400），奖励是 binary outcome reward——完整解法序列全对得 1，错一步得 0。这是 hard verifiable reward。

### 训练配置的工程细节

模型是 Qwen3 dense decoder-only Transformer（grouped-query attention），不共享 embedding 权重。Pretraining 用 AdamW，peak LR $10^{-3}$，cosine schedule，effective batch size 524K tokens。SFT 在 4 块 H100 上 3 个 epoch，LR $3 \times 10^{-4}$。RL 用 verl 框架，LR $10^{-5}$，group size $G=8$，KL 系数 $10^{-3}$，8 块 H200 训练。

50M 模型跑 2000 步 RL 大约要 160 GPU-hours——这是整篇论文能扫 36 个 (pretraining, RL) 组合的代价。规模再大一个量级就做不动了，所以论文把 pretraining-RL 交互的核心 sweep 限制在 1B 以内（外推靠拟合的缩放律做）。

---

## 关键实验：联合 pretraining–RL 缩放律

### 测得是什么

作者对 4 个模型尺寸（20M / 50M / 200M / 680M）每个都扫 8-11 个 pretraining compute budget，每个 checkpoint 都跑 SFT + 1000-5000 步 RL，total 36 个 (N, T, RL steps) 组合。每个组合在 B1-B4 难度的 1480 道题上报告 pass@1 和 pass@16。**所有 compute 都按 $6NT$ 公式换算成 FLOPs**，pretraining、SFT、RL 用同一单位。

![图2：经验 compute frontier](https://arxiv.org/html/2607.16097v1/x2.png)

*图2：4 个模型尺寸 × pass@1 / pass@16 的经验 frontier。横轴是总 compute（对数 FLOPs），纵轴是 puzzle 准确率。颜色是 pretraining compute，开放圆圈是 pre-RL baseline，黑色虚线是 frontier，蓝色百分比是 frontier 点对应的 RL 比例。**几个直观信号**：(1) 20M 模型 pass@1 的 frontier RL 比例从 5% 涨到 32%，随总 compute 提升；(2) 50M、200M、680M 模型的 pass@16 曲线几乎是平的——RL 几乎不提升 pass@16；(3) pre-RL baseline（开放圈）位置越靠右的 checkpoint（SFT 起点越高）frontier 越靠上。*

### 从曲线到公式

每个 (N, T) checkpoint 的 RL 训练曲线（reward vs $\log_{10} C_{RL}$）在中间非饱和区是近似线性的。线性拟合出两个参数：

- $R^{ref}_{N,T}$：在参考 RL compute $C_{ref} = 10^{20}$ 处的拟合 reward
- $B_{N,T}$：斜率，每增加 10 倍 RL compute 涨多少 reward

这两个参数是 (N, T) 的函数，作者尝试用 pretraining 的几个量来预测它们。

![图3：pretraining 性质预测 RL scaling](https://arxiv.org/html/2607.16097v1/x3.png)

*图3：pretraining 性质预测 local RL scaling。**左**：$R^{ref}_{N,T}$ vs pretraining validation loss $L_{pt}$，不同曲线对应不同 $C_{ref}$，从 $\log_{10} C=16$ 时 Spearman $|\rho|=0.93$ 涨到 $\log_{10} C=20$ 时 $|\rho|=0.99$——RL 算力越大，pretraining loss 对 post-RL 性能的预测越准。**中**：$B_{N,T}$ vs $\log_{10} T$，linear fit $R^2=0.70$，Spearman $\rho=+0.84$——pretraining token 数（不是 N）和 RL 斜率强相关。**右**：joint fit $B = -0.208 + 0.017 \log_{10} T + 0.009 \log_{10} N$，Pearson $r=+0.92$，斜率预测值与实际值散点几乎在 $y=x$ 上。*

联合形式：

$$R(C_{RL}, N, T) = f(L_{pt}) + g(N, T) \cdot \left(\log_{10} C_{RL} - \log_{10} C_{ref}\right)$$

其中 $f(L_{pt}) = \gamma_f + \alpha_f e^{-\beta_f L_{pt}}$（指数衰减型，$\beta_f=12.85$），$g(N,T) = \alpha_g + \beta_g \log_{10} T + \gamma_g \log_{10} N$（joint fit，$\log T$ 系数 0.017、$\log N$ 系数 0.009——token 几乎是 size 的两倍权重）。

拟合出的具体数字（log10 C=20 时）：$f(L) = 0.0314 + \exp(4.87 - 12.85 L)$，$g(N,T) = -0.216 + 0.0172 \log_{10} T + 0.0098 \log_{10} N$。

**LOO 验证**：用 leave-one-run-out 评估预测能力，Chinchilla-$L$ 模式（用 (N,T) 预测 loss）下 RMSE 0.0194，observed-$L$ 模式下 0.0102；预测的 $R^{ref}$ vs 实际 Pearson r=+0.97~+0.99。

### compute-optimal frontier

把联合 scaling law 喂给 Chinchilla loss surface $L(N,T) = 0.414 + 95.4 N^{-0.404} + 1.62 \times 10^5 T^{-0.677}$（作者在 chess 上重拟合的参数，与 Hoffmann 2022 的 $\alpha=0.34, \beta=0.28$ 相比，chess 的 $\beta$ 几乎是语言的两倍，意味着 chess 上 token 更值钱），就能在 $(N, T, C_{RL})$ 三维空间搜索最优分配。

![图4：外推的 compute-optimal frontier](https://arxiv.org/html/2607.16097v1/x4.png)

*图4：外推的 compute-optimal frontier。**左**：simulated frontier per model size，黑实线是 law frontier，灰虚线是 empirical frontier，两者贴合得不错。模型尺寸从 50M 到 2B 都能搜。**中**：最优 RL 比例随总 compute 的变化——50M 模型 ~19%，150M ~21%，200M ~22%，300M ~23%，400M ~25%，500M ~27%，680M ~28%。**红虚线是 continuous-N 优化**的结果，比 per-size frontier 略高但趋势一致。**右**：frontier 选中的 $T^* / D_{opt}(C_{pt})$ 几乎贴着 Chinchilla 1.0 这条线——意思是"总 token 量上，frontier 几乎不偏离 Chinchilla 规则"，变化主要在 RL/pretraining 怎么分。*

关键 takeaway：frontier 点的 RL 比例从 50M 的 ~20% 升到 680M 的 ~28%。在低 compute regime，把更多预算花在 pretraining 上是更划算的；但总预算越大，RL 的边际价值越高，应该分到更大的相对比例。

---

## RL 到底做了什么：机制分析

### 不只是 sharpen

pass@1 涨而 pass@16 不涨的现象，作者认为源于 RL 的**异质效应**。他们对每个 puzzle 状态 $s$ 评估"induced move policy" $\pi_\theta(a|s)$（把 token 空间折回走法空间，从 128 个 reasoning trace 里 marginalize），然后按策略更新方向把状态分类。

分类维度（top-3 类别）：

| 类别 | 定义 |
|---|---|
| **Ground-truth amplification** | 正确走法原本在 top-3，被 RL 进一步强化 |
| **Tail discovery** | 正确走法原本在 top-3 之外、且概率 < 0.05，被 RL 拉进 top-3 |
| **Wrong-mode amplification** | 正确走法始终在 top-3 之外，初始首选的错误走法被 RL 进一步强化 |

![图5：RL 重塑策略的方式](https://arxiv.org/html/2607.16097v1/x5.png)

*图5：5 个难度 bin × 3 类更新方向的占比，曲线是不同 RL 训练 step 数。**左**：ground-truth amplification 在简单题（B1）占 ~80%，到难题（B5）降到 ~35%——简单题上 RL 的主要工作就是"再 push 一下正确走法"。**中**：tail discovery 在 B1 几乎为 0，在 B3-B5 涨到 5%——难题上 RL 真能挖出 SFT 看不到的好走法。**右**：wrong-mode amplification 从 B1 的 ~0% 涨到 B5 的 ~20%——难题上 RL 也会强化错误走法。**重要的隐含信息**：右图错强化的比例在 step 750 时比 step 50 时更高，说明这个负效应随训练累积。*

这张图基本上是整篇论文最核心的"机制 insight"：RL 在不同难度上做**完全不同的事**，简单题上"放大已有"，难题上"挖掘尾部 + 强化错误"。

### pass@1 涨 pass@16 不涨的来源

把 Figure 5 的逻辑串起来：
- RL 让 SFT 已经 top-1 的正确走法概率更高（GT amp）→ pass@1 涨
- 但 RL 同时让 SFT 排第 2-3 的"次优但不算错"的走法概率降低（wrong-mode amp）→ pass@16 不涨甚至略降
- 难题上 tail discovery 虽然能挖到新的正确走法（好事），但同时 wrong-mode amp 也在涨（坏事），两者部分抵消

论文还给了一个定性例子（Figure 21，没放出来）：B5 难度的某个局面，SFT 把 95% 概率压在错误走法 d6e5 上，RL 之后正确走法 b6e3 从尾部被拉进 top-3。**但这个例子是 cherry-picked 的**——Figure 5 的平均值告诉我们，RL 平均而言是更倾向于"放大已有"而不是"挖掘尾部"。

### 推理结构也在变

SFT 的合成 trace 是一棵 prefix tree，RL 之后树变得更"宽"而非"深"：width-to-depth ratio 上升，branching factor 增加，但最大搜索深度几乎不变。模型更会"列举候选"而非"深思熟虑"。

更细致地看，提议走法的 Stockfish 排名在 RL 后改善（说明候选质量提升），但**5 步以上深度的搜索依然做不好**——RL 改善的是 candidate generation 和 selection，不是 long-horizon planning。

---

## 拓展到数学：联合 scaling law 跨域成立

为了检验 chess 上的规律是不是偶然，作者在 1B OLMo-2 上跑了一组对照实验：同一架构，pretrain 200B token 的数学语料（70% Nemotron-CC-Math-v1 + 30% Dolma3），每 5B-10B 步留一个 checkpoint，总共 14 个 anchor。每个 anchor 都跑 SFT（NuminaMath-CoT，1 epoch） + RL（GSM8K + MATH + DeepScaler mix，24.9K 题）。

![图6：联合 scaling law 在数学上仍然成立](https://arxiv.org/html/2607.16097v1/x6.png)

*图6：1B OLMo-2 在 14 个 pretraining anchor（10B-200B tokens）上的 RL scaling 拟合。**左**：$R^{ref}_T$ vs pretraining loss，Spearman $\rho=-0.99$（log10 C=18.5 锚点）——chess 上的"pretraining loss 预测 post-RL 性能"在数学上**更**准。**中**：$B_T$ vs $\log_{10} T$，Spearman $\rho=+0.90$，quadratic fit $R^2=0.90$，Pearson $r=+0.95$——"pretraining token 数预测 RL 斜率"在数学上同样成立。**右**：pre-RL baseline $R_0$ vs pretraining loss，exponential fit $R^2=0.93$，Spearman $\rho=-0.96$——SFT 起点也是 pretraining loss 的指数函数。*

数学上拟合出的 $f(L) = R^{ref}_T(L)$ decay rate 比 chess 上还陡（$\beta=15.07$ vs chess 的 12.85），说明**pretraining 对 post-RL 性能的影响在数学任务上甚至更强**。

---

## 怎么读这篇论文

### 亮点

第一，**chess 是一个被低估的 LLM research testbed**。token 空间小、可验证、数据可控，让"逐状态归因 RL 行为"变得可行。这是 Mechanism Analysis 那一节能写出 Figure 5 那种细粒度结果的根本原因。

第二，**给"pretraining 与 RL 的关系"补上了量化拼图**。以前有 Kaplan 2020 / Hoffmann 2022（只看 pretraining），有 Khatri 2025（只看 RL），现在多了这篇**把两个阶段串成一条函数**。

第三，**实操建议可落地**。比如想做 7B reasoning model，你已经有 5T token 的 base：跑个 baseline 看一下 NLL，用文中的 $f(L)$ 估一估 RL 后能到哪个水平。这比拍脑袋"调 LR + GRPO 组大小"靠谱多了。

### 我会问的几个问题

第一，**1B 是个硬天花板**。论文自己的 limit 也承认 1B 之上的 scaling trend 不一定延续。compute-optimal frontier 的外推依赖 law fit 在 observed range 内有效，再外推到 70B / 100B 量级会积累多少误差没人知道。

第二，**wrong-mode amplification 怎么处理**？Figure 5 右图里 RL step 越多 wrong-mode amp 越严重——这其实是个负面发现，但论文没给解决方案。Limitation 提到 "reducing wrong-mode amplification" 是 future work，但坦白说，工业界目前**对 pass@1 涨 pass@16 不涨的"两难"也没有标准答案**，这其实是个 open problem。

第三，**"reasoning 完全是 in-token" 的代价**。这里的 reasoning 是 chess move token，不是自然语言。Table 4 给了个 example：trace 是 17 行走法，answer 是 `Qf3xf7#`。这种 reasoning 是不是能泛化到"自然语言 CoT"是个大问题。论文说他们观察到 "类似 patterns" 出现在 1B 数学模型上，但只是定性观察，**没有定量 scaling law 的数学版本**——数学 1B 只有 14 个 anchor，fit 的 RMSE 0.024 比 chess 上的 0.019 大。

第四，**"chess 比语言更吃 token" 是真的吗**？Chess 上 $\beta=0.679$ vs Hoffmann 2022 语言的 $\beta\approx 0.28$，意味着 chess 训练 token 的边际收益是语言的 ~2.4 倍。可能的解释是 chess 的 vocabulary 小（81），每个 token 信息密度高，loss 下降快。但这也可能是样本量太小（只到 52B token）造成的偏差。

### 跟同期工作的关系

跟 Yue 2025（清华）放一起读最有趣。Yue 2025 的核心实验是"RL 模型在 pass@1 上比 base 好，但 pass@1024 上 base 反超"——和这篇 chess 上 Figure 2 pass@16 的趋势**完全一致**。但 chess 给了更细的机制：Yue 看不到的"RL 在难题上挖尾部 + 错强化"，chess 上能直接看到 Figure 5。这就是**用受控 testbed 的价值**。

跟 Khatri 2025（The Art of Scaling RL）的关系：作者的 RL scaling 部分借用了 Khatri 的 sigmoid 形式，但只在**非饱和区**用 log-linear 一阶 Taylor 近似拟合，而不是拟合完整 sigmoid。理由是 sigmoid 拟合需要跑得足够长才能识别 plateau，而作者的总 RL compute 不足以让大多数 run 进入 plateau，所以局部 log-linear 是更诚实的做法。

---

## 对工程实践的启发

如果你现在正在做一个 reasoning model 的 RL 训练，下面这些是从这篇论文能直接搬走的经验：

1. **先量 pretraining loss 估天花板**。一个简化的 $f(L) = 0.03 + \exp(4.9 - 12.85 L)$（chess 拟合的）虽然不能直接套到你的任务上，但形式应该类似——Loss 越低 RL 起点越高。
2. **把总 compute 投到 RL 上的比例随总预算增大**。低 compute regime（< $10^{19}$ FLOPs）pretraining 主导，高 compute regime RL 的相对价值上升。一个粗略的经验是 20-30% 投给 RL。
3. **如果你看到 pass@1 涨但 pass@16 不涨，要警觉 wrong-mode amplification**。可以借鉴 Figure 5 的分类方法——按 token 概率把"行为改变"分成 amplification / discovery / wrong-mode，看 RL 到底在做什么。如果 wrong-mode 占比 > 10%，该调整策略。
4. **搜索要变宽不要变深**。RL 改进了 candidate generation 但没改 long-horizon search。如果你做的是 multi-step agent（ReAct、tool use），synthetic SFT trace 应该鼓励**广度**而非深度。

---

## 一些收尾的判断

这篇论文最大的贡献不是某个新算法，是**把 pretraining-to-RL 的黑箱打开，告诉你哪些参数控制什么**。在 2026 年这个时间点，OpenAI、Anthropic、DeepSeek、Qwen 这些团队已经在 RL 训练上花了巨额算力，但**没人公开发表过"pretraining 多投一点还是 RL 多投一点"的量化决策框架**。这篇 paper 算是给出了一个可操作的版本——前提是你接受 chess 是一个合理的代理。

短期看，这种工作最有价值的地方可能是给"RL 投资回报率"的争论降温：当你看到 base 模型本身 pass@64 就能反超 RL 模型时，与其急着上 PPO/GRPO/Reinforce++，不如先评估 base 模型的 coverage 有多好。RL 改善的是 sampling efficiency，**不是 sampling ceiling**。

长期看，chess 这个 testbed 值得更多人用。compact action space、exact verification、controllable data distribution、可控的 model size——这些性质在 LLM 主流任务上几乎拿不到。如果你想研究"RL 训练里的信用分配"、"synthetic data 对 reasoning 的影响"、"scaling law 的失效条件"这类问题，chess 是个更干净的实验台。

下一步如果作者想继续推进，我会建议两个方向：一是把 chess 的研究扩到 multi-step agent（tool use），二是把 model size 推到 7B / 13B，验证 1B 的 scaling trend 是否延伸。第二个方向的计算开销可能一个 $10^6$ GPU-hours 量级，估计只有大厂能做得动。

---

觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我。
