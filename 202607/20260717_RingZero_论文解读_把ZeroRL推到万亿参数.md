# 把Zero RL推到万亿参数：苦涩教训的一次完整验证

> Ring-Zero: Scaling Zero RL to a Trillion Parameters for Emergent Reasoning
> arXiv: 2607.12395v2
> 作者：Xinyu Tang, Qianggang Cao, Yurou Liu, Yuliang Zhan, Xiaochong Lan, Yifan Li, Yuchen Yan, Han Peng, Zican Dong, Zhenduo Zhang, Tianshu Wang, Xinyu Kong, Zujie Wen, Wayne Xin Zhao, Zhiqiang Zhang, Jun Zhou
> 提交日期：2026-07-14（v2：2026-07-16）
> 机构：蚂蚁集团 / 阿里
> 论文链接：https://arxiv.org/abs/2607.12395

---

## 核心摘要

如果让你把 zero RL 的训练规模直接拉到 1 万亿参数，你大概率会先皱眉头——光看 GRPO 那套 loss 在大模型上动不动就 entropy collapse、token 越长越失控的德性，扩到 1T 还能不能稳定训练都成问题。

这篇 Ring-Zero（arXiv: 2607.12395）就是冲着这个真问题去的：作者们拿 Ling-2.5-1T-Base（一个 1T 参数的 MoE，激活 63B）从预训练基座直接开 zero RL，完全不经过 SFT，最终在 AIME 2024 上跑到了 **94.1 个点**（即 94.1% pass@1），AIME 2025/2026 上分别是 **92.3 个点和 93.2 个点**。这个成绩已经摸到 GPT-5.5、Claude Opus 4.8 这一档闭源前沿模型的下沿。

更让我感兴趣的是它验证出来的三件事：

1. **规模真的有用**：1T 模型的样本效率和性能天花板都明显高于同源的 104B flash 版本，且越难的基准上差距越大
2. **训练过程是"发现→锐化"两阶段**：用 pass@1024 探测，800 步就平台了，但 pass@1 还能继续涨，这意味着 RL 先把预训练里"睡着"的能力翻出来，再去打磨输出分布
3. **五种高级认知行为是自发涌现的**：拟人化口吻、结构化格式、并行推理、上下文焦虑、自我验证——这些过去需要人工设计 reward 或 prompt 引导的策略，1T 模型在纯 zero RL 下从零自己摸出来了

我的第一反应是：这就是 Sutton 当年说的"苦涩教训"——算力加规模最终会超越人工设计启发式。这篇论文在数学推理这个最硬的 benchmark 上，把这个论点用一整套可复现的工程方案给撑住了。

值不值得读？如果你做 RL/LLM 后训练相关，这篇几乎是必读——它把 zero RL 推到 1T 这一档暴露的所有坑（数值不匹配、长度爆炸、entropy 坍缩、引擎同步）全摊在台面上了，对正在做类似事的人直接是工程参考。

---

## 论文信息

- **标题**：Ring-Zero: Scaling Zero RL to a Trillion Parameters for Emergent Reasoning
- **arXiv ID**：2607.12395
- **作者**（16 人，按原文顺序）：Xinyu Tang, Qianggang Cao, Yurou Liu, Yuliang Zhan, Xiaochong Lan, Yifan Li, Yuchen Yan, Han Peng, Zican Dong, Zhenduo Zhang, Tianshu Wang, Xinyu Kong, Zujie Wen, Wayne Xin Zhao, Zhiqiang Zhang, Jun Zhou
- **第一单位**：蚂蚁集团（机构信息见原文）
- **提交日期**：2026-07-14（v1），2026-07-16（v2）
- **模型规模**：Ling-2.5-1T-Base（1T 参数 / 63B 激活，MoE），Ling-2.5-flash-Base（104B / 7.4B 激活，MoE）
- **训练硬件**：320 × H200 GPU
- **类别**：cs.CL（计算与语言）

---

## 一、为什么要把 Zero RL 推到大模型？

DeepSeek-R1 出来之后，"用纯 RL 在预训练基座上直接激发推理"这条路已经被验证可行。但仔细看现有工作，几乎都停留在 7B/32B/72B 这一档——再往上走，训练直接崩盘的可能性急剧上升。

**作者观察到一个反直觉的现象**：朴素的 zero RL scaling 上去之后，会出三个典型毛病：

1. **可读性崩坏**：训练到后面模型吐出来的推理链东一句西一句，跳来跳去，根本读不下去
2. **Token 冗余失控**：GRPO 这种算法对长序列天然有隐式偏好，token 级 loss 让模型学会"刷长度"而非"刷质量"，最终长到根本没法部署
3. **缺乏自适应深度**：所有问题都按固定预算硬塞推理，简单的也搞个长链条

这意味着 scaling 不是一个"加大 batch、加大学习率"就能搞定的事。背后有一整套稳定性、数值一致性、token 效率的工程问题要解决。

而更重要的研究问题是：**当 zero RL 真的扩展到 1T 参数时，训练动态和涌现能力会怎么变？** 这才是论文要回答的核心科学问题。

---

## 二、方法：四阶段流水线

我直接把论文的 Figure 1 放在这里，它把整篇论文的骨架画得很清楚：

![Ring-Zero 总览图：四阶段训练流水线 + 基础设施优化 + 自发涌现行为](https://arxiv.org/html/2607.12395v2/x1.png)

*图1：Ring-Zero 整体方案。（a）四阶段训练流水线：First-stage RL（推理激发）→ Self-Distillation（压缩与稳定）→ Second-stage RL（持续优化）→ Third-stage RL（自适应深度）。（b）基础设施优化：Mixed-precision Control + Context Parallel Optimization。（c）四种典型涌现行为：拟人化口吻、结构化格式、并行推理、上下文焦虑。*

下面我把每个阶段拆开讲。

### 2.1 First-Stage RL：从基座激发推理

这一步就是经典的 zero RL——拿一个预训练基座，不做任何 SFT，直接用可验证奖励的 RL 推下去。但作者做了两个关键的技术改造：

**改造一：Clipped Importance Sampling Policy Gradient（CISPO）**

这是论文算法选型的核心。来自 MiniMax 2025 年提出的方案。区别于标准 PPO-clip（把裁剪范围外的 token 梯度完全砍掉），CISPO 的做法是：

- 只对裁剪比率 $\hat{\rho}_{i,t}$ 施加 stop-gradient
- 但 **所有 token 的梯度都允许继续流动**

目标函数：

$$\mathcal{J}(\theta) = \mathbb{E}_{q\sim\mathcal{D}, \{o_i\}_{i=1}^G \sim \pi_S} \left[\sum_{i=1}^G \sum_{t=1}^{|o_i|} \operatorname{sg}(\hat{\rho}_{i,t}) \cdot \hat{A}_{i,t} \cdot \log \pi_M^\theta(o_{i,t} | q, o_{i,<t})\right]$$

直觉上：这相当于告诉优化器"小概率 token 也别一刀切压死，只是要小心更新幅度"——在 zero RL 早期，这种"放大低概率 token 梯度"的策略对激发推理至关重要（作者在 Figure 4 里专门对比了 GRPO/DAPO/CISPO/GSPO 四个算法，结论是 CISPO 和 DAPO 前期学习最快，但不稳定；GSPO 稳定但没法引导长链推理增长）。

**改造二：Training-Inference Ratio Correction**

这个改动更工程，但解决了一个非常要命的问题。Zero RL 训练里有个老熟人叫"重要性比率"：

$$\rho_{i,t} = \frac{\pi_M^\theta(o_{i,t} | q, o_{i,<t})}{\pi_S^{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})}$$

分子是训练引擎（Megatron）出来的当前策略，分母是推理引擎（SGLang）跑 rollout 时的旧策略。这两个引擎 **用的浮点精度不同**——训练侧可能是 BF16，推理侧也可能是 BF16，但哪怕一个用了 FP32 的 softmax，分子分母的数值差都会被 softmax 指数运算放大，结果就是训练-推理 log-prob 的差距越拉越大，最终 entropy collapse、reward 崩盘。

作者的做法是：**分子统一用 Megatron 的 logits**（保证和 loss 算出来的梯度一致），分母用 SGLang 的 logits，$\epsilon_{\text{high}}=5.0$ 单边裁剪，不设下界。配合每 400 步更新一次参考模型、K3 散度 KL 惩罚（$\beta=10^{-4}$），整个训练在 1T 模型上稳得不行。

我专门去看了一眼 Figure 6 的消融对比——baseline 配置大概 800 步就崩，IcePop 这种单纯加裁剪阈值的方案能撑到 2700 步但还是崩，作者这套完整方案全程稳如老狗。这个改进看着不起眼，但是 **1T 规模下不做就根本 train 不动**。

### 2.2 Self-Distillation：压缩与稳定

第一阶段跑完，模型推理质量上去了，但序列长度也跟着上去了——这就是上面说的"token 冗余"。第二阶段做的事是：

1. 用第一阶段的专家模型对每个 query 采样多个 rollout
2. **两步长度精炼**：先挑最短且正确的轨迹，再让模型自评打分，过滤冗余片段
3. 用精炼后的语料微调基座（学习率 $7\times10^{-5}$，3 epoch，64k 序列长度）

这一步的意义是 **重置训练-推理引擎之间的数值差距**——蒸馏完得到一个新模型，它和 SGLang 引擎的 log-prob 差距小得多，KL 收敛更快，第二阶段 RL 的稳定性就有了新起点。

### 2.3 Second-Stage RL：从 Token 级切到 Sample 级

这一步和第一阶段算法层面几乎一样（同样用 CISPO + 比率校正），但 **loss 从 token-level 切到 sample-level**：

$$\mathcal{L}_{\text{Ring-2.5-1T-Zero-II}}(\theta) = -\mathbb{E} \left[\sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \operatorname{sg}(\hat{\rho}_{i,t}) \cdot \hat{A}_{i,t} \cdot \log \pi_\theta(o_{i,t} | q, o_{i,<t})\right]$

加了个 $1/|o_i|$ 归一化，长序列不再获得不成比例的梯度信号，token 长度自然被压住。同时这一阶段 **直接拿掉 KL 惩罚**——因为蒸馏后的模型已经足够好起点，KL 反而会限制探索。

这个设计挺巧妙的。Token-level loss 鼓励探索（前期必要），sample-level loss 防止长度爆炸（后期必要），两阶段搭配正好踩在跷跷板的中间。

### 2.4 Third-Stage RL：自适应推理深度

最后一阶段作者引入了 **基于层级的训练策略**：

- **Low**：最大 4k tokens（简单题）
- **Medium**：最大 16k tokens（中等题）
- **High**：最大 64k tokens（难题，可外推 Yarn=2 到 128k）

每层配一个特定系统提示 $p_k$，模型学会"看菜吃饭"——简单问题别浪费 token，复杂问题该花就花。这其实是把"自适应推理"这件事从模型自己摸索变成了训练目标的一部分。

### 2.5 基础设施：两个看似小但要命的优化

**Mixed-Precision Control**：

主体用 BF16，但 **attention 的 softmax 和 LM head 必须用 FP32**。原因就是上面说的——softmax 指数运算会把小舍入误差指数级放大。RL 训练对此特别敏感，差几个 bit 整条曲线就垮了。

**Context Parallelism Optimization**：

Ling-2.5 系列是 MLA（Multi-head Latent Attention）+ Lightning Attention 的混合架构。MLA 层用 all-to-all CP 策略（K/V 压到低秩潜空间，通信量小很多），Lightning Attention 层用 AllGather（KV state 固定大小，一次性广播）。这俩在数学上等价于标准 ring attention，但通信效率高很多。

320 张 H200 跑 1T 模型，能稳如磐石跑完整个四阶段训练，这套基础设施优化功不可没。

---

## 三、实验结果：数据说话

主实验表比较长，但值得仔细看。我把核心数据搬过来：

| 模型 | AIME 2024 | AIME 2025 | AIME 2026 | HMMT 25.2 | HMMT 25.11 | HMMT 26.2 | IMOAnswerBench |
|------|-----------|-----------|-----------|-----------|------------|-----------|----------------|
| GPT-5.5 | -- | -- | 98.3 | -- | 96.5 | 96.7 | 91.4 |
| Claude Opus 4.8 | -- | -- | 95.7 | -- | 96.5 | 96.7 | 83.5 |
| Gemini 3.1 Pro | -- | -- | 98.2 | -- | 94.8 | 87.3 | 81.0 |
| DS-V4-Pro Max | -- | -- | 94.6 | -- | 94.4 | 95.2 | 89.8 |
| **Ring-2.5-1T-Zero** (Stage I) | 89.1 | 83.3 | 84.2 | 76.7 | 75.8 | 66.2 | 59.3 |
| Self-Distillation | 92.3 | 87.3 | 88.1 | 81.9 | 79.9 | 71.2 | 63.8 |
| Second-Stage RL | 93.5 | 91.6 | 92.5 | 87.4 | 87.1 | 78.1 | 72.7 |
| Yarn=2 外推 | 94.1 | 92.3 | 93.2 | 90.6 | 90.8 | 81.0 | 75.5 |
| **Third-Stage Medium 模式** | 90.9 | 88.1 | 90.8 | 80.4 | 84.8 | 74.1 | 70.8 |
| **Third-Stage High 模式（Yarn=2）** | 93.2 | 91.0 | 91.4 | 86.3 | 86.4 | 78.4 | 72.7 |

几个关键观察：

1. **从基座到 Stage I 没有任何 SFT**——AIME 2026 直接干到 84.2%。这个起点相当炸裂，说明 1T 模型的预训练里就潜伏着大量数学推理能力
2. **每一阶段都稳定提升**，尤其是 Self-Distillation（+3.2/4.0/3.9）和 Second-Stage RL（+1.2/4.3/4.4）
3. **第三阶段的 Medium 模式性价比最高**：平均 8085 tokens 就能拿到 90% 以上的成绩，对部署友好
4. **Yarn=2 外推到 128k 上下文**后，AIME 2024 摸到 94.1%，已经和闭源前沿下沿平齐

值得说一句的是，作者把 DeepSeek-R1（77.9/79.8）也拿来对比了，但 DeepSeek-R1 不是同一时间窗口的工作，主要参考价值是验证"zero RL 这条路能走多远"。

---

## 四、三大核心发现

### 4.1 发现 1：规模真的能换来能力天花板

论文用 Figure 10(a) 直观展示了这个差距——同样是 zero RL：

![Pass@1 训练曲线：Ling-2.5-1T-Base vs Ling-2.5-flash-Base](https://arxiv.org/html/2607.12395v2/x31.png)

*图10(a)：1T 基座（红线）vs 104B flash 基座（蓝线）在 AIME 2024 上的 pass@1 演化。1T 模型在 3600 步就跑到 89%，而 104B flash 即使到 5000 步也只有 72%——越难的任务差距越大。*

我的第一反应是：这个差距大得有点反常识。激活参数只差 8 倍（63B vs 7.4B），但 1T 模型在 AIME 2024 上比 104B 高了 17 个点。这其实印证了那句话——**MoE 的总参数量比激活参数重要得多**，因为 router 决定了"知识总量"。

### 4.2 发现 2：训练是"发现→锐化"两阶段

这个发现我特别喜欢，因为它给"RL 到底在学什么"提供了一个清晰答案。

作者追踪了一个有趣的指标——**pass@1024**（1024 个采样中至少一个正确的概率）。理论上，如果 RL 只是让模型"更确信正确的事"，pass@1024 应该基本不变；如果 RL 在"发现新的解题路径"，pass@1024 应该持续上升。

实测结果（Figure 10b）：

![Pass@1024 在训练 800 步后就平台](https://arxiv.org/html/2607.12395v2/x32.png)

*图10(b)：AIME 2024 上 pass@1024 随训练步数的变化。800 步后曲线就完全平台了，稳在 96.5% 左右——这意味着 RL 在这个阶段已经找不到任何新推理模式了，剩下的时间全在"打磨"已有能力。*

对照主表里 pass@1 还在持续上涨（72%→89%），这两条曲线拼起来讲的结论就是：

> **RL 在早期阶段（discover）从预训练的"沉睡能力池"里把解题模式翻出来；后期阶段（sharpen）则专注于在已发现的模式内做策略优化。**

这个发现对做 RL 训练的人很有意义——**800 步之后你的"新进展"可能不是真新进展，只是在重复已有路径上做局部最优调整**。如果你想"发现"更多，可能得换数据、加预训练 token，而不是堆 RL 步数。

### 4.3 发现 3：五种高级认知行为自发涌现

这是最让作者兴奋的部分，也是最容易让读者怀疑的部分。

**先上数据**：作者在 Comprehensibility 维度上拿自己的 Ring-2.5-1T-Zero 跟 MiniMax M2.7、GLM 5.1、Kimi K2.6、Qwen3.5 397B 四个闭源/半开源模型做了 90 道 AIME 题的成对对比：

![Comprehensibility 评估：Ring-Zero 在所有配对中压倒性胜出](https://arxiv.org/html/2607.12395v2/x6.png)

*图3(a)：Comprehensibility 维度上的成对胜率（90 道 AIME 2024-2026 题）。Ring-Zero 对四个强基线全部压倒性胜出：vs MiniMax M2.7 是 78-12，对 GLM 5.1 是 76-14，对 Kimi K2.6 是 72-15-3，对 Qwen3.5 397B 是 64-22-4。*

我得说，看到这组数据我第一反应是怀疑的——这个评估用的是 LLM-as-a-Judge，难保不是 judge 模型的偏见。但结合 Figure 3(b) 的可复现性数据：

![Reproducibility 评估：用 Ring-Zero 的轨迹蒸馏出来的学生模型效果更好](https://arxiv.org/html/2607.12395v2/x7.png)

*图3(b)：用 Ring-Zero 的推理轨迹蒸馏出的学生模型表现。Qwen-32B 蒸馏后 78.4% vs DeepSeek-R1 蒸馏的 72.6%，Llama-70B 蒸馏后 74.5% vs 70.0%。关键是 Ring-Zero 只用了 100K 蒸馏样本，而 DeepSeek-R1 用了 800K——数据量差了 8 倍，效果反而更好。*

这组蒸馏数据其实非常硬核——**用 1/8 的数据，蒸馏效果反而超过 DeepSeek-R1**。这意味着 Ring-Zero 的推理轨迹里真的"塞进了"可学习的解题模式，不是单纯的格式漂亮。

至于那五种涌现行为（拟人化口吻、结构化格式、并行推理、上下文焦虑、自我验证），我看下来觉得：

- **拟人化**（"brain fart"、"genius"、偶发脏话）这个我持保留意见，作者自己也在论文里说"可能是预训练里论坛数据诱导的伪影"——但反过来想，这恰恰说明模型模仿的不只是"专家逻辑"，还有"专家情绪"
- **结构化格式**（自发 "Step 1 / Step 2 / Step 7: Verify" 这种）——这个我比较信。1T 模型面对 64k 长上下文时，结构化输出本来就是组织注意力的自然最优解
- **上下文焦虑**（token 接近上限时主动简化推理以保证格式正确）——这个最有意思，说明模型学会了"博弈规则"——格式错了零奖励，瞎猜至少有非零概率

---

## 五、CoT 质量评估：超越"答案对不对"

论文对"推理质量"这件事的判断标准，我觉得值得单独拎出来说。

过去大家在数学推理 benchmark 上比的全是 pass@1——你答对了就 100 分，没答对就 0 分。**这其实把"推理过程"完全抹平了**。一个用 5000 token 高质量推理链答对的模型，和一个用 50000 token 暴力搜索答对的模型，被同等对待。

Ring-Zero 提了一个三维评估框架：

### 5.1 Comprehensibility（可理解性）

LLM-as-a-Judge 成对比较，看谁的推理链"读起来更顺"。Ring-Zero 在 90 道 AIME 上对四个基线全部胜出（图3a）。

### 5.2 Reproducibility（可复现性）

用你生成的轨迹去蒸馏学生模型，看学生能学走多少。这个指标比"人读着顺"更硬——如果你的 CoT 真的编码了可推广的解题策略，学生就能学走；如果只是格式漂亮，蒸馏出来照样拉胯。

Ring-Zero 用 100K 样本蒸馏 Qwen-32B 拿到 78.4%，超过 DeepSeek-R1 用 800K 样本的 72.6%——**数据效率差 8 倍**。

### 5.3 Efficiency（效率）

最简单也最直接：能答对的题目，平均用多少 token。Ring-Zero 在共同能答对的子集上平均只用了 6368 tokens：

![Efficiency 评估：平均 token 数对比](https://arxiv.org/html/2607.12395v2/x8.png)

*图3(c)：在 5 个模型都能正确回答的 AIME 共同子集上，平均推理 token 数对比。GLM 5.1 用 17,220、MiniMax M2.7 用 16,627、Qwen3.5 397B 用 16,292、Kimi K2.6 用 14,115，而 Ring-Zero 只用 6,368——不到其他模型的一半。*

这组数据很关键。Ring-Zero 6 千多 token 解决 AIME 题，意味着模型**真的在“想”而不是“搜”**——它在拿到题目后大致规划了思路，删掉冗余分支，干净利落地落到答案。

---

## 六、消融实验：哪些设计真的有用

论文做了一组非常扎实的消融，我挑几个有启发的讲。

### 6.1 RL 算法对比（Figure 4）

四个候选：GRPO（基线）、DAPO（双 clip + token-level）、CISPO（clipped IS）、GSPO（sequence-level 重要性比率）

**结论**：CISPO 和 DAPO 早期学习最快、长度涨得最快，但最容易 entropy collapse；GSPO 最稳定，但完全没有激励长链推理增长的能力，不适合 zero RL 起步阶段。

我的判断：早期用 CISPO/GRPO 跑"发现"阶段，后期切到 GSPO/类似稳定算法做"锐化"——这个组合可能比单一算法更优。

### 6.2 比率校正：不做就崩

Figure 5 和 Figure 6 的对比非常直观：

- **无 KL 惩罚**：训练-推理 log-prob 差距无界增长 → entropy 坍缩 → reward 崩盘
- **加 KL**：所有指标保持稳定
- **仅 SGLang 分子**：800 步就崩
- **加 IcePop（裁剪阈值）**：撑到 2700 步还是崩
- **完整方案（Megatron 分子 + SGLang 分母）**：全程稳定

这个消融告诉所有做大模型 RL 的人一个硬道理：**数值不匹配不是优化问题，是生死问题**。

### 6.3 Format Reward：少了就失控

作者试了两种格式约束：
- **Format A**：只要求 `<think>...</think>` 包裹
- **Format B**：要求 `<think>...</think><answer>...</answer>` 且必须以 EOS 结尾

Format A 训练到后面序列长度爆炸但 reward 停滞——模型学会了在 think 块里堆无意义文字。Format B 加了 answer 块和 EOS，强制模型"答完就停"，从根上防止退化性长度增长。

这个观察对所有做 reasoning RL 的人特别重要——**你的 format reward 不只是格式约束，它在塑造模型的"何时停止"判断**。

### 6.4 长度惯性：模型学会刷长度

作者还追踪了一个有趣现象：训练后期，对 **首次 rollout 就答对的简单题**，模型也学会了用越来越长的推理。32k 窗口 vs 16k 窗口，32k 训练出来的响应平均长度几乎是 16k 的两倍，但 reward 只涨了 1-2 个点。

这说明模型学了一个"偷懒捷径"——反正长度是免费的多 token 资源，不如都堆满。这就是为什么后期必须切到 sample-level loss + 长度压缩的蒸馏。

### 6.5 超参敏感性（Figure 9）

- **学习率**：$1\times10^{-6}$ / $2\times10^{-6}$ / $3\times10^{-6}$ 都能收敛到差不多——**对学习率不敏感**
- **Rollout 数 G**：G=32 单步收敛最快，G=8 实际 wall-clock 反而最快（采样开销小），**G=16 是平衡点**
- **Loss reduction**：Token-level 长度涨得猛，sample-level 长度平——**这正是分阶段切 loss 的依据**

---

## 七、我的判断

### 7.1 这篇论文最大的价值在哪

坦白讲，单看 AIME 2024 94.1% 这个数字本身——和 GPT-5.5/Claude Opus 4.8 已经只差 1-2 个点了——已经不是新闻级突破。

但这篇论文 **值钱在三件事**：

1. **把 zero RL 推到 1T 这一档的整套工程方案公开了**。数值不匹配怎么解、长度怎么压、引擎怎么同步、KL 怎么调——这些经验以前是各厂不传之秘，现在直接给你看消融
2. **发现-锐化两阶段论**：用 pass@1024 实证了 RL 不是一锅炖的训练，给训练早期/后期该干什么提供了方向
3. **把"涌现认知行为"这件事从轶事变成了可复现的统计现象**。Comprehensibility/Reproducibility/Efficiency 三维评估框架，提供了"超越 pass@1"的硬指标

### 7.2 哪些地方我要打个问号

**问题一：1T 模型的可读性优势到底是真的涌现还是只是格式更整齐**？Figure 3(a) 的成对对比里 Ring-Zero 对 MiniMax M2.7 是 78-12，对 Qwen3.5 397B 是 64-22-4——这个"整齐度"差距到底来自内容质量还是格式偏好，单看 LLM-as-a-Judge 没法拆清楚。理想情况下应该做一个 ablation：把 Ring-Zero 的格式删掉，裸看内容质量。

**问题二：拟人化口吻到底是不是真正的涌现行为**？作者在论文里也说可能是预训练数据的伪影。这意味着这不是 RL 发现的"真策略"，而是模型从论坛数据里学来的"对话习惯"。如果是后者，那"涌现认知行为"这个叙事就要打折扣。

**问题三：作者没怎么碰同期 DeepSeek-V4 / Qwen3.7 / Claude 的内部数据规模**。表里这些基线具体训练了多少 RL 数据、多少 SFT 数据，其实是不透明的。Ring-Zero 的"纯 zero RL 无 SFT"是优势，但和这些对手的对比有一定"赛道不公平"——对手可能用了 10x 的人工数据 + 标注。

### 7.3 对工程实践的启发

如果你也在做大模型 RL 后训练，这篇论文的工程经验直接可用：

- **CISPO 优于 GRPO 用于 zero RL 起步**（前提是配好数值稳定性）
- **重要性比率的分子用训练引擎 logits**——这条建议含金量极高，能省你几周的 debug 时间
- **Token-level loss 只用在前中期**，后期必须切 sample-level，否则长度爆炸
- **Format reward 必须带 EOS 终止信号**——这是从 Format A/B 消融里学到的血泪教训
- **给模型分层（Low/Medium/High）训练**，让模型学会"看菜吃饭"

### 7.4 这到底算不算"苦涩教训"的胜利

Richard Sutton 当年的"苦涩教训"核心论点是：**依赖算力的通用方法最终会胜过依赖人类知识的专门方法**。

Ring-Zero 这次的发现（1T 模型自己摸出五种认知行为）确实给这个论点增加了证据。但我想说 **不要过度解读**——

- RL 本身仍然依赖精心设计的 reward function（acc + format），不是"完全无人类知识"
- 涌现行为是在 **大量 GPU 算力 + 大量训练数据 + 1T 参数** 的组合下出现的，"苦涩"的是"算力账单"而非"模型结构"
- 同样的行为在 104B 模型上是不是真的不涌现？我没看到 paper 给出负面证据。Figure 4 里 Ring-flash 的 GSPO/CISPO 训练曲线显示 104B 也能学到一些，只是天花板低

所以更精确的说法可能是：**在算力和参数规模都到位的前提下，RL 后训练能减少对人工设计启发式的依赖**——这是苦涩教训的"温和胜利"，不是它的"完全胜利"。

---

## 八、收尾

读完 Ring-Zero，我的整体感觉是——

这是一篇 **工程诚意满满** 的论文。320 张 H200 跑出来的曲线、详尽的消融、清晰的失败案例（"不这么做就崩"），对正在做 RL 的人几乎是"免费"的经验传授。它的科学贡献可能不算颠覆性（"规模越大越好"这个论点不新），但在 1T 这一档确实给出了 **可复现的整套方案**。

至于"涌现认知行为"这件事——我保持一个谨慎的乐观。这些行为是统计意义上的真实存在，蒸馏数据也证明了可学习性，但 **离"AGI 涌现"还差得远**。模型学会的是"如何更高效地通过 reward 拿到分数"，不是"如何真正思考数学"。

最后留一个开放问题给同行：**如果你有一个 1T 模型，320 张 H200，500K 数学题数据，你愿意花 3 周训练一个 Ring-Zero 这样的模型吗**？如果愿意，2026 年下半年 zero RL 可能会成为大厂标配；如果不愿意，那可能还有更便宜的路径在等着被发现。

> 觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。

---

## 论文基本信息

- **arXiv**: [2607.12395](https://arxiv.org/abs/2607.12395)
- **HTML 版本**: https://arxiv.org/html/2607.12395v2
- **作者机构**: 蚂蚁集团 / 阿里
- **代码/模型**: 论文未明确开源链接（截至 v2 发布）
