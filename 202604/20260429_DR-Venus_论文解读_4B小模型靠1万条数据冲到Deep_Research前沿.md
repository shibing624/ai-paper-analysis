# DR-Venus：4B 小模型靠 1 万条开源数据，把 Deep Research 干到 9B 全员碾压

## 核心摘要

Deep Research 这条赛道这半年很拥挤——Tongyi、DeepDive、WebSailor、REDSearcher、OpenSeeker，一堆 30B 起步的模型在 BrowseComp 上互卷分数。但端侧/边缘部署一直是个尴尬的话题：30B 跑不动，4B/8B 又打不过，小模型在长程 Agent 任务上常年是个洼地。

蚂蚁集团 Venus 团队的这篇 DR-Venus 给了一个看起来挺极致的答案：以 **Qwen3-4B-Thinking-2507** 作为基座模型，**全部开源数据，只有 1 万条 SFT 轨迹 + 1 千条 RL 数据**，做出了一个在 BrowseComp 上拿到 29.1、BrowseComp-ZH 拿到 37.7 的 4B Agent。这个分数的含义是：**直接把所有 9B 以下的开源 Agent 全部摁在地上摩擦**——之前最强的 4B 同行 AgentCPM-Explore-4B 在 BrowseComp 上是 24.1，DR-Venus-4B-RL 高出 5 个点；和 30B 级别的 Tongyi-DR-30B（43.4）也只差 14 个点了。

更有意思的是 Pass@16 那张图：**4B 模型的 BrowseComp-ZH Pass@16 居然冲到了 78.5，反超 GPT-5 High（65.0）和 Gemini-3-Pro（66.8）**。这件事说明小模型的能力天花板被 Pass@1 严重低估了，配合 test-time scaling 完全是另一种生物。

技术上的核心是两件事：第一，**SFT 阶段的"长轨迹重采样"**——把 100 turn 以上的轨迹权重拉到 5×，强行把模型往 long-horizon 上掰；第二，**RL 阶段用 IGPO 替代 GRPO**——基于信息增益（Information Gain）的 turn-level reward 替代了稀疏的 outcome reward。在他们的消融里，同样的训练流程换成 GRPO 反而是 -1.5，IGPO 是 +2.3，差异非常硬。

这是一篇典型的"工程整合 + 一个关键算法替换"的 paper。亮点不在于发明了什么新组件，而在于它选了一条正确的小模型路径：高质量数据 + 长轨迹偏置 + 密集 turn-level reward。读完之后你会同意一件事：在 Deep Research 这条赛道上，**模型规模可能确实不是决定性因素**。

---

## 论文信息

- **标题**：DR-Venus: Towards Frontier Edge-Scale Deep Research Agents with Only 10K Open Data
- **作者**：Sunhao Dai、Yong Deng、Jinzhen Lin、Yusheng Song、Guoqing Wang、Xiaofeng Wu、Yuqi Zhou（核心贡献者）；Shuo Yang、Zhenzhe Ying、Zhanwei Zhang（贡献者）；Changhua Meng、Weiqiang Wang（指导）
- **机构**：Ant Group（蚂蚁集团 Venus 团队）
- **arXiv**：[https://arxiv.org/abs/2604.19859](https://arxiv.org/abs/2604.19859)
- **代码**：[https://github.com/inclusionAI/DR-Venus](https://github.com/inclusionAI/DR-Venus)

---

## 为什么需要"小型化的 Deep Research Agent"

先聊一下背景。Deep Research 类任务的范式现在已经基本定型：用户给一个复杂的信息查询（比如"某个非主流学术领域里 2024 年之后引用最多的三篇 paper 是什么"），Agent 通过 `search` + `browse` 两个工具反复迭代，最终综合证据给出答案。这里面"反复"是关键词——一次 trajectory 动辄 50 turn 起步，长的能到 200 turn。

工业界主流做法是堆模型规模。Tongyi-DR、WebSailor-V2、DeepDive、REDSearcher 全是 30B 起步。原因也很直接：长程任务对推理稳定性、format 一致性、工具调用准确性的要求高，小模型一旦在某一步崩了，整条轨迹就废了。

但 30B 模型的部署成本是另一码事。说实话我之前在做端侧 Agent 的时候反复碰到这个矛盾：你想把 Agent 推到用户本地（隐私、离线、低延迟），但你能塞进消费级硬件的，撑死也就 4B-8B。所以这篇 paper 直接立了个 flag：**4B 能不能做到 Deep Research 的前沿水平？**

而限制条件还更狠：

- **只用开源数据**——不能拿内部的高质量轨迹，必须从 REDSearcher 公开的 10K 轨迹起步
- **数据量极少**——SFT 只有 1 万条，RL 只有 1 千条 query-answer
- **基座要小**——Qwen3-4B-Thinking-2507

这种约束下能不能 work？看实验结果，能。但能 work 的关键不在于"魔法 trick"，而在于把每一步都做到位。

---

## 方法核心：两阶段 + 两个关键设计

DR-Venus 的训练 pipeline 干净得有点反直觉——就是经典的 SFT + RL 两阶段，没有任何花里胡哨的中间件。但每一阶段都有一个针对小模型 + 长程任务的关键设计。

### Stage 1：SFT 数据清洗 + 长轨迹重采样

REDSearcher 公开的 10K 原始轨迹其实噪声不小。Venus 团队做了一套四步清洗：

| 步骤 | 操作 | 效果 |
|------|------|------|
| 环境对齐 | 把所有轨迹的 message schema、system prompt、tool 调用格式统一成线上推理一致的 protocol | 减少 train/inference mismatch，10001→10001 |
| 工具裁剪 + 去重 | 只保留 `search` 和 `browse` 两个工具，删除 `PythonInterpreter` 等其他工具调用；同时删除重复的 search/browse 调用 | 移除 3378 个 disallowed 工具调用 + 15728 个重复调用，剩 10000 条 |
| 正确性过滤 | 用 Qwen3-235B-A22B-Instruct-2507 做 judge，只保留最终答案正确的轨迹 | 9365 条（93.65%）幸存 |
| **Turn-aware 重采样** | 0-50 turn 的轨迹 ×1，51-100 turn ×2，>100 turn ×5 | 9365→18745 条；>100 turn 的比例从 13.29% 拉到 33.21% |

前三步是常规操作，关键是第四步。

这个想法其实挺直觉但也挺关键：**Deep Research 任务的难点全在长程**——50 turn 以内的任务，模型基本能搞定；100 turn 以上的任务，是模型最容易在中途崩掉的地方。如果均匀采样训练，模型会学到一堆"短任务模式"，但部署时碰到长任务就跪。

5× 的权重看起来很激进，但消融数据说话——下面会看到，加了 Resampling 之后 BrowseComp 涨了 4.0 个点，BrowseComp-ZH 涨了 1.8 个点。**对于"小模型 + 有限数据"这个 setting，长轨迹的供给密度是关键瓶颈，不是总数据量。**

整个 SFT 阶段的目标函数本身没什么花样，就是标准的 next-token loss，但只在 assistant 生成的 token（reasoning + tool call + final answer）上计算，环境返回的 observation 完全 mask 掉：

$$\mathcal{L}_{\mathrm{SFT}}(\theta) = -\sum_{H \in \mathcal{D}_{\mathrm{SFT}}} \sum_{i \in \mathcal{M}(H)} \log \pi_\theta(x_i \mid x_{<i})$$

这个 mask 设计虽然是 agentic SFT 的标配，但对小模型尤其重要——避免模型把环境返回的网页内容也当成自己要学习生成的目标，那样就把模型拉成搜索引擎了。

### Stage 2：用 IGPO 替代 GRPO 的 turn-level RL

这个是 paper 的算法核心。我先把直觉讲清楚再展开公式。

**问题的本质**：GRPO 是 trajectory-level 的稀疏 outcome reward——一条 200 turn 的轨迹，只在最后一步给一个 0 或 1 的信号。对小模型来说这个信号太稀疏了，特别是在长程任务上：rollout group 里经常出现"8 条轨迹全是错的"的情况，advantage 直接 collapse 成 0，整个 group 没有任何梯度可学。

**IGPO 的核心思路**（来自 wang2026information）：把"信息增益"作为 turn-level 的 dense reward。具体一点：在 turn $t$，模型新看到一些证据（比如 browse 了一个网页），那么模型当前对 ground truth 的 log-likelihood 应该会比 turn $t-1$ 高一些。这个**增量本身就是一个稠密的奖励信号**：

$$r_{i,t}^{IG} = \log \pi_\theta(g \mid h_{i,\leq t}) - \log \pi_\theta(g \mid h_{i,\leq t-1}), \quad 1 \leq t \lt T$$

其中 $g$ 是 ground truth 答案 token 序列，$h_{i,\leq t}$ 是到 turn $t$ 为止的全部交互历史。直白讲：**这一 turn 让我变得多有信心说出正确答案？**

工程意义在于——即使最终答案错了，如果中间某些 turn 拉高了对正确答案的信心，那些 turn 仍然是好行为，模型应该被鼓励这么做。这就把 200 turn 的稀疏信号变成了 199 个 turn-level dense 信号 + 1 个 outcome 信号。

DR-Venus 在 IGPO 基础上叠加了三个改良：

#### 1. Browse-Aware IG 分配

`search` 主要返回短摘要，证据稀薄；`browse` 才是真正读取网页内容的环节。所以**只在 browse turn 上算 IG reward**，并把这个 reward 回溯分配给前一次 browse 之后的所有 search turn。等价于把"探索（search）+ 检验（browse）"绑成一个整体来评估。

这个设计在我看来是真有 Agent 工程经验的人才能想到的——**search 和 browse 在因果关系上是耦合的**，单独评估 search 没有意义。

#### 2. Turn-Level Format Penalty

传统 trajectory-level format penalty 有个明显问题：200 turn 里有 1 turn format 错了，整条轨迹的所有 token 都被惩罚——这对 199 个写对的 turn 是不公平的。

DR-Venus 改成 **per-turn 粒度**：

$$\hat{r}_{i,t} = \begin{cases} r_{i,t}, & \text{turn } t \text{ format 合法} \\ -\lambda_{\mathrm{fmt}}, & \text{否则} \end{cases}$$

只惩罚出错的 turn，不殃及无辜。$\lambda_{\mathrm{fmt}} = 1.0$。

#### 3. IG-Scale 自适应 rebalance

这个是处理"超长轨迹下 outcome reward 几乎全 0"的极端 case。当一个 group 内所有 8 条 trajectory 都失败时，outcome 的均值/方差归一化会失效，这时候优化几乎完全被 IG reward 主导，容易陷入局部最优。

IG-Scale 用一个比值 $s = \min(\max(M^O, \eta) / (M^{IG} + \delta), s_{\max})$ 来自适应调节 IG reward 的尺度，让它和 outcome reward 量级匹配（$\eta = 0.3, s_{\max} = 10$）。

最终每个 turn 的 reward 还会做一个 $\gamma = 0.95$ 的折扣累积：

$$\tilde{R}_{i,t} = \sum_{k=t}^{T_i} \gamma^{k-t} \bar{r}_{i,k}$$

这样每个 turn 的 reward 不仅反映自己的贡献，也包含对未来的"前瞻性"。

#### IGPO 的最终目标函数

剩下的就是标准的 GRPO 风格 PPO clip + KL 约束：

$$\mathcal{J}_{\mathrm{IGPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|u_i|} \sum_{k=1}^{|u_i|} \min\left( \rho \tilde{R}_{i,k},\; \mathrm{clip}(\rho, 1-\epsilon, 1+\epsilon) \tilde{R}_{i,k} \right) - \beta \mathbb{D}_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}}) \right]$$

其中 $\rho = \pi_\theta(u_{i,k} \mid c_{i,k}) / \pi_{\theta_\mathrm{old}}(u_{i,k} \mid c_{i,k})$。和 GRPO 长得几乎一样，但 advantage 换成了 turn-level 的 $\tilde{R}_{i,k}$。

---

## 实验：4B 干翻所有 9B，逼近 30B

### 主实验：六个 Deep Research benchmark 的全方位对比

![图1：DR-Venus-4B 在 BrowseComp 和 BrowseComp-ZH 上和其他开源模型的对比](https://arxiv.org/html/2604.19859v1/x1.png)

*图1：DR-Venus-4B-RL 在 BrowseComp 上拿到 29.1，BrowseComp-ZH 拿到 37.7。比第二名 AgentCPM-Explore-4B 在两个 benchmark 上分别高 5.0 和 8.6 个点，把所有 9B 以下模型拉开一个明显的身位。*

主表（Table 1）覆盖了三类对手：foundation model（带工具）、≥30B 的训练 Agent、≤9B 的训练 Agent。我把核心数字整理在下面：

| 模型 | 规模 | BrowseComp | BrowseComp-ZH | GAIA(Text) | xBench-DS-2505 | xBench-DS-2510 | DeepSearchQA |
|------|------|-----------:|-------------:|----------:|---------------:|---------------:|-------------:|
| **Foundation Models（带工具）** |  |  |  |  |  |  |  |
| GPT-5 High | - | 54.9 | 65.0 | 76.4 | 77.8 | 75.0 | 79.0 |
| Gemini-3-Pro | - | 59.2 | 66.8 | – | – | 53.0 | 76.9 |
| Claude-4.5-Opus | - | 67.8 | 62.4 | – | – | – | 80.0 |
| DeepSeek-V3.2 | - | 67.6 | 65.0 | 75.1 | 78.0 | 55.7 | 60.9 |
| **Trained Agents (≥30B)** |  |  |  |  |  |  |  |
| Tongyi-DR-30B | 30B | 43.4 | 46.7 | 70.9 | 75.0 | 55.0 | – |
| WebSailor-V2-30B-RL | 30B | 35.3 | 44.1 | 74.1 | 73.7 | – | – |
| DeepMiner-32B-RL | 32B | 33.5 | 40.1 | 58.7 | 62.0 | – | – |
| REDSearcher-30B-A3B | 30B | 42.1 | 49.8 | 80.1 | – | – | – |
| **Trained Agents (≤9B)** |  |  |  |  |  |  |  |
| WebSailor-7B | 7B | 6.7 | 14.2 | 37.9 | 34.3 | – | – |
| OffSeeker-8B-DPO | 8B | 12.8 | 26.6 | 51.5 | 49.0 | – | – |
| WebExplorer-8B-RL | 8B | 15.7 | 32.0 | 50.0 | 53.7 | 23.0 | 17.8 |
| AgentCPM-Explore-4B | 4B | 24.1 | 29.1 | 63.9 | 70.0 | 34.0 | 32.8 |
| **DR-Venus-4B-SFT** | 4B | 26.8 | 35.7 | **65.4** | 69.0 | 35.3 | 37.7 |
| **DR-Venus-4B-RL** | 4B | **29.1** | **37.7** | 64.4 | **74.7** | **40.7** | **39.6** |

几个数值需要重点圈一下：

**第一**，DR-Venus-4B-SFT 单是 SFT 阶段就已经在 5/6 个 benchmark 上击败了之前最强的 4B 同行 AgentCPM-Explore-4B。这意味着光是数据清洗 + 长轨迹重采样，就已经能把 4B 模型拉到一个新的水位。

**第二**，DR-Venus-4B-RL 在 xBench-DS-2505 上拿到 74.7，已经几乎追平了 Tongyi-DR-30B 的 75.0——这是 4B 接近 30B 的硬证据。

**第三**，对比 30B 级别 baseline 的失败案例：DeepDive-32B-SFT 在 BrowseComp 上只有 9.5，DeepMiner-32B-RL 也只有 33.5。**这说明把 30B 模型直接扔进 Deep Research 不一定能拿到好结果，训练 recipe 比模型规模更关键。**

但也要客观看，DR-Venus-4B 和闭源 frontier model（GPT-5、Claude-4.5、Gemini-3-Pro）之间还有一个 30 个点左右的差距。Deep Research 这条赛道，规模和数据的红利还没用完。

### 消融：到底是哪一步在起作用？

| 模型 | 训练 | BrowseComp | BrowseComp-ZH |
|------|------|-----------:|-------------:|
| REDSearcher-30B-A3B（SFT） | SFT | 34.7 | 26.8 |
| DR-Venus-4B-SFT (w/o Resampling) | SFT | 22.8 | 33.9 |
| DR-Venus-4B-SFT (w/ Resampling) | SFT | **26.8 (+4.0)** | **35.7 (+1.8)** |
| DR-Venus-4B-RL (w/ GRPO) | SFT+RL | 25.3 (-1.5) | 35.6 (-0.1) |
| DR-Venus-4B-RL (w/ IGPO) | SFT+RL | **29.1 (+2.3)** | **37.7 (+2.0)** |

这张表有两条信息特别值得品。

**第一条**：DR-Venus-4B-SFT 在 BrowseComp-ZH 上反超 REDSearcher-30B-A3B（35.7 vs 26.8）——同样的训练数据源，4B 把 30B 干翻了。这说明 REDSearcher 团队自己的轨迹其实**还没榨干价值**，认真做数据清洗 + 重采样能挖出比堆参数更多的收益。

**第二条**：GRPO 在 SFT baseline 上是 **-1.5 / -0.1**，IGPO 是 **+2.3 / +2.0**。差异极大。

说实话看到这个数我愣了一下——同样的训练数据、同样的初始化、同样的 rollout 流程，只是 reward 信号从稀疏 outcome 换成 turn-level IG，效果差了 4 个点。这相当于反向证明了一件事：**对于 200 turn 这种超长任务，trajectory-level 稀疏 reward 已经无法提供有效梯度了**。GRPO 在 200 turn 上几乎没法 work，这本身就是一个值得记的工程经验。

### Pass@K：4B 的能力天花板被严重低估

![图2：BrowseComp Pass@K 曲线](https://arxiv.org/html/2604.19859v1/x2.png)

*图2：BrowseComp 上的 Pass@K 表现。RL 模型（橙）从 K=1 到 K=16 都全程领先 SFT（蓝），最终 Pass@16 拉到 63.7，反超 GPT-5 High（54.9）和 Tongyi-DR-30B（43.4）。*

![图3：BrowseComp-ZH Pass@K 曲线](https://arxiv.org/html/2604.19859v1/x3.png)

*图3：BrowseComp-ZH 上的 Pass@K——这张图更夸张。SFT 模型在 K=8 时就达到 74.0，K=16 冲到 78.5，已经反超 Gemini-3-Pro（66.8）和 GPT-5 High（65.0）。RL 模型在小 K 区间（Pass@1, Pass@2）领先 SFT，但在大 K 区间反而稍低（K=16 时 76.5 vs SFT 78.5）。*

这两张图给了一个很反直觉的洞察。

我的第一反应是：**RL 不一定提升模型的能力天花板，它提升的是"用低预算实现这个天花板的可靠性"**。看 BrowseComp-ZH 上 SFT 的 Pass@16 反而高于 RL，原因 paper 给的解释是 RL 的训练数据全是英文（distribution mismatch）。但更普遍的解释可能是：**RL 在缩小 trajectory 多样性，让模型在某一种正确路径上变得更可靠，但牺牲了探索其他路径的能力。**

无论如何，这个发现对工程的指导意义很明确：**部署 4B Deep Research Agent 时，搭配 test-time scaling（多次采样 + 答案聚合）几乎是必选项**。Pass@1 的 37.7 看起来一般，但 Pass@16 的 78.5 已经超过 GPT-5 High——投入 16 次推理预算可能比换更大的模型更划算。

### 工具使用模式分析：browse 是关键

![图4：六个 benchmark 上 SFT 和 RL 的 browse ratio 对比](https://arxiv.org/html/2604.19859v1/x4.png)

*图4：browse ratio 分析。蓝色（实/斜纹）= 答对的轨迹的 browse 比例，黄色 = 答错的轨迹，绿色 = 整体平均。一个一致的模式：答对的轨迹的 browse ratio 普遍高于答错的轨迹，RL 之后这个差距更明显。*

这张图的核心信息：

- 整体 browse ratio：SFT 17.49% → RL **22.46%**
- 正确轨迹的 browse ratio：SFT 23.71% → RL **28.96%**
- 反例修正：xBench-DS-2510 上 SFT 时错的轨迹反而 browse 更多（15.57% vs 14.51%），RL 把这个反向关系修正了（22.99% vs 17.50%）

工程上的解读：**search 只能给 snippet，browse 才能拿到完整证据做 grounded reasoning**。模型如果只 search 不 browse，其实就是在凭片段信息硬猜，错误率自然高。RL 的作用是让模型学会"在合适的时候去 browse"，而不是简单地多 search 或者多 browse。

这个发现和我的直觉一致——之前在做检索增强 Agent 的时候也发现，**只看 snippet 的模型会被检索结果的标题党严重误导，必须读全文才能判断**。

---

## 我的判断

### 这篇 paper 的真正价值

抛开"4B 拿到 SOTA"的标题党价值（这种标题在 Deep Research 这个赛道上其实越来越通货膨胀了），这篇 paper 真正硬的地方有两个。

**第一是工程可复现性**。10K 开源数据 + 1K RL 数据 + 8 张 A100（SFT）+ 16 张 A100（RL），这是大部分研究团队都能负担的预算。模型也开源了。意味着任何团队都可以基于这个 recipe 起跑。Deep Research 这条赛道上**门槛被显著拉低了**——不再需要 30B + 闭源数据 + 上百卡的训练 setup。这一点对学术界和小团队的意义其实比"刷分"大得多。

**第二是把 IGPO 的工程价值打实了**。IGPO 本身是 wang2026information 提出的算法，但这是我看到的第一个把 IGPO 在 200 turn 这种极长任务上跑通并和 GRPO 做严格对比的工作。**+2.3 vs -1.5 这个差距在 RL 论文里是非常少见的硬证据**——同样的数据、同样的 rollout、同样的 PPO objective，只换 reward 设计，差出 4 个点。我之前在做 trajectory-level reward 的多步 Agent 训练时也碰到过类似的现象——长任务下 advantage 一旦 collapse，整个 group 就废了，这种时候你需要一个能让"中间过程"也产生梯度的机制，IG reward 是其中一种很自然的实现方式。

**第三是给"小模型 + test-time scaling"的范式做了一个完整的实证背书**。BrowseComp-ZH 上 4B 模型 Pass@16 反超 GPT-5 High 这个数据点，足够说服很多团队重新考虑"是不是真的需要继续堆模型规模"。这个 trade-off 在生产环境里非常实用——尤其是离线分析、批量数据处理这类对单次延迟不敏感的场景。

### 不那么对的地方

**第一，benchmark 选择有点取巧**。DR-Venus 主要在 BrowseComp、BrowseComp-ZH、xBench-DS 这种"长程网页搜索"类任务上跑，而 GAIA 这种偏综合推理的任务表现并不突出（SFT 65.4, RL 64.4，RL 反而退步）。如果加上 HLE、SimpleQA 这种更难的任务，可能优势会打折扣。

**第二，"4B 反超 30B"的对比对象选得有点巧**。DeepDive-32B-SFT 在 BrowseComp 上只有 9.5——这个 baseline 显然没好好训。如果对比 Tongyi-DR-30B（43.4），4B-RL 还是落后 14 个点。所以**别被"narrows the gap to 30B"这种叙事完全说服**——差距是被显著缩小，但远没消失。

**第三，全英文 RL 数据在 BrowseComp-ZH 上的负效应没解决**。Pass@16 上 RL 反而比 SFT 低了 2 个点，paper 自己也承认是 distribution mismatch，但没给后续方案。如果把 RL 数据扩到双语，效果可能会更好——这个坑其实留给社区去填了。

### 对工程的几个启发

如果你也在做 Agent 训练（不限于 Deep Research），这几个 takeaway 我觉得是直接可借鉴的：

1. **数据清洗的 ROI 比加数据高**——10K 原始轨迹清洗 + 重采样得到 18745 条 SFT 数据，效果就足够碾压 30B baseline。在小数据 setting 下，**数据质量是首要瓶颈**。

2. **长轨迹重采样是免费午餐**——就是改个 sampling weight，不改算法、不加数据，BrowseComp 就涨 4 个点。任何 long-horizon 任务都该试一下。

3. **Long-horizon RL 必须用 dense turn-level reward**——GRPO 在 200 turn 上的失败已经很清楚了，下次再做长程 Agent RL，IGPO 应该是默认选项之一。哪怕不用 IGPO，也得想办法引入 turn-level signal（process reward、state value、information gain，都行）。

4. **搭 test-time scaling 是小模型部署的关键**——4B 模型 Pass@1 看着平庸，Pass@16 直接对标 GPT-5 High。如果你的产品场景能接受 16× 的推理预算（比如 batch 推理、离线分析），用小模型 + 多次采样可能是最划算的方案。

### 一个更本质的追问

DR-Venus 的成功背后有一个隐含命题：**Deep Research 是否被工业界系统性地"过设计"了？**

30B 模型 + 复杂 pipeline + 闭源数据，这套配置可能不是必要的，只是"我有资源所以我堆"的副产品。当一个 4B + 10K 数据的精心训练能逼近 30B 的水平时，意味着前面那些 30B 模型的价值是不是没充分被释放？或者说，**当前 Deep Research benchmark 的难度，是不是还没逼出大模型应有的能力上限？**

如果是后者，下一代 benchmark 可能要换思路了——不能只考"能不能找到答案"，可能还要考"信息源的可信度判断"、"跨语言交叉验证"、"识别和处理矛盾证据"这些更深层的能力。否则 4B 模型可能真的就够用了。

---

如果你也在做端侧 Agent、小模型 Deep Research、或者长程 RL，这篇 paper 值得通读一遍源码——不是为了学某个 trick，而是看它怎么把"小模型不可能"这件事一步一步证明成"可能"的。

更广义地说，这种"小数据 + 算法精修 + 小模型"的路径，可能是未来一段时间内值得重点关注的方向。当大模型的 scaling law 收益开始边际递减、训练成本继续指数级上涨时，回头去打磨数据质量、reward 设计、长程稳定性这些"老问题"，反而更容易出真正能落地的成果。这篇 paper 至少给了一个具体的样板——你不一定要做 Deep Research，但它的方法论是迁移得动的。

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
