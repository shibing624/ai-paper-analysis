---
title: "AREX：把验证当方向舵，深度研究 Agent 的递归自改进"
date: 2026-07-25
arxiv: 2607.21461
short_name: AREX
---

# AREX：把验证当方向舵，深度研究 Agent 的递归自改进

你有没有这种感觉——让一个 deep research Agent 去查一个跨多源、带约束的研究问题，它越搜越长，但越搜越像在原地打转？第 30 步 search 的时候，它可能还在回访第 5 步就已经被一篇博客否决过的候选。

这不是 Agent 不够努力。**是它不会「判断自己已经走到哪儿了」。**

BAAI 昨天扔出来的 AREX（[arXiv:2607.21461](https://arxiv.org/abs/2607.21461)），就是想正面解决这个事。它的核心思路说起来也不玄乎：**把 verification 当成研究方向，而不是事后过滤器**——每一轮 research 结束，Agent 自己做一次「按约束逐条审计」，得到一个带置信度的临时答案和未解决清单，然后用这份清单去驱动下一轮 research。10B 激活参数的 AREX-Base 在 BrowseComp、GAIA、xbench-2510、DeepSearchQA、WideSearch-en、HLE 这六个常被刷到爆的 deep research / 搜索增强推理基准上，整体和动辄几百 B 激活的对手打平甚至反超。

光看数字你可能觉得又是「又一篇 SOTA 论文」。但 AREX 的写法比它的成绩更值得聊聊。

---

## 核心摘要

**痛点**：deep research 任务往往要同时满足多个交叉约束。发现一个满足全部约束的答案很难（要在稀疏的搜索空间里多跳），但**验证一个候选答案是否能满足每条约束却相对容易**——这种「发现-验证不对称」是 deep research 任务的本质结构。现有 deep research Agent 大多用「搜得更长」来对冲，导致大量轮次其实是在重复已经被证据否决的方向。

**方案**：AREX 把递归自改进（Recursively Self-Improvement, RSI）框架显式拆成两层循环——**内研究循环**负责搜证据、构造临时答案；**外自改进循环**负责按约束逐条审计临时答案的置信度，决策是 Accept / Refine / Restart；外层还会把「保留下来的证据 + 未解决约束」打包成新一轮研究目标。配套引入**自主上下文更新（ACU）**工具，让 Agent 学会自己把交互历史压缩成「研究状态快照」。

**效果**：4B 密集版 AREX-Turbo 与 122B-A10B MoE 版 AREX-Base 在六个基准上稳定超过同级开源基线；AREX-Base 用 10B 激活参数在 WideSearch-en 拿到全场最高分 **82.0**，BrowseComp 82.5（仅次于 Kimi-K2.6 83.2 和 Gemini-3.1-Pro 85.9），GAIA 85.4，DeepSearchQA 89.9，HLE 52.4。

**一针见血**：这不是底层架构突破，是一次非常扎实的**工程范式升级**——把 verification 从「最后一道闸」提升为「方向控制器」，并把长程训练里的 credit assignment 问题用「关键步聚焦监督 + 步级 RL」这套组合拳打掉。值得细读。

---

## 论文信息

- **标题**：AREX: Towards a Recursively Self-Improving Agent for Deep Research
- **机构**：北京智源人工智能研究院（BAAI，AREX Team）
- **作者**：Shuqi Lu, Chaofan Li, Kun Luo（共同一作），Zhang Zhang, Hui Wang, Hongwang Xiao, Zheng Liu（项目负责人） 等 23 人
- **链接**：[arXiv:2607.21461](https://arxiv.org/abs/2607.21461)
- **模型**：基于 Qwen3.5，分别放出 4B 密集版（AREX-Turbo）与 122B-A10B MoE 版（AREX-Base）
- **代码/模型**：https://arex-research.com / https://huggingface.co/collections/BAAI/arex

---

## 为什么要再做一篇 deep research 论文

先说清楚背景：deep research Agent 这条线最近半年相当拥挤。MiroThinker 系列从 v0.1 一路迭代到 1.7 和 H1，提出过「交互式 scaling」当第三个性能维度；Tongyi DeepResearch、WebDancer、SearchR1、WebSailor、WebThinker、AFM、DeepMiner、ReSearcher……每个名字背后都是一家团队在研究同一个事：怎么让模型在浏览器里多跳搜索、多源验证，最终拼出靠谱的复杂答案。

这些工作普遍沿着「**更强的工具调用 + 更长的搜索轨迹**」这条主线去堆料。问题是，这条路线天然有结构性缺陷：

1. **方向纠错慢**。早期步骤搜到错误候选后，即使后面拿到了反例，模型也很难主动放弃这条线。
2. **预算分配不均**。好的 trajectory 里其实只有 10% 不到的步骤真正「决定胜负」，剩下都是 routine。GRPO 把整条 trajectory 的 advantage 平摊下去，信号被严重稀释。
3. **上下文管理粗糙**。128K context 满了就压缩，threshold-based summarization 会把决策相关的「反证、已否决候选、未解决约束」一起丢。

AREX 的作者对这三个问题都给了独立的对策——而它们的共同底色就是**「让 Agent 学会看自己的中间状态」**。

---

## 方法：双层循环 + ACU + 关键步训练

### 1. 整体框架：内循环做研究，外循环做审计

AREX 的骨架是一个 hierarchical bi-level RSI 过程：

![图 1：AREX 的递归自改进框架。蓝色外层循环负责按置信度决策 Accept/Refine/Restart；橙色内层循环负责具体研究与证据整合；中间是 ACU 触发的研究状态刷新。](https://arxiv.org/html/2607.21461v1/x2.png)

*图 1：AREX 的递归自改进框架。内研究循环维护研究状态、外化一个带置信度的临时答案；外自改进循环根据置信度与轨迹评估决定 Accept / Refine / Restart。Refreshed Research State 那一栏（Verified Findings、Current Candidates、Unresolved Constraints、Validity Concerns、Rejected Candidates、Next-Step Plan）是 ACU 留下的「研究快照」。*

具体怎么跑：

- 给定一个 query $x$，先生成一个 research objective $q^{(1)}$。
- **内研究循环**按 $q^{(k)}$ 跑：分析当前目标 → 调 search/visit 工具 → 拿到 observation → 更新内部轨迹 $h_t^{(k)}$。循环到一定步数或 evidence 充分时，调用 `finish` 输出**结构化结果 $r^{(k)} = (y^{(k)}, \mathcal{E}^{(k)}, s^{(k)})$**——provisional answer + supporting evidence + 一个 0–100 的答案级置信分。
- **外自改进循环**拿到 $r^{(k)}$，按规则做三选一：
  - $s^{(k)} \geq \tau$ → **Accept**，直接作为最终答案；
  - $s^{(k)} < \tau$ 且 trajectory 还可救 → **Refine**，把保留下来的 $\mathcal{P}^{(k)}$、待解决的 $\mathcal{I}^{(k)}$ 一起打包成下一轮目标 $q^{(k+1)}$，从刷新过的研究状态开始新内循环；
  - $s^{(k)} < \tau$ 且 trajectory 已经乱到没救 → **Restart**，丢弃当前轨迹，从原始问题 $x$ 重开。

整个过程被最大递归轮数（实验里是 5 轮）封顶；如果全跑完都没人能 Accept，就返回置信度最高的那个。

这跟我之前在调一些 deep research 流水线时很类似的体感——**与其在「下一个 query 该搜什么」上死磕，不如先认真问一句「我现在手里的东西，哪些是真的、哪些是猜的」**。AREX 的形式化其实就是把这个朴素的工程经验直接做进了 policy。

### 2. 自主上下文更新（ACU）：让 Agent 自己会「打草稿」

deep research 最头疼的工程问题是上下文管理。搜十几轮之后，prompt 里塞满了过期的中间结论、被否决的候选、互相打架的引用。MiroThinker 这类工作用的是 sliding window + 截断：粗暴但有效，只是会丢证据。AREX 选择让 Agent **自己**学会调用一个 `update_context` 工具：

$$z_t^{(k)} = f_\theta(h_t^{(k)})$$

也就是说，给定当前的完整轨迹 $h_t^{(k)}$，模型自主生成一个压缩后的「研究状态 $z$」。这个状态被刻意结构化成 6 个槽位：

- Verified Findings（已验证的发现 + 来源标识）
- Current Candidates（当前在跑的候选）
- Unresolved Constraints（未解决的约束）
- Validity Concerns（来源/时效/一致性的疑虑）
- Rejected Candidates（已否决的候选 + 否决理由）
- Next-Step Plan（下一步计划）

Agent 自主决定**何时**调用 update_context——不是到 128K 才被动触发。从后面 Table 2 的统计看，AREX 在 80.3% 的 BrowseComp 案例中都会主动调用，平均在 25,721 tokens 处触发，远低于 128K 的硬上限。绝大多数触发是「搜索策略要换」（66.9%）或「要否决一个候选」（13.6%）。

这个设计挺漂亮的：它不是把 context 当「预算问题」，而是把它当「**研究状态维护问题**」——保留决策相关信息、显式记录已否决项、给出下一步计划。下一轮内循环读 $z_\tau^{(k)} \oplus$ 后续步骤（式 5），相当于「从上一份草稿继续写」而不是「重读一遍所有搜索结果」。

### 3. 结构化 Finish：把 verification 做成一等公民

内循环的 `finish` 不是输出一个自然语言答案，而是输出一个三元组（答案 + 证据 + 置信度）。置信度的设计很有意思：它由模型自评，**反映证据的完整性、一致性、来源权威性和时效性**。

这一点对后面的训练非常重要——置信度既是外循环的决策信号，也是 RL 阶段做 credit assignment 的桥梁。Figure 3 的实验数据很能说明问题：开启 ACU 后，**95.9 个百分点**的正确答案落在 90–100 分区间，而**55.2 个百分点**的错误答案置信度低于 60。也就是说，置信度本身就是一个可用的「正确性概率代理」——不需要重新解释整条 trajectory，光看分就能 catch 掉超过一半的失败。

![图 2：ACU 开启/关闭时，置信度分布在「正确」与「错误」输出上的差异。开启 ACU 后，正确答案的 95.9% 集中在 90–100 分，错误答案的 55.2% 落在 60 分以下。](https://arxiv.org/html/2607.21461v1/x3.png)

*图 2：置信度分布在「正确/错误」输出上的对比。每根柱子在方法和结果组内归一化。ACU 显著拉开了「正确答案高置信、错误答案低置信」的距离，是外循环做 Accept/Refine/Restart 决策的关键依据。*

---

## 训练：Agentic Mid-Training + Step-Aware RL

光有推理框架不够，AREX 的训练 pipeline 也得聊——它的 ablation 实验表明，**每一个训练环节的贡献都比 inference 框架更具体**。

### 1. 多阶段 Agentic 中期训练

Long-horizon research 需要的能力是异质的：搜索/浏览、证据整合、长程推理、context tracking、答案合成。AREX 用三阶段训练来避免相互干扰：

1. **Browse-intensive 阶段**：喂大量多轮 browse trajectory，先把工具使用、网页导航、query 改写、证据获取这些基础能力学扎实。
2. **Reasoning-intensive 阶段**：换成专家级 reasoning 数据，强调长链思考、假设检验、答案选择。
3. **Mixed-capability consolidation 阶段**：把 browse 长程轨迹的关键步 + 学术论文 research + 知识密集 reasoning 一起混着训练，并显式把 verification-driven 转换（provisional answer → 审计 → 下一轮目标）作为训练内容。

第一阶段先把工具调好，第二阶段再加推理，第三阶段把两者「焊」在一起。Table 4 表明，把 1+2 阶段直接从「分阶段」改成「混合训练」会让 BrowseComp 从 82.5 掉到 77.5——5 个点，**阶段顺序确实有讲究**。

### 2. 关键步聚焦监督（Key-Step Focused Supervision）

这一步是整篇论文最工程化、也最有意思的设计之一。

直觉是：一条成功 trajectory 里，**绝大多数步骤都是 routine（普通搜索、格式化、确认状态），但有少数步骤决定了最终能不能做对**。这些关键步包括：

- **Evidence discovery**：找到第一个给出决定性证据的工具调用；
- **Path rejection and redirection**：第一次明确否决错误候选并改方向；
- **Key context-update**：调用 update_context 整理研究状态的时刻。

AREX 用基于规则的 detector 离线标注这些关键步（**只能用「确实拿到了外部证据」的关键步，不接受模型自报**），然后对它们做**只保留 prefix context、只对关键步本身计算 loss** 的精读式监督。

**为什么这样做？** 作者用 Figure 4 给了直接的证据：

![图 3：全 trajectory 训练后，ordinary steps 与三类关键步的平均 loss 对比。关键步 loss 高出 19%–29%，说明它们在全 trajectory 监督下被严重欠拟合。](https://arxiv.org/html/2607.21461v1/x4.png)

*图 3：全 trajectory 训练后，不同类型步骤的 loss 对比。普通步 loss 0.232，三类关键步分别为 0.277、0.298、0.300——相对涨幅 19%–29%。普通步被过度监督、关键步被欠拟合，loss 分布严重不均。*

这个数字很关键：**关键步 loss 比普通步高 19 到 29 个百分点**，说明全 trajectory 监督下，模型在最有价值的决策点上反而学得最差。把监督算力从「均匀分给所有 token」改成「集中砸在关键步」，是直接对应 loss 分布做的修正。

实验上，**把关键步聚焦监督换成等算力的 random-step replay，BrowseComp 从 82.5 掉到 74.1——8.4 个点，是论文里所有 ablation 里降幅最大的一项**。这个降幅说实话我有点震惊。它说明在 long-horizon agentic 训练里，「**对的地方、用对的 loss 算力**」比单纯加 token 重要得多。

### 3. 步级强化学习（Step-Aware RL）

最后一步是 RL。AREX 没有用 vanilla GRPO，而是做了两层改造：

- **Hierarchical normalization**：先在 trajectory 内对所有 assistant step 求平均，再在 rollout group 内对 trajectory 求平均。这避免「trajectory 越长、loss 越大」的 bias。
- **Key-step shaping**：在最终 trajectory reward 的 advantage 上，对被标注为关键步的 step 加一个有界 bonus $\lambda_{\mathrm{key}}$。注意 bonus **只在 trajectory 整体答对时**才生效，避免奖励「胡乱的中间行为」。

把步级 RL 换回标准 GRPO，BrowseComp 从 82.5 掉到 79.4（-3.1），比关键步监督的降幅小，但仍然是实打实的贡献。

---

## 实验：6 个基准、2 个模型规模

AREX 评估的基准覆盖四个方向：deep research（BrowseComp、DeepSearchQA）、agentic 任务（GAIA、xbench-2510）、wide search（WideSearch-en）、高难度推理 + 工具（HLE）。下表是完整数据：

| 类别 | 模型 | BrowseComp | GAIA | xbench-2510 | DeepSearchQA | WideSearch-en | HLE (tool) |
|------|------|-----------:|-----:|------------:|-------------:|--------------:|-----------:|
| Closed frontier | GPT-5.4 | 82.7 | – | – | 88.5 | 77.5 | 52.1* |
| Closed frontier | Opus-4.6 | 83.7 | – | – | 91.3 | 77.5 | 53.0* |
| Closed frontier | Gemini-3.1-Pro | 85.9 | 80.6 | 53.0 | 93.3 | 66.4 | 51.4* |
| Open source | GLM-5 | 75.9 | 70.0 | – | – | 69.8 | 50.4 |
| Open source | Kimi-K2.6 | 83.2 | 80.6 | 90.0 | 92.5 | 80.8 | 54.0* |
| Open source | DeepSeek-V4-Pro | 83.4 | – | 80.0 | 88.7 | 78.0 | 48.2 |
| Open source | Tongyi-DeepResearch-30B | 43.4 | 70.9 | 55.0 | – | – | 32.9 |
| Open source | Qwen3.5-397B | 78.6 | 83.5 | 61.0 | 82.1 | 74.0 | 48.3 |
| Open source | MiroThinker-H1 | 88.2 | 88.5 | 72.0 | 80.6 | – | 47.7 |
| Open source | MiroThinker-1.7 | 74.0 | 82.7 | 62.0 | 72.1 | – | 42.9 |
| **Ours** | **AREX-Turbo** (4B dense) | **70.7** | **81.6** | **57.0** | **78.5** | **68.5** | **40.6** |
| **Ours** | **AREX-Base** (122B-A10B) | **82.5** | **85.4** | **71.0** | **89.9** | **82.0** | **52.4** |

*HLE 列带 * 表示在完整 HLE 上跑，否则是 text-only 子集。*

![图 4：AREX 在六个基准上与同规模 / 更大规模模型的对比。AREX-Base（10B 激活）在 WideSearch-en 拿到全场最高 82.0，BrowseComp 82.5 紧追 Kimi-K2.6。](https://arxiv.org/html/2607.21461v1/x1.png)

*图 4：六个基准的横向对比。深蓝是 AREX-Base（10B 激活），浅蓝是 AREX-Turbo（4B dense）。在 4B 级别 Turbo 几乎在所有榜单上压过 Qwen3.5-35B；122B-A10B 的 Base 在与参数量数倍于自己的闭源 / 开源系统对比里也能正面对刚。*

值得拎出来说的几个点：

- **WideSearch-en 82.0 是全场最高**——比 Kimi-K2.6（80.8）还高 1.2，比 GPT-5.4（77.5）高 4.5。这是个**宽覆盖检索**任务，AREX 在「能搜多、能整合好」上的优势在这里充分体现。
- **BrowseComp 82.5** 紧追 Kimi-K2.6（83.2）、Gemini-3.1-Pro（85.9）、MiroThinker-H1（88.2）。MiroThinker-H1 的 88.2 是 16× 算力预算才拿到的（[他们的报告里写得很清楚](https://blog.csdn.net/shibing624/article/details/159420375)），AREX 用 10B 激活参数就拿到 82.5，单位算力性价比是另一档。
- **GAIA 85.4** 高于 Kimi-K2.6（80.6）、MiroThinker-1.7（82.7），仅次于 MiroThinker-H1（88.5）。
- **HLE 52.4** 略低于 Kimi-K2.6（54.0*），但显著超过 GLM-5（50.4）、Qwen3.5-397B（48.3）。注意 HLE 跑的是 text-only 子集，对比要小心。
- **4B 的 AREX-Turbo** 在 6 个基准里 5 个超过 Qwen3.5-35B，BrowseComp 70.7 vs 61.0 是最显著的一档。

---

## 消融：每一项都「具体可见」

AREX 把我最看重的「具体 ablation」做得相当漂亮。BrowseComp 上四组对照：

| 配置 | BrowseComp |
|------|-----------:|
| 完整 AREX | **82.5** |
| 替换 step-aware RL 为标准 GRPO | 79.4 (-3.1) |
| 替换多阶段 mid-training 为混合训练 | 77.5 (-5.0) |
| 替换关键步监督为 random-step replay | 74.1 (-8.4) |
| 去掉 ACU（保持 outer loop） | 71.4 (-11.1) |
| 去掉 outer loop（保持 ACU） | 69.8 (-12.7) |
| 都没有 | 59.6 (-22.9) |

把 inference 框架拆开来看，**ACU 单独贡献 +11.8（59.6 → 71.4），外自改进循环 +10.2（59.6 → 69.8），两者同时 +11.1（71.4 → 82.5）**——不是简单加法关系，是相互放大。

而训练侧，**关键步监督的 8.4 个点降幅是所有 ablation 里最大的**。这其实很反直觉——「把监督算力从关键步挪到 random step」理论上不该掉这么狠，但实际表现就是「训错地方」远比「训得少」更糟。配合 Figure 4 的 loss 分布数据，这个结果非常自洽。

---

## 我的判断

**亮点**：

- **Idea 本身不复杂，但工程化很扎实**。把 verification 当 transition operator 这个想法并不新（test-time search、self-refine、verifier-as-reward 都摸到过），但 AREX 的形式化做得足够干净——`finish` 输出 (y, E, s) 三元组、外循环的 Accept/Refine/Restart 三态决策、ACU 的六槽位结构，都是直接可实现的工程原语。
- **ACU 的 80.3% / 25K tokens 这个数据**是 paper 里我最喜欢的数。说明 Agent 真的「学会了什么时候该打草稿」，而不是被动等到爆 context。置信度分布（Figure 3）也呼应得非常好——ACU 让正确答案的置信度分布「更尖」，让错误答案的置信度分布「更靠近 0」，可分性大幅提升。
- **关键步监督 + 步级 RL** 是真正「做了别人没认真做的工程」。Long-horizon agentic 训练的 credit assignment 是公认痛点，AREX 用规则 detector + loss 分布分析 + 分级 bonus 给出了一种**低成本可复现**的解法。

**值得吐槽的地方**：

- **HLE 跑的是 text-only 子集**（* 标注），和 Kimi-K2.6 的 54.0* 表面上接近，但口径是否完全一致存疑。Table 1 里 AreX-Base 的 HLE 是 52.4、Kimi-K2.6 是 54.0*，但 OpenAI 系列在 HLE 上也只给到 51–53，**HLE 这个榜本身在多模态题目上的差异会让对比显得微妙**。论文里没把 HLE 的具体子集分布和评测 protocol 写得很细。
- **Tongyi-DeepResearch-30B 在 BrowseComp 上只有 43.4**——这个数字和 Tongyi 系列平时表现差距很大。我怀疑是 30B 这个特定 checkpoint 在 BrowseComp 这个特定子集上被某些 scoring 因素压住了，**但论文没解释为什么不用更主流的 Tongyi DeepResearch 版本**。这让对比表对「我应该选谁」这个问题的指向性变弱。
- **和 MiroThinker-H1 的对比**严格说是不公平的。H1 是用 16×–64× 算力预算拿的 88.2，AREX-Base 应该是默认预算的 82.5，**单位算力的性能曲线上谁更陡，论文里没给 token-scaling 图**。这是 deep research 这个赛道的标准评测缺口，不光是 AREX 的问题，但作为读者会希望看到。
- **Outer self-improvement loop 的最大 5 轮封顶**和「如果都没 Accept 就取置信度最高」这个回退策略，听起来简单，但在压力测试下会不会出现「乱 Restart 浪费算力」的情况？论文没分析每轮决策的分布。

**对工程的启发**：

如果你也在搭 deep research 流水线，**最值得抄的是 ACU + 关键步监督这两个组件**——它们不依赖任何花哨的框架，就是 prompt 格式 + 训练 loss 设计的调整。ACU 让你用 25K tokens 的有效 context 跑完 30 步 search，关键步监督让你训 4B 模型时不被「无效步骤」淹没训练信号——这两招比任何 agent 框架升级都更普适。

至于外层 RSI 循环，得看你的工程成熟度——它要求 Agent 在 `finish` 时给出**真实可信的置信度**和**结构化证据**，对底层模型的能力有要求。如果你的 base model 在自我评估上还不太行，盲目套 RSI 可能会放大错误。

---

## 收尾

AREX 不是那种会让人拍桌子喊「idea 怎么这么天才」的论文。但它是我最近看到的、把「deep research Agent 该怎么迭代」这个事**想得最透、做得最系统**的一篇：

- 推理端，把 verification 从「最后一道闸」提升为「方向控制器」；
- 训练端，把监督算力从「均匀分给所有 token」聚焦到「真正决定胜负的关键步」；
- 数据端，用合成任务 + 质量过滤控制 trajectory 噪声。

如果让我给一个数字来概括 AREX 的核心贡献——**10B 激活参数、6 个基准、22.9 个绝对点的外消融涨幅**。这是一个「每多投入 1B 激活算力、回报都非常清晰」的工程范式。

至于它和 MiroThinker-H1、Kimi-K2.6 这些同期的 SOTA 怎么比？我的建议是别光看绝对分——**先看 token-scaling 曲线和单位算力性价比，再决定跟谁抄作业**。deep research 这个赛道，现在比的是「同样算力下谁更稳」，而不是「谁能无脑堆」。

如果你最近也在调 deep research Agent，强烈建议把这篇的 ACU 设计（结构化 6 槽位 update_context）和关键步 detector 抄过去——成本极低，但提升立竿见影。

---

觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。
