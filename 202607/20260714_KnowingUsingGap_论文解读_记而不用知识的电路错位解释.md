---
title: 知识记住了却用不出来：LLM微调的"记而不用"是个电路错位问题
date: 2026-07-14
arxiv: 2607.08393
short_name: KnowingUsingGap
---

你有没有过这种体验：背单词的时候单词书翻了好几遍，单拎出来认得，但一放到阅读里就抓瞎？

微调大模型注入新知识这件事，逻辑几乎一样——模型很快就能"记住"新事实（比如"Sydney 位于 Australia"），但一旦换个问法问"the capital of the country where Sydney is located"，它就答不上来。准确率从 1.0 掉到 0.2，中间差了 0.8 个点；时间上记忆用 5.9 个 epoch 就饱和，泛化要拖到 10.1 个 epoch 才到位。这条 gap 在不同模型、不同数据规模、不同领域上稳定复现。

这篇 arXiv 论文 `2607.08393` 把这个现象正式命名为 **Knowing–Using Gap（知识使用差距）**，并给出了一个让我觉得"对，就该这么查"的诊断工具 **self-patching**——直接探测模型内部"知识存在哪、推理要用哪、这两者对不上"。最后顺手提了一个不用搜的固定启发式策略，能把 58–75% 的 oracle 头部空间拉回来。

## 核心摘要

论文作者来自 HKUST（广州）、University of Illinois Urbana-Champaign（UIUC）等机构（Lu Dai、Ziyang Rao、Yili Wang、Hanqing Wang、Hao Liu、Hui Xiong）。他们形式化了"知道却用不出来"这一现象，并提出一个看起来很自然的机制假说：**knowledge-circuit misalignment**——新事实被编码到模型中"容易存"的层（早期或最晚期），但推理电路活跃在中间层，两者没接上。作者用 self-patching 这一干预技术画出了微调过程中知识在层之间的"渗透图"（permeation map），可视化了这个错位过程。作为副产品，他们设计了一个仅用两个固定层对的简单策略，无需搜索就能在 Chaining 任务上把泛化准确率从 0.12 拉到 0.36，恢复 75% 的 oracle 头部空间。

坦白讲，这篇不是新方法论文，**是诊断 + 机制论文**。它解决的是"为什么 LLM 微调泛化差"这个老问题，但给了一个比"加数据、换 LoRA、调学习率"高一个解释层级的答案——是路由问题，不是容量问题。值得做 RAG、Agent 训练、持续学习的人仔细读。

---

## 论文信息

- **论文**：Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Finetuning
- **arXiv**: [2607.08393](https://arxiv.org/abs/2607.08393)
- **作者**：Lu Dai, Ziyang Rao, Yili Wang, Hanqing Wang, Hao Liu, Hui Xiong
- **机构**：HKUST（Guangzhou）、UIUC 等
- **日期**：2026 年 7 月 9 日提交
- **领域**：cs.AI, cs.CL

---

## 一、问题：记而不用

### 1.1 一个让人崩溃的实验现象

先上图。

![图1：Knowing-Using Gap现象示意](https://arxiv.org/html/2607.08393v1/x1.png)

*图1：Knowing-Using Gap 的三重刻画。(a) 真实微调动态曲线，记忆（蓝）早早冲到 1.0，泛化（粉红）在 epoch 15 才爬到 0.2，中间至少差 0.8 个点，且滞后 4-8 个 epoch。(b) 概念示意：模型记住了 "Sydney located in Australia" 和 "Australia has capital Canberra" 两条事实，但被问 "the capital of the country where Sydney is located" 时答不上。(c) 在 1000 个 case 上的饱和 epoch 统计，1st Memorize=5.9±2.8，2nd Memorize=10.1±2.8，Chaining=15.2±6.7，泛化最难。*

这条 gap 不只是"模型差一点"，它是**结构性的、可复现的**。论文在 6 个模型、2 个领域、4 个数据规模上验证了它。我把关键的"gap 强度"数据列一下：

| 维度 | 现象 |
|---|---|
| 数据规模 (10→10000) | 记忆始终 1.0，泛化从 0.4 掉到 0.08（**数据越多泛化越差**）|
| 模型规模 (1B→8B) | 记忆始终 0.99，泛化从 0.10 升到 0.18（**大模型稍好，但 gap 依然巨大**）|
| 模型架构 (LLaMA / Qwen) | 全部存在 gap，差异在 5% 以内 |
| 任务类型 (Chaining / Intersection) | Chaining gap 大（0.2），Intersection gap 小（0.1-0.2）|

最后一点特别值得划重点：**数据规模增大反而让泛化准确率变差**。

![图2：数据规模与模型规模的影响](https://arxiv.org/html/2607.08393v1/figures/datascale.png)

*图2 左：训练集从 10 扩到 10000，Chaining 准确率从 0.4 跌到 0.08，记忆稳定在 1.0。*

![图2右：模型规模的影响](https://arxiv.org/html/2607.08393v1/figures/modelscale.png)

*图2 右：模型从 1B 升到 8B，Chaining 从 0.10 升到 0.18，记忆稳定在 0.99。*

这是反直觉的——按理说"加数据"应该让模型学得更好，但在这里，**记忆和泛化走的是两条不同的曲线**。这不是数据质量问题，而是知识编码机制的问题。

### 1.2 形式化定义

作者给出了两个量化指标。设 $\theta_t$ 是训练 $t$ 步后的模型参数，$\mathcal{K}=\{f_i\}_{i=1}^n$ 是注入的事实集。

**Accuracy Gap**（饱和时泛化和记忆的差）：
$$\Delta A(\mathcal{T}) = A_{\text{mem}}(T_{\max}) - A_{\text{gen}}(T_{\max}; \mathcal{T})$$

**Temporal Lag**（泛化饱和相对于记忆饱和的滞后）：
$$\Delta T(\mathcal{T}) = T_{\text{gen}}(\mathcal{T}) - T_{\text{mem}}$$

其中 $T_{\text{gen}}$ 是泛化准确率达到 1.0 并保持稳定 $w$ 个 epoch 的最早时间点。Table 3 里的实测：LLaMA-3.1-8B 上，FFT 训练 Chaining 任务，记忆 2.4 epoch 到位，泛化要 7.9 epoch，**滞后 5.5 个 epoch**。

顺带说一句，作者在 Llama-3.2-8B（注意，是 3.2 不是 3.1）上验证了 temporal lag 对饱和阈值 $w$ 和成功率阈值 $\tau$ 的敏感性，在 $\tau\in\{0.9,0.95,1.0\}$ 和 $w\in\{1,2,3\}$ 的 9 网格上模式都成立。看起来结论是稳的。

---

## 二、方法核心：Self-Patching 把"知识存在哪"画出来

### 2.1 诊断思路

既然 gap 稳定复现、且与数据/模型规模无显著缓解，那问题就出在**机制层**——模型内部到底发生了什么？

作者没有用 attention 可视化、也没有用神经元级 probing，而是借了一个更直接的因果干预工具：**activation patching**。它的逻辑是：

> 如果某事实 $f$ 已经在模型内部某层被表示，那么把这一层的表示**替换**到目标推理任务对应位置，看推理能不能跑通——如果能，说明知识"在场"，只是没被正确路由。

作者把这个思路扩展到**跨层**和**跨上下文**，称之为 **self-patching**。

### 2.2 与同类方法对比

![图3：三种因果干预方法对比](https://arxiv.org/html/2607.08393v1/x2.png)

*图3：Causal Tracing（左）需要"干净"正确轨迹腐蚀后恢复，PatchScope（中）把隐藏状态解码成自然语言来解读含义，Self-Patching（右）直接跨层跨上下文交换表示，看推理能不能用。Self-Patching 胜在：(1) 不需要 clean reference run；(2) 关注目标任务的准确率提升（ΔAcc），不是可解释性；(3) 能扫所有 (l_src, l_tgt) 对，生成渗透图。*

Causal Tracing 是 Meng et al. 的 ROME 路线，要先有一个"正确的清洁运行"做参照；PatchScope 是 Ghandeharioun et al. 提出的"把隐藏状态当成 prompt 解码"路线。Self-Patching 跟它俩的关键区别是**只关心插上去能不能让答对，不关心插上去表达了什么语义**——这恰好是我们要回答的问题。

### 2.3 算法流程

Self-Patching 的输入是：模型 $M$、源提示 $P_s$（已让模型答对的事实行）、目标提示 $P_t$（需要用这条事实的推理题）、头实体锚点 $E$、正确答案 $y^*$、评分函数 $\mathcal{I}$（log-likelihood 差）。

对每一对层 $(l_s, l_t)$：
1. 在 $P_s$ 中定位头实体的 token 位置 $T_s$，缓存 $l_s$ 层的隐藏状态 $z=h^{l_s}_{T_s}(P_s)$
2. 在 $P_t$ 中定位 $T_t$，把 $P_t$ 跑到 $l_t$ 层后用 $z$ 替换 $h^{l_t}_{T_t}$
3. 继续前向得到 $\tilde{M}(P_t)$，计算 $\Delta I = \mathcal{I}(\tilde{M}(P_t), y^*) - \mathcal{I}(M(P_t), y^*)$

输出是一张 $(l_s, l_t)$ 的**渗透图**（permeation map）$A[l_s, l_t]=\Delta I$。颜色深红 = 完全恢复，深蓝 = 无效。

### 2.4 渗透图揭示的真相

作者把这套工具沿训练 epoch 跑了一遍，画出**训练过程 + 渗透图的联合演化**。

![图4：失败案例与成功案例的渗透动力学对比](https://arxiv.org/html/2607.08393v1/x3.png)

*图4：上图是自然失败案例（Probenecid→Vertigo），下图是成功案例（SLC30A8→Decreased circulating copper concentration）。每行从左到右是训练推进，红色区域表示"把 l_src 的表示 patch 到 l_tgt 能修复推理"。失败案例的红区在早期就出现，但**始终没有扩展到对角线**——意味着知识存在但没"渗透"到推理发生的位置。成功案例的红区在 epoch 20+ 覆盖到对角线。右侧是该 case 的 Memorization Task 和 Generalization Task 示例。*

这个图信息量极大，我多讲两句：

**核心观察**：
- 训练一开始，记忆准确率就冲到 1.0（蓝色曲线），loss 急剧下降；
- 但渗透图上能"修复"泛化的红色区域，仅在某些层对 $(l_s, l_t)$ 上出现；
- 失败 case 的红区在**对角线**附近但就是够不着对角线，意味着"知识在 $l_s$ 存着，但推理的因果流在 $l_s+1$ 层就丢掉了"；
- 成功 case 的红区**最终跨过对角线**，知识跟着推理流走通了。

**作者的核心假说（knowledge-circuit misalignment）**：

> 注入的新事实被编码在面向**存储**的层（通常是早期层和最晚期层），这些位置支持直接回忆，但**没有可靠地路由到多步推理因果所需的有效位置**（通常在中间层）。所以记忆满分、推理不及格。

这不是容量问题——模型记住了，也不是注意力问题——证据在正确位置，是**路由没接上**。

### 2.5 "两个簇"现象

渗透图虽然好看，但每个 case 都跑一遍 $(l_s, l_t)$ 扫描太贵了。作者把所有"有效 patch 点"投到 $(l_s, l_t)$ 坐标里，看空间分布。

![图5：有效 patch 位置的双簇分布](https://arxiv.org/html/2607.08393v1/figures/position_distribution.png)

*图5：6 个模型 × 2 个任务 = 12 个 panel，X 轴是源层索引，Y 轴是目标层索引。每个点是一个 case 的"最佳 patch 对"。**清晰的双簇结构**：一簇源层在 ~0.1L（早期），另一簇源层在 ~0.85L（晚期），**两者都打到 ~0.45L（中间层）**。所有模型、所有任务都呈现这个模式。*

这个发现漂亮的地方在于**跨架构一致性**。Qwen 和 LLaMA 内部结构、训练目标、tokenizer 都不同，但有效 patch 对几乎都聚集在"early→mid"和"late→mid"两个簇上。说明这不是某个模型的特殊行为，而是**微调注入知识的标准错位模式**。

作者还做了一个跨上下文一致性验证（Figure 6）：把 Memorization Prompt 里 head-entity 的表示 patch 到 Generalization Prompt，对比 Patching from Memorized Fact 和 Patching from Generalized 两者的"最佳 patch 对"分布——散点图沿对角线分布，相关性强。说明 self-patching 转移的就是注入知识本身，不是某个未知的噪声信号。

---

## 三、实验结果

### 3.1 Self-Patching 作为 Oracle 诊断

先把 6 个模型在两个领域（STaRK-Prime 生物医学 / STaRK-MAG 学术）的 oracle self-patching 结果摆出来（Table 4 简化版）：

| 模型 | Mem. | Chaining w/o pat. | Chaining pat. | Intersection w/o pat. | Intersection pat. |
|---|---|---|---|---|---|
| Qwen-2.5-1.5B | 0.998 | 0.078 | **0.440** | 0.793 | **0.987** |
| Qwen-2.5-3B | 0.997 | 0.114 | **0.542** | 0.798 | **0.986** |
| Qwen-2.5-7B | 0.996 | 0.124 | **0.504** | 0.774 | **0.956** |
| LLaMA-3.2-1B | 0.994 | 0.102 | **0.316** | 0.874 | **0.975** |
| LLaMA-3.2-3B | 0.993 | 0.126 | **0.404** | 0.815 | **0.969** |
| LLaMA-3.1-8B | 0.986 | 0.182 | **0.458** | 0.795 | **0.921** |

数字读法：每行第 3 列是"不 patch 时的 Chaining 准确率"，第 4 列是"用 oracle 找到的最佳层对 patch 后的 Chaining 准确率"。

**最狠的提升是 Qwen-2.5-1.5B 的 STaRK-Prime：0.078 → 0.440，5.6 倍**。1.5B 的小模型都能恢复这么多，验证了"知识在场只是没路由到"这个诊断——它是路由问题，不是容量问题。

McNemar 检验下，Mem. - Chain. 的差距全部显著（$p<10^{-50}$）。数据上没水分。

### 3.2 消融实验：Patching 的有效性来源

Table 6 里作者排除了几种"伪解释"：

| 模型 (STaRK-Prime) | w/o pat. | CoT | Irrelevant pat. | Self pat. |
|---|---|---|---|---|
| Qwen-2.5-1.5B (Chain.) | 0.078 | 0.132 | 0.150 | **0.440** |
| Qwen-2.5-3B (Chain.) | 0.114 | 0.288 | 0.184 | **0.542** |
| Qwen-2.5-1.5B (Inter.) | 0.793 | 0.793 | 0.243 | **0.873** |

- **CoT（思维链提示）**：从 0.078 涨到 0.132，有用但远不够；
- **Irrelevant patching**：用无关事实的表示去 patch，反而比 baseline 还差（0.078 → 0.150 是个小涨但 p 值显著低于 self patching）；等等，Chain 上 Irrelevant 0.150 比 CoT 0.132 高？这个细节值得注意。可能是无关表示里碰巧混入了与目标重叠的语义信息，Intersection 上 Irrelevant 0.243 比 baseline 0.793 显著差，说明"无关表示确实不携带目标知识"；
- **Self patching**：唯一能突破到 0.4+ 水平的方案。

Table 5 还做了 token 位置的消融：在 head-entity 处 patch，mean gain = 0.6408；在 `<EOS>` 处 patch = 0.4029；在 `<BOS>` 处 patch = 0.0485（不显著）。说明**在对的 token、对的层做 patch 才能生效**，不是位置无关的玄学。

### 3.3 Fixed Heuristic：不需要搜索就能用

Oracle 找到最佳层对 = 诊断上界，但实用上谁愿意每个 case 跑一次 $(l_s, l_t)$ 扫描？作者用 Figure 5 的双簇观察，直接定了**两个固定层对**：

- 层对 1：源在 $\lfloor 0.82L \rceil$，目标在 $\lfloor 0.45L \rceil$（late→mid）
- 层对 2：源在 $\lfloor 0.10L \rceil$，目标在 $\lfloor 0.45L \rceil$（early→mid）

每个模型架构就这两个固定值，不做 per-instance 搜索。效果（Table 7 平均）：

| 任务 | No Pat. | Fixed | Oracle | Fixed 恢复比例 |
|---|---|---|---|---|
| Chaining | 0.121 | **0.357** | 0.444 | **约 80 个百分点** |
| Intersection | 0.808 | **0.926** | 0.966 | **约 73 个百分点** |

论文里给的范围是 **58 到 75 个百分点**（按 75% 头部空间的口径），Chaining 任务上能拉到 80% 左右。

**这个数字挺关键的**。它意味着——只要知道了"知识储存在 early/late、推理发生在 mid"这个机制，我们就能用一个**完全免训练、零额外数据**的轻操作把泛化能力拉回一截。Chaining 任务 0.12 → 0.36 是 **2.95 倍**，对于不更新权重的部署场景是个非常实用的兜底。

---

## 四、我的判断

### 4.1 这篇论文到底牛在哪

第一，**gap 本身被系统量化了这件事**。过去大家都在抱怨"LLM 微调泛化差"，但很少有人把"差"拆成 accuracy gap + temporal lag 两条曲线、在 6 个模型、2 个领域、4 个数据规模上系统地跑出来。Table 2 的"模型 zero-shot 准确率 < 6%"还顺手解决了"是不是预训练已经见过"这个常见怀疑。

第二，**诊断工具够直接**。Causal Tracing 和 PatchScope 都是借用方法，但它们各有各的痛点（Causal Tracing 需要 clean reference run，PatchScope 关心可解释性不关心下游任务）。Self-patching 把"我能不能用"当成唯一指标，绕过了所有"语义解释"陷阱。

第三，**发现够普适**。Figure 5 的双簇在 6 个模型、2 个任务、2 个领域上一致出现。LLaMA 早期存得多一点、Qwen 晚期存得多一点，但整体结构稳。这不是某个 corner case 的怪现象，是**微调注入新知识的通用错位模式**。

第四，**副产品有工程价值**。Fixed Heuristic（两个固定层对）能恢复 58-75% 的 oracle 头部空间，且**不需要训练**、**不需要搜索**。对于已经在跑微调 pipeline 的团队，这个 insight 至少能让你在评估"这个知识注入了没用"时多一个数据点——是训得不够，还是训的位置不对。

### 4.2 这篇论文的问题在哪

第一，**Oracle self-patching 的恢复本身就是上界**。作者也承认了（Appendix H.1）——self-patching 只在固定锚点位置干预，且新知识可能分布在多个位置或被冗余编码，因此估计可能**低估**了完全可恢复的 headroom。但反过来，0.44 / 0.97 这个数是不是又有点过于乐观？比如说，patch 一个错误的 token 位置可能反而给模型"提示"，这种效应没法完全排除。Table 5 里 `<EOS>` 位置的 mean gain 0.4029 不算低，说明位置敏感性可能比作者宣称的弱一些。

第二，**Fixed Heuristic 的"层对 0.82L/0.45L/0.10L"是 hardcode**。这些数字是从 Figure 5 的双簇聚类中读出来的，对 LLaMA 和 Qwen 各自适用，但**没验证在 MoE、DeepSeek 这类架构上还成立**。如果换一个完全不同的模型家族，0.82L 是不是还是要 0.82L？我有点怀疑。

第三，**理解和修复之间的距离不可忽视**。这篇论文主要在**事后**做事——训练完发现泛化差，再用 self-patching 诊断、再用 fixed heuristic 修复。它还没回答：**能不能在训练早期就预测哪些事实会卡在 Knowing-Using Gap 上**？如果能，那才是真正改变 RAG / 持续学习 pipeline 的时刻。作者在 Limitations 里也承认了这一点。

第四，**机制分析粒度到 layer-level**。作者自己也提了，更细到 attention head 或 MLP 子层可能会进一步精炼这个 hypothesis。我猜再往下挖，会发现"early 存、mid 算、late 取"这个模式在不同 head 上的具体分配，才是真的有趣的地方。

### 4.3 对工程实践的启发

如果你在做下面这些事，这篇论文值得花一个下午细读：

- **RAG + 微调混合架构**：当 RAG 检索到的事实需要被模型"真正使用"时，Knowing-Using Gap 提示我们，光把文档塞进 context 还不够，模型可能"读到了"但"没用上"。Self-Patching 这类工具能帮你诊断"是 retriever 烂，还是 retriever 烂"——到底是检索错了，还是检索对了但模型没路由。
- **持续学习 / 知识更新**：当你在 base model 之上持续注入新知识（agent、tool、domain knowledge），gap 会不会越积越大？尤其是 Chaining 任务上的 lag，作者的数据让人警惕。
- **LLM 评估**：评估微调效果时，光看 in-domain 准确率不够，必须看多步推理的下游任务。**如果只测 memorization 准确率，gap 是完全隐形的**。
- **可控生成 / 知识激活**：如果你想让模型"在想用某条事实时主动用上"，Self-Patching 的 fixed heuristic 给了你一个非常便宜的兜底方案——在推理时跑两次 forward pass，把 head-entity 的 mid-layer 表示 patch 一次，可能比加 prompt 工程更有效。

### 4.4 跟同期工作的对比

"Remembering but not using" 不是一个新概念，Ovadia et al. (2024)、Soudani et al. (2024)、Zhong et al. (2023)、Cohen et al. (2024)、Berglund et al. (2023) 都有过类似观察。但之前的工作大多是**现象描述 + 经验方案**（比如"换 LoRA"、"加 CoT"、"用 ICL 替代微调"），没有真正给出**机制解释**。这篇论文的"knowledge-circuit misalignment"假说，把现象从"模型烂"提升到了"路由对不上"的解释层级。

跟 mechanistic interpretability 领域的同期工作比，Self-Patching 的设计非常务实——不追求"理解模型在想什么"，只追求"验证我的干预有没有用"。这种工具化的 mechanistic interpretability，我个人更喜欢。

---

## 五、收尾

读到结尾你可能会问：**这知识它到底存在哪？**

答案很清楚：在模型里。在 early layer 和 late layer 里。模型在训练过程中**已经把它写下来了**。

那为什么"想用的时候用不上"？

答案也很清楚：**它没被接到推理电路上**。像一个把工具放进抽屉但忘了放在哪的设计师——东西都在那儿，但需要的时候摸不到。

作者接下来要做的事（也可能是这篇论文最值得 follow up 的方向），是把这个诊断从"事后修复"推进到"训练时干预"——比如设计一个 loss 或正则项，鼓励微调过程把新知识路由到 mid layer。或者更进一步：能不能**预测**哪些事实会卡在 Knowing-Using Gap，从而在训练数据选择阶段就避开？

工程上不一定要等作者后续工作。光 Figure 5 那张双簇图，就值得打印出来贴墙上。

---

## 参考资料

- 论文：arXiv:2607.08393, [Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Finetuning](https://arxiv.org/abs/2607.08393)
- 相关前期工作（"remembering but not using"现象）：Ovadia et al. 2024、Soudani et al. 2024、Zhong et al. 2023、Cohen et al. 2024、Berglund et al. 2023
- Mechanistic interpretability 方法：Causal Tracing (Meng et al., ROME)、PatchScope (Ghandeharioun et al.)
- 数据集：STaRK-Prime (生物医学, 来自 PrimeKG)、STaRK-MAG (学术, 来自 Microsoft Academic Graph)

---

觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我。
