# LLM 读表格总抄错数：被忽视的「数据引用错误」

> 论文：*When LLMs Read Tables Carelessly: Measuring and Reducing Data Referencing Errors*
> 链接：https://arxiv.org/abs/2606.32029
> 作者：Yuqing Yang¹、Qi Zhu²、Zhen Han²、Boran Han²、Zhengyuan Shen²、Shuai Wang²、Vassilis N. Ioannidis²、Huzefa Rangwala²
> 机构：¹University of Southern California，²AWS AI Labs
> 会议：ACL 2026 Oral
> arXiv ID：2606.32029

---

## 核心摘要

如果你让 GPT 或者 Qwen 回答"哪一年 XXX 比赛的总冠军是 YYY"，它大概率给你一个看着挺顺的答案——但表格里可能压根没这个冠军，或者答案引用错了列。这种错误有个新名字：**Data Referencing Errors (DREs)**，即「数据引用错误」。这篇 ACL 2026 Oral 的论文是第一次系统性地把这件事摆到台面上：他们在 1.7B 到 20B 的九个模型上测了一圈，发现 DRE 在所有模型里都普遍存在，5.71% 到 46.48% 不等——模型理解表格结构是没问题，但抄数的时候就是会抄错、抄漏。

更关键的是，他们顺手训练了一个 4B 的小裁判模型（Critic-4B），平均 F1 干到 **78.2%**，比未训练基线高 8.65 个点。这个小裁判加上拒绝采样，能让 Distill-Qwen-7B 在 TableBench 上的准确率从 54.77% 拉到 66.73%——**+12 个点**，对应到 DRE 子集上更是 +15.27 个点。代价？只需要一个 4B 模型跑一遍裁判。论文摘要里那个"up to 12.0%"就是这么来的。

我读完的最大感受是：**这件事居然是 2026 年才被系统提出来**。我们做表格问答的时候天天看到模型胡诌一个数字，原来一直没人正经量化过这个错误类型。

---

## 第一个问题：什么叫「数据引用错误」？

作者在论文里把 DRE 定义得很干净：在 chain-of-thought 响应里，"数据引用"（data referencing）就是模型**从表格中正确定位并抄写信息**的能力；这一步出的错就叫 DRE。他们进一步把 DRE 拆成两类：

| 类型 | 含义 | 典型表现 |
|---|---|---|
| **Incorrect Citation（错误引用）** | 抄错了值、抄错了列、或者直接编了一个 | 把"Organization"列抄成"Award"列里的内容 |
| **Omitted Information（信息遗漏）** | 需要的子集没抄全，漏行了 | 表格有 20 行，只识别出 19 行 |

下图是 Figure 1 里的两个例子，是 DRE 最经典的两种姿势。

![Figure 1(a)：错误引用示例](https://arxiv.org/html/2606.32029v1/x1.png)

*Figure 1(a)：错误引用。问题是"2008 之后下一个列出的奖项是什么"。表格里 2010 年 Organization 列是 "Nikkan Sports Grand Prix (Fall)"，Award 列才是 "Best Supporting Actress"。模型把 Organization 抄成了 Award，给出答案"下一个奖项是 Nikkan Sports Grand Prix (Fall)"——抄串列了。*

![Figure 1(b)：信息遗漏示例](https://arxiv.org/html/2606.32029v1/x2.png)

*Figure 1(b)：信息遗漏。问题是"10 月总共有几个节日"。表格里 Oct 17/Oct 20/Oct 23 都在 10 月，模型把 Oct 23 整行漏掉，只数了 Oct 17/Oct 20 和 Nov 11。问的是 10 月，答的是"两个"——漏了。*

你看这两个例子有个共性：**模型对表格结构是懂的**，问题也读对了，最后答错纯粹是"抄"那一步出了岔。我自己在做表格问答评测时确实经常碰到这种——尤其金融表格，列名长得像"Adj. Revenue FY2024 Q3"，列与列之间区分度低，模型"看清"但"抄错"非常常见。

---

## 这件事为什么之前没人发现？

作者在引言里坦白了一件事：之前的工作只在小规模、人工标注上观察到 DRE，没人系统量化。根本原因是**最终答案准确率这个指标太宽容了**。

举个真实例子（图 Figure 5 在附录里给出来的）：Qwen3-8B 在 SciTab 上把 `0.714` 抄成了 `0.704`，但因为后面 Wmd-2 = 0.763 更高，结论不变，最终答案还是对的。换句话说，**最终答案对，不等于中间推理对**。如果你的系统是给分析师做报告中间态展示的，这种"过程错误但结果正确"就是定时炸弹。

所以这篇论文的核心论点其实是：**我们需要一个独立于最终答案的中间过程评测维度**。这个观察不算新，但 DRE 这个名字是新的，工具也是新的。

---

## 实验规模：1.7B 到 20B，九个模型全测了一遍

为了保证结论站得住脚，作者拉了一个相当扎实的评测矩阵：

**模型清单**（覆盖推理、非推理、蒸馏、表格专用、MoE 等多种类型）：

- Qwen3-1.7B / 4B / 8B（推理模型）
- Qwen2.5-7B-Instruct（标准 LLM）
- Table-R1-Zero-7B（专门做表格 RL 训练的）
- Distill-Qwen-7B / Distill-Llama-8B（蒸馏推理模型）
- Llama4-Scout（MoE）
- gpt-oss-20b（OpenAI 开源推理模型）

**数据集**（覆盖 5 类任务）：

| 任务 | 数据集 |
|---|---|
| 表格问答 | WTQ、TableBench、FinQA |
| 声明验证 | SciTab |
| Table-to-Text | ToTTo |

**评判模型**：Sonnet-3.7+gt（在 judge prompt 里塞 ground truth 答案，让人手验 92.67% 准确）。这个设计很关键——不塞 gt，judge 容易被模型响应对齐带跑偏（Figure 8 给了例子）。

---

## 主要发现：DRE 普遍存在，但程度差异巨大

下面是 Table 1 的核心数据（Qwen3-8B 在 WTQ 上 + 各模型在 WTQ 上的对比）。

**Qwen3-8B 在不同数据集上**：

| 数据集 | Accuracy | DRE Rate | DRE-in-Incorrect | Correct-in-DRE |
|---|---|---|---|---|
| WTQ | 77.14% | 14.04% | 32.63% | 46.89% |
| WTQ + prompting | 77.51% | **12.50%** | 28.76% | 48.25% |
| WTQ (CSV 格式) | 75.94% | 17.54% | 37.32% | 48.82% |
| WTQ (Markdown 格式) | 77.26% | 14.34% | 32.69% | 48.15% |
| TableBench | 77.48% | 10.55% | 30.63% | 13.43% |
| FinQA | 63.21% | **33.57%** | 39.34% | 56.88% |
| SciTab | 77.53% | 14.06% | 21.54% | **65.57%** |
| ToTTo | – | 14.06% | 18.45% | – |

**不同模型在 WTQ 上**：

| 模型 | Accuracy | DRE Rate |
|---|---|---|
| Qwen3-1.7B | 57.76% | 35.52% |
| Qwen3-4B | 75.69% | 16.18% |
| Qwen3-8B | 77.14% | 14.04% |
| Qwen2.5-7B-Instruct | 43.32% | 17.56% |
| Table-R1-Zero-7B | 76.10% | 19.29% |
| Distill-Qwen-7B | 49.47% | **46.04%** |
| Distill-Llama-8B | 59.78% | 37.96% |
| Llama4-Scout | 55.71% | **46.48%** |
| gpt-oss-20b | 78.38% | **5.71%** |

几个我看完心里一沉的点：

1. **Table-R1-Zero-7B 这个专门为表格 RL 训练过的模型，DRE Rate 居然有 19.29%**。比 Qwen2.5-7B-Instruct（17.56%）还高。说明"专门训练"这件事没把"抄对"给解决掉，模型可能在优化最终准确率时绕过了引用错误。

2. **Correct-in-DRE 在 SciTab 上达到 65.57%**——意思是每 3 个有 DRE 的响应里，有 2 个最终答案还是对的。这跟前面的 Figure 5 例证互相印证：**准确率高不代表过程干净**。

3. **prompt 注入对 DRE 几乎没用**。作者实验了在 prompt 里加"请不要误引或漏抄"这种明确指示，WTQ 上的 DRE Rate 从 14.04% 降到 12.50%，但准确率几乎没动（77.14% → 77.51%）。说明模型不抄错这事，靠"提醒"没用。

4. **gpt-oss-20b 是唯一 DRE Rate 在 10% 以下的**（5.71%）。它本身是 OpenAI 那波"推理强化"的新模型，可能在内部已经针对这种错误做了对抗——但论文没展开说为什么 gpt-oss 这么好，这个我觉得是值得追问的点。

5. **格式有影响但不大**：CSV 格式把 DRE Rate 从 14.04% 推到了 17.54%，Markdown 反而和 JSON 接近。说明模型不是"看"不懂格式，是"抄"那一步容易在列分隔符上出错——这点对工程实践有直接启发：复杂金融表别用 CSV。

---

## 怎么修：Critic-Based Filtering + Rejection Sampling

既然 DRE 普遍存在，作者给出的解法非常直接：**再训一个 4B 的小模型当裁判**。两阶段训练：

**阶段一：SFT warm-up**。拿 2000 个 Sonnet-3.7 标注好的正负样本，1e-5 学习率，batch size 8 训 2 个 epoch。作者特别提到"不做 SFT warm-up 直接上 RL 会让模型疯狂重复输出 `<judgment></judgment>` 标签"——RL 训 LLM 的格式不稳定这事是公认痛点，他们这步很务实。

**阶段二：RLVR (GRPO)**。5712 个样本，batch size 256，每 prompt 8 rollouts，温度 1.0，1e-6 学习率训 20 个 epoch。工具是 verl。

**训完的产物**：Critic-4B。基础是 Qwen3-4B-Instruct。

### 4B 小裁判的 F1 表现

下图是 Figure 2——9 个 model-dataset pair × 3 个 critic 的 F1 柱状图。

![Figure 2：不同 model-dataset pair 的 F1 分数](https://arxiv.org/html/2606.32029v1/x3.png)

*Figure 2：9 个 model-dataset pair（3 个数据集 × 3 个生成模型）上三个 critic 的 F1 分数对比。橙色是未训练的 Qwen3-4B-Instruct，绿色是 Critic-4B-Synthetic（用合成 DRE 样本训练），蓝色是 Critic-4B（用真实 Sonnet-3.7 标注训练）。*

最右边的 Overall 列最能说明问题：

- Qwen3-4B-Instruct（未训练）：约 69%
- Critic-4B-Synthetic：约 70%（某些场景涨、某些场景跌）
- Critic-4B：**78.16%**（论文里 78.2%）

涨了 **+8.65 个点**。这个数据让我有点意外——4B 的小模型做 DRE 分类居然能跑到 78% F1，而且是在它**只在 Qwen3-8B 生成的 WTQ 响应上训练**的前提下，跨模型（Qwen3-8B / Distill-Qwen-7B / Llama4-Scout）+ 跨数据集（WTQ / TableBench / FinQA）都能保持优势。这说明 DRE 这个错误模式本身是**跨模型可迁移的**——你抄错数、漏抄数这个习惯，模型之间是相通的。

另外一个值得说的细节：**Critic-4B-Synthetic 比 Critic-4B 差**，特别是在 FinQA + Llama4-Scout 这种跨域场景。原因是合成数据是按"四种固定模式"（Mix up rows/columns、Remove row、Remove a listed row）插入 DRE 的，这跟真实 DRE 的分布有偏差。**所以合成数据这条路没走通，作者最后用的是真实 Sonnet-3.7 标注**。这个我之前在别的项目也踩过类似坑——合成对抗样本通常看着像那么回事，但分布跟真实错例差太多。

### 用这个 4B 裁判能涨多少分？

**Critic-Based Filtering (CF)**：让模型采样 N=8 个响应，用 critic 挑 DRE 最少的子集，再做 majority voting。

以 Qwen3-8B / WTQ 为例：
- Avg Acc（N=8 平均）：64.59%
- CF Acc（critic 筛选后）：70.44%
- MV Acc（majority voting）：70.84%
- **CF + MV Acc：73.49%**

比直接 majority voting 多涨了 2.65 个点，验证了 critic 筛选和 majority voting 是**互补**的。

**Rejection Sampling (RS)**：在段落级别重新采样，丢掉 DRE 段落。

下面是 Table 3（用 Sonnet-3.7+gt 作为 critic）和 Table 4（用 Critic-4B 作为 critic）的核心数据对比。我把最值得看的几行挑出来：

**Distill-Qwen-7B 在 TableBench 上**（论文里涨幅最大的场景）：

| 方法 | Acc in DRE | Acc in Full |
|---|---|---|
| 无 RS | 53.58% | 54.77% |
| + RS (Sonnet-3.7+gt) | **68.85% (+15.27)** | **66.73% (+11.96)** |
| + RS (Critic-4B) | 63.24% (+9.66) | 61.66% (+6.89) |

**Qwen3-8B 在 TableBench 上**：

| 方法 | Acc in DRE | Acc in Full |
|---|---|---|
| 无 RS | 63.54% | 77.48% |
| + RS (Sonnet-3.7+gt) | 69.09% (+5.55) | 79.31% (+1.83) |
| + RS (Critic-4B) | 66.85% (+3.31) | 78.50% (+1.02) |

规律很明显：

1. **越弱的模型，RS 收益越大**。Distill-Qwen-7B 涨 11.96%，Qwen3-8B 涨 1.83%。强模型本身 DRE 少，RS 没多少发挥空间。
2. **Acc in DRE 比 Acc in Full 涨得多**。这跟直觉一致——RS 直接针对 DRE 子集，能纠错的也就是这部分。
3. **Critic-4B 涨的没 Sonnet-3.7+gt 多，但绝对值仍可观**。考虑到 Sonnet-3.7+gt 是商业大模型 + 知道 ground truth，而 Critic-4B 是 4B 参数量 + 不知道 gt，**这个 trade-off 在很多成本敏感场景下完全划算**。

---

## 我的判断

这篇论文属于"问题定义型"工作——它没有提出什么惊天新方法，核心贡献其实是三件事：

1. **第一次系统量化 DRE**。1.7B 到 20B、九个模型、五类任务，给出可复现的 DRE 评测协议。这件事本身就有工具价值——以后做表格 LLM 的评测，可以直接用 DRE Rate 作为辅助指标。

2. **确认现有手段都不work**。Self-reflection、prompt 注入、专门 RL 训练（Table-R1-Zero-7B）都不能显著降低 DRE。这一点是反共识的——很多人以为"推理模型 + 长 CoT = 不会抄错"，数据告诉我们不是。

3. **给出 4B 小裁判这个工程上很甜点的解**。一个 4B 模型 + 拒绝采样，弱模型涨 12 个点强模型涨 1-2 个点。部署成本极低，收益曲线非常漂亮。

**几个我读完后想吐槽的点**：

- **gpt-oss-20b 的 DRE Rate 只有 5.71%，这件事没展开**。如果 OpenAI 在新模型里专门针对"抄对"做了优化，这是个非常重要的工程信号。论文作者跟 OpenAI 没关系，没法深挖，但读者应该会好奇。

- **Table-R1-Zero-7B 的 DRE Rate 比 Qwen2.5-7B-Instruct 还高**。一个"专门为表格训练"的模型表现比通用模型还差？说明 RL 训练在奖励设计上有盲区——你只奖励最终答案对，没奖励"中间过程对"，模型就会走捷径。这其实跟 RLHF 的 reward hacking 是同构的。

- **作者没做 attention 可视化或 attention steering 实验**。他们自己在局限性里也承认了"初步观察到增加对整张表的注意力有助于减少后续错误，但没做大模实验"。这其实是 DRE 成因分析最关键的一块，留给后续工作了。

- **Sonnet-3.7+gt 作为 judge 的人手准确率 92.67%**——意味着 Table 1 的所有 DRE Rate 数字里大概有 7% 是 judge 本身误判。这个噪音没在表里展示置信区间，统计意义上能不能这么直接对比不同模型，是个开放问题。但 7% 的 judge error 在所有模型上一视同仁，相对排序应该还是稳的。

- **判定 DRE 的"段"切分方式**（在 reflection token 如 "Wait" 处切）挺巧妙但有点 ad hoc。Qwen3 系列长 CoT 里有 "Wait" 切分自然，但 Llama4-Scout 的反思 token 不一定长这样。这块在论文里写得有点简略。

**整体定位**：这是一篇工具型论文，不会上 SOTA 榜单刷出新数字。但它把一个长期被忽视的错误类型拎出来量化、给出评测协议、给出低成本缓解方案——对做表格 LLM、做 RAG 系统、做数据分析师 Agent 的人来说，这都是一个值得纳入评测流程的维度。

**工程启发**（这部分给我自己做项目时用）：

- 如果你的表格问答产品对**最终答案正确率**敏感，但**过程可解释性**也敏感（比如金融、医疗、合规），加一个 4B 的 DRE critic 做后置过滤，成本可控，收益曲线漂亮。
- **别用 CSV 格式喂模型**。JSON 或 Markdown 更稳，列分隔符越少越好。
- **表格专用 RL 不等于不抄错**。Reward 设计要把"中间引用一致性"加进去，单看最终答案会绕开 DRE。
- **"推理模型 + 长 CoT" 不是 DRE 的银弹**。Qwen3-8B 已经是推理模型，DRE 率仍有 14%。如果你的场景对中间过程敏感，仍然需要 critic。

---

## 几个未解的开放问题

最后留几个我自己读完想追问但论文没解决的：

1. **DRE 是不是表格任务的特例？** 数学推理里的"把 0.714 抄成 0.704"也是 DRE。论文局限里也提了，但他们只做表格。这个错误模式在 code、math、retrieval 里是不是也普遍？值得跟一篇后续工作看看。

2. **gpt-oss-20b 为什么这么好？** 是训练数据、reward 设计、还是某种 internal CoT 格式？这一点完全黑盒。

3. **Critic-4B 的 F1 在 OOD 上还有提升空间**。从 Figure 2 看，FinQA + Llama4-Scout 的 pair 上 Critic-4B 也只到 78% 左右，跟 in-domain 接近。但这是**最难**的 pair（训练数据是 Qwen3-8B + WTQ，跨模型跨数据集），能做到这个水平已经不错。如果做大规模跨模型蒸馏的 critic，能不能到 85%+？

4. **RLVR 阶段 20 epoch 训 4B 模型**——这训练成本其实不算低。论文没给具体 GPU 数和时长，但 verl + 4B + 5712 样本 × 20 epoch × 8 rollouts，估算在 8×H100 上一两天内能跑完。这对想做复现的团队来说是个合理的预算。

---

如果你也在做表格问答、表格 Agent、或者 RAG 这类"模型要照着数据说话"的事情，DRE 这个维度真的值得加到你的评测里——它揭示的是模型**引用一致性**的能力，跟"答得对不对"是两个独立维度。一个 4B 小模型 + 一点 SFT/RL 成本，就能把这条线拉起来，性价比相当不错。

> 论文：https://arxiv.org/abs/2606.32029
> 标题：When LLMs Read Tables Carelessly: Measuring and Reducing Data Referencing Errors
> 会议：ACL 2026 Oral

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。*
