---
title: "FFN幅度门控：在不动模型权重的条件下，工具调用涨了11个点"
date: 2026-07-20
arxiv: 2607.11183
authors:
  - Sheng Xu
  - Junhua Wang
  - Boyuan Huang
  - Ke Jia
  - Jiadun Zhu
  - Zhen Chen
institution: Alibaba Cloud
---

# FFN幅度门控：在不动模型权重的条件下，工具调用涨了11个点

你有没有碰到过这种情况——模型明明"懂"你想调哪个工具，最后一步给出的函数名却多带一个后缀、参数列表少一个右括号、或者 JSON 在 prompt 边界处突然多出一截 "user" 字符串，然后整条调用被判为失败？

在普通的开放问答里，这种"差一点就对"几乎无伤大雅。但在工具调用、Agent、JSON 模式这些场景下，输出是要被 parser、validator、API 消费的——一个错的边界字符就会让一条原本合理的 action 报废。这是 LLM 当前最尴尬的一类错误：模型语义上靠近答案，但 FFN 在最后一公里走偏了。

阿里云的一篇新论文 [arXiv:2607.11183](https://arxiv.org/abs/2607.11183)（Xu et al., 2026-07-13）就盯上了这个"最后一公里"。他们提出 **Amplitude Gating (AG)**，一种**不改变预训练 FFN 权重、只在推理时调制激活幅度**的干预方法。在 Qwen3.5-9B 的 Hermes function-call 上，**learned gate 提升 7.64 个百分点**（29.02% → 36.66%）；在 Qwen3-8B 的 Hermes JSON mode 上，**提升 11.36 个百分点**（41.83% → 53.19%）。而且整个方法不需要重训任何权重。

## 核心摘要

这篇论文的真正价值不在于"又一个推理时干预方法"，而在于它把**一次失败**打磨成了一套**保守、可部署、有严格评估边界**的方案。

- **痛点**：tool-structured 失败（错函数名、错 JSON 边界、错 schema 续写）大量发生在 FFN 内部——模型"接近"但"走偏"。
- **方案起点**：作者原本尝试了 **Orthogonal Residual Projection (ORP)**，用 SVD 推导正交投影来"修"FFN 方向。结果在 1,032 题的 GSM8K 矩阵审计里，34 题修对、56 题修坏，净 -22。ORP 把"FFN 哪里敏感"摸清楚了，但也证明了"改方向"是危险动作。
- **核心方案**：**Amplitude Gating (AG)**——保留预训练 FFN 权重方向不变，只对激活的幅度做掩码/缩放，并配上一个**在线可观测特征的 gate**来选择"接受 AG 还是 fallback 到 baseline"。
- **关键效果**：
  - Qwen3.5-9B tool/structured/agentic 类别：38.66% → 42.92%（**4.27 pp**，mixed gate）
  - Qwen3.5-9B Hermes function-call：29.02% → 36.66%（**7.64 pp**）
  - Qwen3-8B Hermes JSON mode：41.83% → 53.19%（**11.36 pp**）
  - Qwen2.5-7B：候选空间有 oracle headroom，但当前 learned gate 抓不住——**重要的负面结果**
- **一针见血的判断**：这不是"FFN 干预的万能开关"，而是一份**带可证伪边界的工程实证**。它把"非破坏性 FFN 干预"做成了 tool-structured 推理优化的可信第一站，但跨模型、跨任务、跨 attention 架构的通用性仍待验证。

---

## 论文信息

| 项目 | 内容 |
|------|------|
| 标题 | Amplitude-Only FFN Intervention for Tool-Structured LLM Inference: Method, Gated Evaluation Protocol, and Cross-Model Empirical Results |
| 作者 | Sheng Xu, Junhua Wang, Boyuan Huang, Ke Jia, Jiadun Zhu, Zhen Chen |
| 机构 | Alibaba Cloud（邮箱域名：alibaba-inc.com / taobao.com）|
| arXiv | [2607.11183](https://arxiv.org/abs/2607.11183) |
| 提交日期 | 2026-07-13（v1）|
| 类别 | cs.CL |

---

## 1. 问题动机：为什么要在 FFN 层做干预？

普通的"加速"或者"约束解码"都不直接解决这个问题：

- **Constrained decoding / JSON-schema 引导**：在 decoder 层强制语法，但模型内部已经算好下一 token 分布了，它能强制你输出合法 JSON，但**管不了模型选哪个函数名、哪个 schema 续写**。
- **Representation engineering / 激活 steering**：注入任务特定的语义方向（比如诚实、情感）。AG 不做这事，它**不注入任何语义向量**，只调"幅度"。
- **FFN 加速 / sparsification**：Fast Feedforward 这类方法用更少的 FFN 神经元降推理成本。AG 的目标不同——**AG 不是为了快，是为了让 FFN 内部的 transform 改对**。
- **Tool training / 数据构造**：ToolLLM、ToolBench 这类工作解决的是训练侧的数据、检索、prompt。AG 是**纯推理时**的方法。

那么 AG 想干嘛？作者在 Section 3.1 给出了一个非常清晰的直觉：

> Attention 把信息带进上下文，FFN 把这些信息转成"下一个 token 偏好的特征模式"。tool-structured 失败很多是 last-mile 转换错误：模型在对的语义邻居里，但输出了错的边界、错的参数、错的 schema 续写。

FFN 之所以有吸引力，是因为它是跨 attention 架构都存在的"非线性 channel-mixing 模块"。一个对 FFN 友好的干预，在 dense full-attention、hybrid linear/full-attention 之间都能 map。但**任何对内部表征的改动都有"修一条、毁十条"的风险**。所以 AG 必须同时回答两个问题：在 FFN 的哪一点介入？以及用什么形式的介入**不破坏预训练模型的语义结构**？

第一个问题的答案来自 ORP 的诊断；第二个问题的答案来自 ORP 的失败。

---

## 2. 方法核心：从 ORP 失败到 AG 保守

### 2.1 ORP 的"科学价值"与"部署失败"

ORP（Orthogonal Residual Projection）是作者团队内部早期的一个推理时修复尝试。它的核心思路是：FFN 内部表征如果漂向"坏的子空间"，就用 SVD 推导出的正交矩阵把方向"扳回去"。

诊断结果是混合的：

- **有用**：ORP 跑通 P1（FFN 输入）/P2（FFN 内部 fusion）/P3（FFN 输出）三个干预位置，发现 e-值（保留能量比例）对 fix/harm 的影响是**非单调的**。比如在 1,319 题的 P1 扫参里，e=0.93 净 +9（21 benefit / 12 harm），e=0.97 净 -5，e=0.80 净 -2。
- **有毁**：在完整 GSM8K 矩阵审计里，加载的 `qwen35|svd_orp|p1|e0.95|w3` 跑 1,032 题，**修了 34 题、毁了 56 题，净 -22**。1,012/1,032 答案在 token 级就跟 baseline 分叉了。

这个 -22 才是论文真正的概念支点——它告诉我们：

> Pretrained FFN 的方向是**语义结构**，不是可调噪声。你想"扳"它，它就崩给你看。

![ORP 失败诊断：P1 能量扫参非单调，34 fixes vs 56 regressions 净 -22](https://www.mulanai.com/fs/files/0720_ba7ff179_page_5-0.jpg)

*图 1：ORP 失败诊断。左侧 P1 energy sweep 在 1,319 GSM8K 例子上显示 fix/harm 净效果随 e 非单调（0.97、0.80 都是净负），右侧加载的 SVD-ORP checkpoint 给出 34 fixes vs 56 regressions 的 -22 净亏损。来源：论文 Figure 1。*

### 2.2 AG 的设计原则

既然"改方向"是危险动作，那就只剩下一条路：**改幅度**。AG 的核心主张极其保守：

1. **保留所有预训练 FFN 权重不变**；
2. 在选定的 FFN 点（位置 P），对激活做**掩码或缩放**，不构造新的投影基；
3. 在推理时给每条样本跑 baseline + AG 候选，**用 gate 决定接受 AG 还是 fallback 到 baseline**。

这个 gate 是关键——它把"AG 候选空间"和"实际部署动作"分开。AG 提供候选，gate 做选择。

![AG vs ORP 设计支点：ORP 改 SVD 投影方向，AG 保持预训练 FFN 权重不动](https://www.mulanai.com/fs/files/0720_ba7ff179_page_5-0.jpg)

*图 2：设计支点对比。左侧 ORP 把 FFN 表征投到 SVD 估计的"修复子空间"，但 34 fixes / 56 regressions 的诊断结果说明"扳方向"不可控。右侧 AG 保留预训练权重，只对激活幅度做门控，gate 在线决定接受还是 fallback。来源：论文 Figure 2。*

### 2.3 SwiGLU FFN 的六点干预系统

AG 的具体干预位置不是拍脑袋定的，而是从 SwiGLU 的结构推出来的。Qwen 家族用的是 SwiGLU-style gated FFN（gate / value 分支 + elementwise fusion + down projection）。AG 在这个块里定义了**六个干预点**：

| 点 | 位置 | 维度 | 角色 |
|------|------|------|------|
| **P1** | 归一化后、up/gate projection 之前 | hidden | 继承自 ORP。FFN 输入标准点，但混合了 gate/value 两路 |
| **P1s** | 仅 gate 分支的输入变体 | hidden | **新点**。SwiGLU 特有，gate-branch 选择。隐藏维度成本，研究与部署优先级都高 |
| **P2a** | gate projection + SiLU 之后，融合前 | 4×hidden | **新点**。直通道切换信号，扩张维度成本高、噪声大，研究优先级高、部署优先级低 |
| **P2b** | up projection 之后，融合前 | 4×hidden | **新点**。value-branch 信号，扩张维度，某些 tool 任务上效果不错 |
| **P2** | gate/value 融合后，down projection 之前 | 4×hidden | 继承自 ORP。最信息丰富的融合表征，但成本和噪声都高，主要是 research/oracle 用途 |
| **P3** | down projection 之后、residual 之前 | hidden | 继承自 ORP。输出端 hidden 维度信号，**最保守的部署首选** |

这张表是论文"动 FFN 该动哪"的最直接答案。P1/P2/P3 是"在 ORP 阶段就验证过的点"，P1s/P2a/P2b 是"转入 AG 之后、利用 SwiGLU 分支结构新加的细粒度点"。论文明确说：**P1s/P2a/P2b 的"机制正交性"还只是假设，不是已被证明的机制拆分**。

![AG 干预点在 Transformer FFN 块内的位置](https://www.mulanai.com/fs/files/0720_fb5e991e_page_6-0.jpg)

*图 3：AG 干预点架构图。在 SwiGLU FFN 块内，从 FFN 输入（P1 / P1s）→ 内部扩张（P2a / P2b）→ 融合（P2）→ 输出（P3）共 6 个可干预点。prefill 阶段不动，AG/gate 决策从第一个 decode token 开始。来源：论文 Figure 3。*

### 2.4 AG 的六种变体

按"用什么信号排序维度"和"怎么改幅度"两个轴，AG 给出六种变体。这里挑三种最有代表性的：

- **Hard AG**：按 $x_i^2$ 排序，取累积能量达到 $eE$ 的最短前缀做硬掩码。最简单，"能量保留"语义清晰。
- **Soft AG**：用同样的 L2 排序选阈值 $\tau = |x_{t,\pi_{last}}|$，但**软缩**而不是硬切。$\tau$ 是 $K_{hard}$ 最后一个入选维度的值。幅度更平滑但更"破坏性"。
- **NS AG (Newton-Schulz windowed)**：这是论文重点比较的"另一支"。用最近 4 个 decode token 的激活窗口 $W_t \in \mathbb{R}^{w \times d}$ 跑 Newton-Schulz 迭代，得到一个**窗口级分数** $s_t^{NS}$，再用这个分数排序维度，套同样的 entropy-mass 掩码。**窗口是关键**：单个向量上跑 NS 几乎退化成 magnitude rank；只有 4-token 窗口才让 NS 跟 magnitude 区分开来。

NS 这个名字容易让人误会——它**不是 SVD 投影**，**不在 AG 里构造任何投影基**。它只是改了"用什么分数给维度排序"。

### 2.5 Gate：让"候选空间"和"在线动作"分开

AG 是个候选生成器，gate 才是部署决策。论文的 gate 协议有几个关键设计：

- **只用在线可观测特征**：prefix/format（JSON 节点数、最大深度、参数名计数）、early logprob trajectory（前几个 token 的 slope、trend）、baseline uncertainty（top-k 概率 gap、acceptance 统计）、divergence/energy（首分叉点、本地分叉比、保留能量）。
- **禁用诊断用特征**：cross-PPL、post-5 overlap、label-derived 字段。**只用这些特征做的 gate 才能声称"可部署"**。
- **四级证据分级**：
  - **Combination oracle**：看分数选最优候选（**仅上界，不可部署**）
  - **Best fixed single configuration**：在数据集/类别上挑一个固定 AG 跑全局（弱证据）
  - **Learned gate**：用在线可观测特征学的策略（**主要 deployable 证据**）
  - **Diagnostic-only**：用 post-hoc 标签相关特征（仅解释用）

这四级的区分是论文"防过度宣称"的最重要护栏。

---

## 3. 实验设计：覆盖而不越界

论文的实验设计有意识地"做宽"和"做严"同时拉满：

### 3.1 三个模型，覆盖三种 attention 结构

| 实验键 | Checkpoint | Attention | FFN 形状 | 角色 |
|--------|-----------|-----------|---------|------|
| `qwen35` | Qwen3.5-9B | 32 层 hybrid（24 层 linear + 8 层 full attention，3:1 周期）| SwiGLU, hidden 4096 → 12288 | 最强 function-call / tool-use 证据 |
| `qwen3` | Qwen3-8B | 36 层 dense full-attention | SwiGLU, hidden 4096 → 12288 | 最强 JSON / prefix-format 证据 |
| `qwen25_7b` | Qwen2.5-7B | 28 层 dense full-attention | SwiGLU, hidden 3584 → 18944 | 边界情况：oracle headroom 无可部署 gate |

Qwen3.5-9B 测试 hybrid 架构，Qwen3-8B 和 Qwen2.5-7B 测试 dense 架构但深度和 FFN 扩张维度不同。所有三个都用 SwiGLU，**因此六点干预系统可以在三模型上直接对比**。

### 3.2 八个数据集，三条评估路线

- **tool_structured_agentic**：ToolBench in/out-domain、Hermes function call（双轮 + 单轮）、Hermes JSON mode 单轮
- **text_long_qa**：ConvFinQA dev、Leval multidoc QA
- **text_hallucination_factuality**：HEB RAG hallucination QA
- **MC diagnostics**（仅在 MMLU-Pro、C-Eval 上做边界测试）

### 3.3 Matched candidate matrix

每个样本在每个模型/数据集上，**baseline + 24 个 AG 候选**（12 old-P + 12 new-P）一并跑。old-P 是 P1/P2/P3 × 2 个 e × 2 个 AG 家族（pure_entropy_ag / ns_entropy_ag w4）；new-P 是 P1s/P2a/P2b 同样的设计。

这里有个常被忽略的细节：e 的两个值（0.93、0.95）**不是从测试标签上学到的**，而是 ORP 早期诊断里反复出现"敏感区"的两个 paired probe。这样设计是为了**测位置 × e 的交互**，而不是做"哪个 e 最好"的超参搜索——后者容易陷入 best-cell overinterpretation。

![e0.93 / e0.95 是 paired probe，不是 dense sweep](https://www.mulanai.com/fs/files/0720_02d0c0d8_page_11-.jpg)

*图 4：matched e-grid 设计。0.93 和 0.95 处在 ORP/early-AG sweep 反复出现的敏感区内，作为"足够近但又能区分"的两个 paired probe，用来检验 P × e 的交互是否稳定。来源：论文 Figure 5。*

### 3.4 严格防泄漏

四个泄漏控制：

1. 每个 run 都跟**同一模型、同一 dataset、同一 decoder 配置**的 baseline 比；
2. 候选生成和 gate 评估**分开**；
3. learned gate 只用 gate-policy 特征；
4. 任务类别和跨模型的压力测试（不靠"把所有 dataset 池起来"涨分）。

---

## 4. 实验结果：三个模型，三种"故事"

### 4.1 Qwen3.5-9B：tool/function-call 才是 AG 的真正主场

在 Qwen3.5-9B 上，AG 最强的证据集中在 tool-structured 任务上：

| 类别 / 方法 | Baseline | Learned / current | Lift | Oracle delta | Oracle capture |
|-------------|----------|-------------------|------|--------------|----------------|
| Tool/structured, mixed gate | 38.66% | 42.92% | **4.27 pp** | +11.99 pp | 35.6% |
| Tool/structured, ridge gate | 38.66% | 42.87% | +4.21 pp | +11.99 pp | 35.1% |
| Tool/structured, best fixed | 38.66% | 41.12% | +2.46 pp | +11.99 pp | 20.5% |

mixed gate 比 best fixed 涨得更多（+4.27 vs +2.46），说明**样本级特征门控有效**——不是"一个 AG 配就能赢"，而是"每个样本的 routing 才赢"。

数据集级最强的是 **Hermes function call**：

| 数据集 | Baseline | Combination oracle | Best learned | Learned lift | Oracle capture |
|--------|----------|--------------------|--------------|--------------|----------------|
| Hermes function call | 29.02% | 40.03% | 36.66% | **7.64 pp** | 69.4% |
| Hermes function call single-turn | 28.76% | 39.38% | 36.38% | +7.62 pp | 71.7% |
| ToolBench out-domain | 40.93% | 61.00% | 47.49% | +6.56 pp | 32.7% |
| ToolBench in-domain | 33.99% | 53.42% | 37.95% | +3.96 pp | 20.4% |

Fix/harm accounting：mixed gate 在 strict 下 388 fixes / 317 harms（净正），partial10 下 2,425 / 1,401（强正）。Oracle 更大但不可部署（838 / 349 strict、4,125 / 1,251 partial10）。**这个对照——learned gate 正、oracle 更大但包含不可部署的策略——是论文最诚实的呈现**。

在非 tool 任务上，Qwen3.5-9B 也正但弱：HEB RAG 21.64% → 25.10%（+3.47 pp，类别只有一个 dataset，不该广推），Long QA 3.71% → 4.42%（+0.71 pp，弱正）。

![Qwen3.5-9B tool-structured 结果：baseline / learned / oracle 三色对比](https://www.mulanai.com/fs/files/0720_053de78a_page_15-.jpg)

*图 5：Qwen3.5-9B 在 tool/structured 路线上 baseline（灰）、learned gate（蓝）、combination oracle（橙）三条柱。Tool 类别 learned gate 从 38.66% 升到 42.92%，oracle 上界 50.65%。Hermes function-call 涨 7.64 pp，是最显著的样本级证据。来源：论文 Figure 6。*

### 4.2 Qwen3-8B：JSON / prefix-format 才是 AG 的最强信号

Qwen3-8B 也呈正，但最强的子任务换了：

| 类别 / 方法 | Baseline | Learned / current | Lift | Oracle delta | Oracle capture |
|-------------|----------|-------------------|------|--------------|----------------|
| Tool/structured, mixed gate | 42.78% | 45.42% | +2.64 pp | +7.77 pp | 34.0% |
| Tool/structured, ridge gate | 42.74% | 45.23% | +2.49 pp | +7.77 pp | 32.1% |
| Tool/structured, single indicator | 42.75% | 45.34% | +2.59 pp | +7.77 pp | 33.3% |

Qwen3-8B 的最强信号在 **Hermes JSON mode single-turn**：

| 数据集 | Baseline | Combination oracle | Best learned | Learned lift | Oracle capture |
|--------|----------|--------------------|--------------|--------------|----------------|
| **Hermes JSON mode single-turn** | 41.83% | 56.49% | 53.19% | **11.36 pp** | 77.6% |
| ToolBench out-domain | 47.11% | 57.14% | 50.58% | +3.47 pp | 36.0% |
| ToolBench in-domain | 45.32% | 53.60% | 47.84% | +2.52 pp | 29.2% |
| Hermes function call single-turn | 40.89% | 45.89% | 42.94% | +2.05 pp | 41.0% |
| Hermes function call | 41.46% | 46.03% | 43.38% | +1.92 pp | 42.1% |

但风险画像不同：Qwen3-8B mixed gate 在 strict 下 22 fixes / 27 harms（**净负**），partial10 下 218 / 40（强正）。所以 Qwen3-8B 的成就是**靠 partial-credit 涨的**，不是"零风险 strict 涨"。论文也明说它的 framing 是"structured-output score improvement"而不是"zero-harm binary deployment"。

跨模型对比的一个关键观察：**Qwen3.5-9B 最强是 function-call，Qwen3-8B 最强是 JSON mode**——同一个 Qwen 家族，最佳 gate 都不一样。这是后面"必须 routing"结论的根。

![Qwen3-8B tool-structured 结果](https://www.mulanai.com/fs/files/0720_b9e2bfd1_page_16-.jpg)

*图 6：Qwen3-8B 在 tool/structured 路线上的 baseline / learned / oracle 三色对比。Hermes JSON mode single-turn 是最大赢家（41.83% → 53.19%，+11.36 pp），跟 Qwen3.5-9B 的 function-call 最强形成对比。来源：论文 Figure 7。*

### 4.3 Qwen2.5-7B：oracle headroom 在，但 gate 抓不住

这是论文最重要的"负面证据"：

| 类别 | Best learned method | Baseline | Learned / current | Learned lift | Oracle delta | Oracle capture |
|------|--------------------|---------:|------------------:|-------------:|-------------:|---------------:|
| Tool/structured | mixed gate | 44.17% | 44.13% | -0.04 pp | +4.03 pp | -0.94% |
| Long QA | ridge gate | 4.64% | 4.45% | -0.19 pp | +2.98 pp | -6.30% |
| Factuality | mixed gate | 32.10% | 32.86% | +0.76 pp | +4.78 pp | 15.87% |

Qwen2.5-7B 的 tool/structured 类别有 +4.03 pp 的 oracle headroom，**但 learned gate 净负 0.04 pp**。这说明："候选空间里有好东西" ≠ "当前 gate 能挑出来"。模型能用 ≠ 模型能稳选。

论文在 fixed configuration 上做了细致的诊断：Qwen2.5-7B 在 Hermes function call 上，**所有展示的 new-P 单元都是负的**；ToolBench out-domain 只有一个小的 P1s 路线正，幅度也很小。

这一个负面 case 才是论文防止过度宣称的最重证据——**有 headroom 不等于能上**。Qwen2.5-7B 落在"边界情况"区间。

### 4.4 NS AG vs non-NS AG：没有统一赢家

很多读者会期待"NS 比 non-NS 更强"——因为 NS 用了窗口级信息，听起来更"高级"。但 paper 显式反对这个假设。

| Model | Category | Baseline | Score non-NS | Score NS | Winner | Winner lift |
|-------|----------|---------:|-------------:|---------:|--------|------------:|
| Qwen3.5-9B | Tool/structured | 38.92% | 40.76% | 40.11% | non-NS | +1.83 pp |
| Qwen3.5-9B | Long QA | 4.22% | 4.64% | 5.00% | NS | +0.78 pp |
| Qwen3.5-9B | Factuality | 22.24% | 22.15% | 23.75% | NS | +1.51 pp |
| Qwen3-8B | Tool/structured | 49.76% | 49.40% | 50.20% | NS | +0.44 pp |
| Qwen3-8B | Long QA | 10.89% | 10.41% | 10.57% | NS | -0.32 pp |
| Qwen3-8B | Factuality | 32.31% | 32.55% | 32.55% | tie | +0.24 pp |
| Qwen2.5-7B | Tool/structured | 43.95% | 43.90% | 43.90% | near tie | -0.04 pp |
| Qwen2.5-7B | Long QA | 5.38% | 5.76% | 6.06% | NS | +0.68 pp |
| Qwen2.5-7B | Factuality | 31.91% | 32.55% | 32.42% | non-NS | +0.64 pp |

规律是清楚的：

- **Tool/structured 任务、current-token-specific 决策**（开闭 JSON 括号、选函数名）—— non-NS 在 Qwen3.5-9B 上赢，因为它更"尖"，不模糊局部峰值。
- **Long QA、factuality 这种"局部稳定性更重要"的任务**—— NS 在大多数 case 上赢，因为它用 4-token 窗口拿一个稳定的 ranker。

但"赢"不是"可部署"——Qwen3-8B Tool/structured 的 NS 赢 +0.44 pp 看着小，但**学出来的最强 gate 来自 prefix/format 特征**，不是 NS 规则本身。

论文对 NS 的最终结论是 **"two-family candidate system"**：

- **non-NS**：sharp current-token amplitude gate，task 形态是"快速、清晰的本地决策"。
- **NS**：windowed local-trajectory gate，task 形态是"短窗口稳定性"。

两者**互补**，**不是上下位**。可部署的 AG 系统应该**同时暴露两类，让 gate 决定用哪个**。

---

## 5. 为什么 tool-structured 任务最受益？

论文给了一个我非常认同的机制解释（Section 6.4）：

1. **输出空间是结构化的、约束的**。函数名、JSON 节点、参数列表——prefix/format 特征**能在线观察到**模型是否进入了正确的结构态。这让 gate 决策比开放式 QA 容易得多。
2. **小的内部 shift 能在外部产生离散改善**。换一组 FFN 激活就可能保住函数名、参数顺序、括号结构，得到一个 evaluator 认可的 action。开放式 QA 里语义正确性是弥散的，依赖全局 reasoning 和 retrieval grounding，AG 即使 oracle 有 headroom，gate 也抓不住。

Qwen3.5-9B 的一个 ToolBench out-domain 样本能直观说明——

| Case | 任务线索 | Gold first action | Baseline first action | AG first action | 解读 |
|------|----------|-------------------|----------------------|-----------------|------|
| Fix | quote categories | list_categories_for_famous_quotes | list_categories_for_famous_quotes**user** | list_categories_for_famous_quotes | AG 删掉 prompt 边界后缀，action 合法 |
| Fix | streaming availability | search_basic_free_for_streaming_availability | search_pro_for_streaming_availability | search_basic_free_for_streaming_availability | AG 把"pro" endpoint 换成"basic/free"，对齐 state 要求 |
| Harm | streaming availability | search_pro_for_streaming_availability | search_pro_for_streaming_availability | search_pro_for_streaming_availability**user** | AG 反而在正确 action 后加了边界后缀，弄脏了一条原本对的 |

这些 case 印证了作者的判断：**AG 修的不是"语义"，是"action 边界的 last-mile"**。这也是为什么他们强调 first-action exact match 比 loose string-contains 更可信。

---

## 6. 关键判断：哪些地方要打问号？

### 6.1 P × e 的交互不能"一招通用"

论文 Section 6.1 的 matched P-by-e 表特别值得看：

**Hermes function call (Qwen3.5-9B non-NS)**：P1s 强（+2.43 pp at e=0.93，+1.42 pp at e=0.95），P2a 和 P2b 在两个 e 值下都是负。

**ToolBench out-domain (Qwen3.5-9B non-NS)**：P2b/e0.93 是最强 fixed（+3.07 pp），P1s 仍正但弱，**P2a 跨 e 改符号**（e=0.93 负、e=0.95 正）。

跨模型看：Qwen3-8B 在 Hermes function call 上**最优 P 换成 P2a**，Qwen2.5-7B 在这个数据集上 new-P 单元全是负。ToolBench out-domain 在 Qwen3.5-9B 和 Qwen3-8B 上都是 P2b/e0.93 最强，但 Qwen2.5-7B 只有 P1s/e0.95 这个小的正 cell。

结论：**没有一个 P 永远赢，没有一个 e 永远赢**。位置、e、模型、任务类别**联合决定 AG 是否有效**。

### 6.2 候选空间 headroom ≠ 可部署 gain

Qwen3.5-9B ToolBench out-domain：oracle +20.08 pp（在 mining 池里），learned gate 只捕获 +6.56 pp。
Qwen2.5-7B tool 类别：oracle +4.03 pp，learned gate **净 -0.04 pp**。

论文把这个区分讲得很清楚——**combination oracle 看的是"候选空间有没有有用的替代"，不是"能不能在线挑出来"**。大 oracle gap + 弱 learned gate 说明研究问题没解决，不是说方法 ready。

### 6.3 跨模型不能套同一个 gate

Qwen3.5-9B 最强是 function-call（Hermes function call，prefix/format 特征 + 早 token trajectory）。
Qwen3-8B 最强是 JSON mode（Hermes JSON mode，prefix/format 特征占主导）。
Qwen2.5-7B 没有可部署的 category gate。

论文因此提出 "model- and category-specific routing" 的部署建议：

1. 先选 model family + task category；
2. 用该路线专门 train + validate 的 gate；
3. 上线前要求 held-out 验证；
4. gate 不置信就 fallback 到 baseline。

这是论文最克制也最值得尊重的一句话——它没有说"AG 让所有 tool 调用涨 5 个点"，它说"按 routing 部署，AG 在 tool-structured 上是安全 first target"。

### 6.4 Choice 任务上的边界证据

为了不让你被 +11.36 pp 冲昏头，论文 Section 6.3 给了 MC/choice 的负面诊断：

| Scope | Baseline | P/e/config indicator | Choice/trace feature gate | Combination oracle | 解读 |
|-------|---------:|----------------------:|--------------------------:|-------------------:|------|
| MC choice all | 54.27% | +0.410 pp, CI [0.160, 0.661] | -0.080 pp, CI [-0.228, 0.057] | +0.980 pp, CI [0.741, 1.242] | 小静态信号；feature gate 未建立 |
| MMLU-Pro | 52.85% | +0.399 pp, CI [0.133, 0.664] | -0.085 pp, CI [-0.229, 0.060] | +0.966 pp, CI [0.736, 1.232] | oracle headroom 仍小 |
| C-Eval | 77.98% | +0.000 pp, CI [-0.808, 0.808] | +0.000 pp, CI [0.000, 0.000] | +1.212 pp, CI [0.404, 2.222] | baseline 已高；无 learned 痕迹 |

AG 在 choice 上**几乎没 oracle headroom**。原因符合直觉：choice 是"一个 token 决定"，AG 是 decode-only，**几乎没 post-prefill trajectory 让 AG 改**。

这是 AG "claim boundary" 最关键的一块。论文不靠 choice 涨分来撑"AG 万能"的说法。

---

## 7. 部署建议：safety-first / balanced / performance-first

Section 7 给出了三档部署 profile：

| Profile | 目的 | Gate 风格 | 适用场景 |
|---------|------|----------|---------|
| Safety-first | 最小化伤害 | 低覆盖保守 gate + baseline fallback | 高风险域 |
| Balanced | 稳定捕获 tool gain | category/model 特定的 mixed 或 ridge gate | 一般 tool calling |
| Performance-first | 抓更多 oracle space | 高覆盖 gate + 监控 | 内部或批处理工作流 |

三个具体工程建议（值得划重点）：

- **优先 P1s / P3 这类 hidden 维度的点**，不要轻易上 P2/P2a/P2b 这些 4×hidden 的扩张点。理由是 activation 存储和噪声。
- **不要一上来就做 multi-flow multi-candidate voting**。离线 mining 用它 OK，但在线成本随候选数线性涨。"single-flow 或 light-augmented baseline + candidate" 更现实。
- **cross-PPL 之类的 post-fork 诊断特征在当前实现是 diagnostic-only**，因为它依赖生成后的 AG tokens。生产系统要么用在线近似，要么真把 cost 测出来。

---

## 8. 我的判断：这是工程整合还是底层突破？

说几个明确的判断：

**1. 这不是底层突破，是一次极其严谨的"工程整合 + 失败转设计原则"**。核心 idea——**对 FFN 激活做幅度门控而非方向修改**——本身并不新（activation steering、FFN sparsification、representation editing 都是这条线上的邻居）。论文真正的贡献在于：
- 把 ORP 的失败转成 AG 的设计原则（"方向不要动"），这在论文写作上是非常漂亮的 negative-as-design-principle；
- 定义了**四级证据分级 + 严格防泄漏协议**——这套评估协议比方法本身对社区的价值还大；
- 给出了"tool-structured 是 AG 的可信 first target"的可证伪边界。

**2. +11.36 pp / +7.64 pp 是真的，但要在 routing 框架下读**。如果有人跟你吹"AG 让 Qwen3-8B 在 Hermes JSON 上涨 11.36 pp"，你可以反问："那在 MMLU-Pro 上呢？C-Eval 呢？另外两个模型上呢？" 答案就是：在 choice 任务上几乎没 headroom；Qwen2.5-7B tool 类别 learned gate 还是负的。**AG 不是 universal switch，是 routed intervention**。

**3. NS vs non-NS 的"two-family"结论是真有价值的**。很多人（也包括我）会想"NS 用了 Newton-Schulz 听起来更高级，肯定更强"。论文用三模型 × 三类别的实证打掉了这个假设，**说明 routing 该在 family 级别做，不在 family 内部做**。

**4. 跨模型 ≠ 跨架构**。论文自己说这是"cross-checkpoint existence within one model family"。Qwen3.5-9B（hybrid）、Qwen3-8B 和 Qwen2.5-7B（dense full attention）覆盖了 attention 结构变化，但**都是 Qwen 家族**。要把它推到 Llama、Mistral、DeepSeek 这些家族，目前还只是假设。

**5. 一个我没看到但很想知道的事**：gate 训练数据规模是 7,396 样本 / 模型（feature-family ablation 范围），看起来不算巨大。AG 的"强结果"是否在更大规模的 held-out 上仍然成立？这是上线前必过的一关，论文自己也承认了。

---

## 9. 关键工程经验

如果我自己在团队里推进 AG 落地，会做这几件事：

1. **先在 tool-structured 上小流量 AB**。Qwen3-8B Hermes JSON mode +11.36 pp 是最大的诱惑，但 strict fix/harm 是 22/27（净负）——只能拿 partial-credit 涨分说话。Qwen3.5-9B Hermes function-call +7.64 pp 配 388/317 strict 净正，是更稳的第一站。
2. **gate 必须用 prefix/format + 早 token trajectory 特征**。Qwen3/Qwen3.5 工具路线上这两类特征是 bootstrap CI 阳性的"主菜"，单独 P/e/config 都不够。
3. **固定 P × e 的 matched grid**。别做 dense sweep——既贵又容易 best-cell 过度解释。e0.93 / e0.95 + P1s/P2a/P2b 是现成的 sensitive bracket。
4. **NS 和 non-NS 都保留**。System 层面要能"同时跑两类候选"，让 gate 决定。不要 hard-code 一个 family。
5. **P1s / P3 优先**。Hidden 维度，存得下、噪声低，先把它们跑通再考虑扩张点。
6. **fallback 必须有**。任何 AG 路由的"接受阈值"背后都该有一条 baseline 兜底。

---

## 10. 收尾：可证伪的乐观

这篇论文让我比较舒服的地方在于它**没说"AG 万能"**。它说：

- 在 tool-structured、且 routing 对了的条件下，AG 是**可信的 first deployment target**；
- 在 choice 任务、长 QA、跨模型迁移上，**还不行**；
- 当前 learned gate 在 Qwen2.5-7B 上是**失败 case**，不是边界 case。

这种"画清楚可证伪边界"的工作在 LLM 干预领域其实不多见。更多的论文要么只报正样本（"涨了 5 个点！"），要么把负面 case 包装成"未来工作"。

AG 的负面 case 不是失败——它是让 AG 这个 idea 能在 production 里安全使用的"安全网"。Qwen2.5-7B 的 oracle headroom 在、learned gate 抓不住，恰恰说明了"AG 候选空间"和"AG 部署能力"是两件事，得分别证明。

下一步值得追的事，按论文自己的 future work 排：

- **跨家族验证**（Qwen3.6、其他全 attention 家族）；
- **Gate 过拟合独立检验**（新 split、time split、新 dataset）；
- **P1s/P2a/P2b 的机制正交性**——它们真的分别对应"gate-branch correction"和"value-branch"吗？
- **Online 验证**——目前所有结果都是 offline 的；
- **Training-time use**：FFN 轨迹信号能不能反过来帮训练侧挑 noisy 监督、做 activation-aware curriculum？

对工程团队来说，**先把 Qwen3.5-9B Hermes function-call 这条路由上线，做小流量 AB，看 strict fix/harm 是不是真能 hold 住**——是最低成本验证 AG 真实价值的路径。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。*
