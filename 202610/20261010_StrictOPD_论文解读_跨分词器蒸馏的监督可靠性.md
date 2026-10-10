# 对齐覆盖率是个陷阱：跨分词器蒸馏，监督越"全"反而越差

做蒸馏的人大概都有过这种直觉：教师和学生的分词器不一样，那对齐不上的位置就是"丢失的监督信号"，想办法把它们找回来，效果总该更好吧？

这篇论文直接给了这个直觉一记耳光。阿里云团队把跨分词器 On-Policy Distillation（OPD）拆开量了一遍，发现：严格对齐的 1:1 位置已经覆盖了 85%–97% 的学生 token；把剩下对不齐的 mismatch 区间也补上监督，覆盖率是 100% 了，准确率反而掉了。18 个正权重配置，无一幸免。

## 核心摘要

跨分词器蒸馏的痛点是：同一段文本，教师和学生的 token 边界不同、词表不同，两边的概率分布没法直接比。此前的工作（比如 SimCT、Byte-Prefix Marginalization）都在想办法"找回"对不齐的监督。这篇论文反其道而行，先做诊断再下结论：严格 1:1 对齐位置上，共享词表平均保留了教师 99.69%–99.90%、学生 98.99%–99.81% 的概率质量；只用一个学生自己选出来的 top-16 子集算 reverse KL，就能保住 full 共享词表蒸馏至少 96% 的提升幅度，还全面超过 ULD、Extended ULD、GOLD、SimCT 四个 baseline。而给 mismatch 区间加 span log-probability MSE 监督，梯度方向跟主损失弱相关甚至负相关，而且梯度模长相对主损失还在涨——覆盖补全了，信号却是"脏"的。论文的口号值得记住：别再追 alignment coverage，去看 supervision reliability。

## 论文信息

- **标题**：Rethinking Cross-Tokenizer On-Policy Distillation: From Alignment Coverage to Supervision Reliability
- **作者**：Bingxi Hou、Guochao Jiang（共同一作，通讯）、Guofeng Quan、Weiqing Li、Wenfeng Feng、Guohua Liu、Yuewei Zhang
- **机构**：Alibaba Cloud Computing
- **链接**：https://arxiv.org/abs/2610.08448 （2026 年 10 月 6 日提交，arXiv ID 2610.08448）
- **代码**：https://anonymous.4open.science/r/Cross-Tokenizer-OPD

---

## 📖 背景：分词器不同，蒸馏就"对不上表"

先快速对齐一下概念。On-Policy Distillation 的做法是：学生模型自己 rollout 生成回复，教师模型在学生实际走过的前缀上给出逐 token 的分布监督，一般用 reverse KL（MiniLLM、GKD 都是这条路）。同一个分词器下这事很直接，每个位置的分布一一对应。

跨分词器就麻烦了。同一句 "Hello world"，Qwen 可能切成 2 个 token，Llama 切成 3 个；更要命的是两边词表只有一部分重叠，下一个 token 的分布定义在不同的集合上，KL 都没法算。

现有思路大体分两派。一派是换接口：ULD 按排序后的概率做匹配，Byte-Level Distillation 干脆换成字节级接口；另一派是找回丢失的监督：SimCT 把最小对齐的多 token 单元加进监督空间，Byte-Prefix Marginalization 把教师概率质量映射到学生 token 上。隐含假设都一样——对齐覆盖得越多，监督越全，学生学得越好。

这篇论文问了个更基础的问题：严格匹配到底已经留住了多少有用的监督？把丢掉的那部分找回来，对学习到底贡献了什么？

## 🏗️ 方法：把对齐拆成两类，区别对待

先把形式化讲清楚。给定学生生成的回复，用两边的分词器各切一遍，按字符偏移把回复划分成若干"token 组"——每组内学生的若干 token 和教师的若干 token 拼出来是同一段文本。这些组分两类：

| 类型 | 定义 | 能否直接做分布匹配 |
|------|------|-------------------|
| 严格 1:1 组 | 学生 1 个 token、教师 1 个 token 覆盖同一区间 | 能（词表对齐后算 KL） |
| mismatch 组 | 至少一侧需要多个 token 才能拼出同一区间 | 不能，需要额外构造目标 |

对严格 1:1 组，论文的做法是：把两边的分布都限制到**共享词表** $\mathcal{V}_{\cap}$（去掉特殊 token 和歧义项）上，重新归一化，然后算 reverse KL：

$$\mathcal{L}_{1{:}1}(\theta)=\mathbb{E}_{x,y\sim\pi_{\theta}}\left[\sum_{r\in\mathcal{A}_{1{:}1}(y)}\mathrm{KL}\left(\bar{\pi}_{\theta}(\cdot\mid x,y_{\lt i_r})\|\bar{\pi}_{\mathrm{T}}(\cdot\mid x,v_{\lt j_r})\right)\right]$$

对 mismatch 组，则把组内各 token 的概率连乘，得到这个 span 的"路径概率"，用 log 概率的 MSE 去贴教师：

$$\mathcal{L}_{\text{span}}(\theta)=\mathbb{E}\left[\sum_{r\in\mathcal{A}_{\text{mis}}(y)}\left(\log q_{\theta}^{(r)}-\log q_{\mathrm{T}}^{(r)}\right)^{2}\right]$$

总损失 $\mathcal{L}_{\lambda}=\mathcal{L}_{1{:}1}+\lambda\mathcal{L}_{\text{span}}$。$\lambda=0$ 就是只管严格对齐位置；任何 $\lambda\gt 0$ 都实现了 100% 的结构覆盖。这个设计挺干净的——固定严格损失不动，单独扫 $\lambda\in\{0, 0.25, \dots, 1.5\}$，就能隔离出"补全覆盖"这件事本身的因果效应。

![论文核心诊断三联图](https://arxiv.org/html/2610.08448v1/overview.png)

*图 1：Qwen2.5-7B-Instruct → Llama-3.2-3B-Instruct 的三个诊断。左：训练过程中学生侧严格对齐率一直很高，尽管静态词表重叠低得多；中：给 mismatch 组加监督后数学和代码性能都在掉；右：蒸馏前学生回复的严格对齐位置上，共享词表几乎保留了全部教师和学生的概率质量，学生自选的 top-16 子集也保留了大部分。*

## 🧪 实验一：覆盖率的错觉

实验覆盖四个模型系列、三对师生组合：Qwen2.5-7B-Instruct → Llama-3.2-3B-Instruct、Granite-4.1-8B → Phi-4-mini-instruct、Granite-4.1-8B → Qwen2.5-7B-Base（注意第三对是个 base 学生，这个设定挺少见也很有价值）。训练数据是 1 万条 DAPO-Math-17K 数学题加 1 万条 CodeForces 代码题，每条 prompt 采样 1 条回复，跑 100 个 on-policy 迭代，batch 512，lr 1e-6，8 张 H20，SGLang 做 rollout。

第一个发现就挺反直觉的。看表 1：

| 教师 → 学生 | 静态词表 Jaccard 重叠 | 学生 token 严格覆盖率 | 教师 token 严格覆盖率 |
|------------|----------------------|----------------------|----------------------|
| Qwen → Llama | 64.32% | 93.56% | 85.91% |
| Granite → Phi | 39.49% | 96.98% | 97.26% |
| Granite → Qwen | 64.87% | 85.57% | 82.82% |

Granite → Phi 这一对，词表重叠连四成都不到，照"覆盖率焦虑"的逻辑这蒸馏没法做了。结果呢？96.98% 的学生 token 落在严格 1:1 组里。而且按训练窗口（1–20、41–60、81–100 步）拆开看，覆盖率波动最多 1.43 个百分点，非常稳。

说实话这个数字让我愣了一下。词表重叠是按"条目"算的，每个词条权重一样；但真实生成里高频 token 反复出现，词表长尾那 60% 的差异根本轮不上几次出场。静态重叠率和动态覆盖率完全是两回事——这个区分是全文立论的地基。

## 🧪 实验二：补全覆盖，反而掉点

那剩下 3%–15% 对不齐的 token 呢？加上 span MSE 监督试试。结果是三个师生对、六个正权重、一共 18 个配置，**全部**比 $\lambda=0$ 的纯严格监督差，full 平均分掉了 0.27–1.20 个百分点，而且整体随 $\lambda$ 增大单调下行。

覆盖面 100% 了，效果反而差了。这个实验设置的巧妙之处就在于它够简单——不是新方法的工程加成输了，是"补全 mismatch 监督"这个动作本身在拖后腿。

## 🧪 实验三：概率质量集中在哪

为什么严格监督就够了？论文在蒸馏前的学生回复上做了个探针：20k 条 prompt 各采样一条，在严格对齐位置上测两边分布在共享词表上的概率质量。

- 完整共享词表：平均保留教师 99.69%–99.90%、学生 98.99%–99.81% 的概率质量；
- 学生自选的 top-16 子集：保留至少 93.54% 的教师质量、94.55% 的学生质量；
- 从 top-16 扩到 top-128，教师侧最多再涨 3.44 个点，学生侧 2.70 个点，边际收益递减很明显。

关键点在于：这个子集是**学生自己**按概率选的，教师可没参与，但它照样把教师的概率质量兜住了大半。两个模型的"高概率候选"在共享词表上天然就是高度重合的。

## 📊 主实验：top-16 就够了

把严格目标里的共享词表换成学生自选的 top-k 子集（两边都在这同一个子集上归一化再算 reverse KL），跟完整共享词表和四个跨分词器 baseline 对比。full 平均分如下（数学七榜、代码三榜，数学 mean@32、代码 mean@8）：

| 方法 | Qwen→Llama Full | Granite→Phi Full | Granite→Qwen Full |
|------|----------------|------------------|-------------------|
| Base（未蒸馏） | 26.96 | 36.55 | 29.56 |
| ULD | 29.12 | 38.63 | 27.12 |
| Extended ULD | 31.13 | 38.35 | 32.57 |
| GOLD | 27.16 | 41.75 | 46.55 |
| SimCT | 31.59 | 41.68 | 46.10 |
| **Strict full** | **32.86** | 42.49 | **47.35** |
| **Strict top-16** | 32.64 | 42.26 | 47.06 |
| **Strict top-128** | 32.38 | **42.54** | **47.65** |

几个值得注意的点。top-16 在每一对上都保住了 Base → Strict full 提升幅度的至少 96%；三个严格变体在全部三对上的 math 和 full 平均分都超过四个 baseline，top-16 领先最强 baseline 0.51–1.05 个点。ULD 在 Granite→Qwen 上甚至把 base 学生蒸差了（27.12 对 29.56），这类不做 token 身份对齐的排序匹配，碰上 base 模型确实容易翻车。

附录里还有两个加分实验。Qwen3-235B-A22B-Instruct-2507 → Granite-4.1-8B 的大教师设定下，Strict top-16 拿下 62.46 的 full 均分，比 SimCT 的 56.25 高出 6 个多点，AIME-2026 上 38.44 对 27.29，差距拉得相当大；ALFWorld 智能体任务上，Strict top-16 的 overall 成功率 30.34%，不仅超过所有蒸馏 baseline，还超过了冻结教师本身的 29.56%。蒸馏能超过教师，说明学生在自己分布上学到的确实是"提纯"过的行为。

## 🔬 诊断：span 梯度到底哪里不对

为什么加了 mismatch 监督反而掉点？论文做了个我觉得全文最漂亮的分析。在纯严格损失训练的 checkpoint（第 0、20、60、100 步）上，对同一批复放的回复分别算严格损失的梯度 $g_{1{:}1}$ 和 span 损失的梯度 $g_{\text{mis}}$，然后看两个指标：

$$c_{\text{mis}}=\frac{\langle g_{1{:}1},g_{\text{mis}}\rangle}{\|g_{1{:}1}\|_2\|g_{\text{mis}}\|_2},\qquad \rho=\frac{\|g_{\text{mis}}\|_2}{\|g_{1{:}1}\|_2}$$

为了有个参照系，作者还把同一批严格位置随机劈成两半 A、B，算两半梯度的余弦 $c_{\text{ctrl}}$——同一目标下位置子集之间的一致性，算是个"正常水平"的基线。

结果：Qwen→Llama 和 Granite→Phi 的 $c_{\text{mis}}$ 在零附近晃，Granite→Qwen 开局是负的、后来才趋向零；每个 checkpoint 上 $c_{\text{mis}}$ 都低于 $c_{\text{ctrl}}$。更麻烦的是 $\rho$——span 梯度相对严格梯度的模长比，在三对上都随训练**持续增大**。

把两件事合起来看就清楚了：span 监督提供的梯度方向跟主目标弱相关甚至打架，而它的相对嗓门还越来越大。固定权重的 $\lambda$ 下，训练后期这个"脏信号"的影响力一直在膨胀。掉点不是意外，是必然。

不过坦率讲，这个解释是"consistent with"而不是"证明了"。梯度方向弱相关导致掉点，中间还隔着优化动态这一层，论文自己也用词谨慎，只说"may help explain"。我觉得这是诚实的写法，但也提醒读者别把诊断当成因果的铁证。

## 🤔 我的判断

这篇论文值钱的地方不在于提出了什么新方法——Strict top-16 简单到一句话能说完。它值钱在把"覆盖率"这个跨分词器蒸馏里的默认信仰拆穿了，而且拆的方式很扎实：训练对比、概率质量测量、梯度诊断三条线互相印证。跟 SimCT 放在一起看尤其有意思：SimCT 从"找回丢失监督"出发报告了收益，这篇则从"监督可靠性"出发发现找回来的东西有害。两个结论未必矛盾——span MSE 是很粗糙的找回方式，SimCT 的单元选择更挑——但它明确告诉你，这个方向的核心问题不是"覆盖多少"，而是"补回来的信号可不可信"。

也有几个要泼冷水的地方。span 监督只试了 log-probability MSE 这一种形式，换更精细的目标（比如带权重的、或者只在高教师置信度的 mismatch 组上加）结论可能会松动，论文没有扫这个维度。任务也只覆盖了数学、代码加附录的 ALFWorld，开放式生成、多轮对话这类分布更散的场景，严格覆盖率和 top-16 够不够用，还是未知数。另外所有师生对都是 7B/8B → 3B/7B/8B 这个量级，跨度更大的组合（比如 70B 教师蒸 1B 学生）没测。

工程上的启发很直接：如果你在做跨家族模型的蒸馏，先量一下严格 1:1 覆盖率和共享词表的概率质量——大概率你会发现它们已经高得超乎想象，根本不需要引入复杂的跨分词器对齐机制。先跑 strict top-k 这个简单基线，再决定要不要为那百分之几的覆盖率付复杂度税。而"补全覆盖反而伤效果"这个教训，其实不只适用于蒸馏——任何"把更多信号塞进训练目标"的尝试，都值得先问一句：这些信号的梯度方向，跟主目标是一条心吗？

---

**参考文献**

- Hou B, Jiang G, Quan G, et al. Rethinking Cross-Tokenizer On-Policy Distillation: From Alignment Coverage to Supervision Reliability. arXiv:2610.08448, 2026. https://arxiv.org/abs/2610.08448
- Sun J, et al. SimCT: Recovering Lost Supervision for Cross-Tokenizer On-Policy Distillation. arXiv:2605.07711, 2026.
- Wang H, et al. Cross-Tokenizer On-Policy Distillation via Byte-Prefix Marginalization. arXiv:2607.22334, 2026.
- Boizard N, et al. Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs. TMLR, 2025.
- Agarwal R, et al. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. ICLR 2024.
- Gu Y, et al. MiniLLM: Knowledge Distillation of Large Language Models. ICLR 2024.

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
