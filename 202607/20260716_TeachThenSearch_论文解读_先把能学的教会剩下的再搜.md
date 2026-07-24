---
title: "先把能学的教会，剩下的再搜：聊聊 Agentic 视觉生成里那条'知识边界'是怎么被发现的"
date: 2026-07-16
arxiv: 2607.05382
tag: 论文解读
---

# 先把能学的教会，剩下的再搜：聊聊 Agentic 视觉生成里那条"知识边界"是怎么被发现的

你有没有让文生图模型画过 2025 年大阪世博会的吉祥物？我试过，得到的图精美、自信的不得了，但完全是个"非官方的虚构角色"——和真正的官方形象毫无关系。再比如让模型画斯巴达方阵在温泉关的阵型，它会给你一套漂亮的青铜铠甲，但铠甲的形制和历史完全对不上号。

现代生成器在"渲染"这件事上已经卷到飞起，但它们"不知道"的东西，照样会一本正经地编出来。问题不在画笔，而在知识。这篇 arXiv 2607.05382 戳中的，就是这个看起来朴素但其实很少有人系统讲清楚的问题。

## 核心摘要

研究团队从生产级 AIGC 平台挖出 2 万多条真实用户 prompt，归纳出 **12 类失败模式**和 22 个领域，搭出 SearchGen-20K / SearchGen-Bench / SearchGen-Corpus-1M 这一整套可复现的数据集。

第一发子弹：在这套 benchmark 上，主流开源生成器在"搜索密集型 prompt"上崩塌到 21-28/100，比日常 prompt 足足掉了 40 分，**GPT-Image-2 只掉 0.1 分**——因为它背后有联网检索。

第二发子弹：大家自然会想"那就给模型加搜索工具啊"，但论文发现 **naive search 反而把模型本来能画对的 prompt 也搞砸了**（Qwen-Image-2 在 NoSearch prompt 上从 70.7 掉到 60.4）。这背后是搜索噪声对生成器的污染，论文叫它 concept corruption 和 copy effect。

第三发子弹，也是最让我觉得值回票价的部分：作者把根因归结为**生成器特定、且随训练演化的"知识边界“**（knowledge boundary），并提出"先教后搜"两阶段联合训练：先 DPO 把生成器能内化的世界知识压进参数，再 RFT 重新校准 Reasoner 只对生成器学不会的东西触发搜索。在 Klein-4B 上，一个 8B 的 Reasoner 配上 4B 生成器，Phase 2 拿到 31.8 分，**略超**同一个生成器配 Gemini-3-Flash 推理器（31.2 分）。

**我的判断**：这不算是 Agent 工具使用的"端到端 SOTA"刷榜，而是一篇**机制级别的论文**——把"什么时候该检索"这个一直被工程界当成经验调参的问题，给出了一个可微、可证、可迭代的解法。Dataset + Harness + Principle 三件套，方法本身只是 Principle 的最小实现。这条思路一旦泛化到视频、3D、SVG 渲染，潜力远比当前数字看到的要大。

---

## 论文信息

- **标题**：Search Beyond What Can Be Taught: Evolving the Knowledge Boundary in Agentic Visual Generation
- **作者**：Haozhe Wang, Weijia Feng, Jinpeng Yu, Che Liu, Ping Nie, Fangzhen Lin, Jiaming Liu, Ruihua Huang, Jimmy Lin, Wenhu Chen, Cong Wei
- **机构**：香港科技大学、滑铁卢大学、阿里 Qwen 应用团队、帝国理工
- **arXiv**：[2607.05382](https://arxiv.org/abs/2607.05382) (cs.CV)
- **v3 时间**：2026-07-09
- **项目页**：https://haozheh3.github.io/SearchGen
- **协议**：CC BY 4.0，全量数据、模型、推理轨迹可下载

---

## 问题：为什么"会画"不等于"会画对"

先说一个让我自己都很诧异的发现。在标准文生图 benchmark（比如 GenEval、CompBench 之类）上，Bagel、Qwen-Image-2、Nano Banana Pro 等模型差距其实没拉开。但一旦把 prompt 换成"画 2025 大阪世博会官方吉祥物的标准姿势"这种需要**世界知识**的请求，**开源模型平均崩塌 40 分**，而带联网的商用模型几乎纹丝不动。

直白点：现有 benchmark 大多考的是"画得好看不"、"prompt 跟上了没"，而**从来不考模型知不知道自己要画的东西长啥样**。这就解释了为什么论文里 Figure 6 的图让人觉得反差这么强烈——同一批模型在 NoSearch prompt 上差距不大，到了 Search-Intensive 上立刻分层。

![图1：world-knowledge bottleneck，九个生成器在 NoSearch vs Search-Intensive 上的崩塌](https://arxiv.org/html/2607.05382v3/x6.png)

*图 1：world-knowledge bottleneck 在九个生成器上的体现。横轴是模型，纵轴是 0-100 的质量分。灰色条是 NoSearch（不需要世界知识），蓝色条是 Search-Intensive（需要）。开源模型普遍掉 20-40 分，GPT-Image-2 只掉 0.1 分。*

这种崩塌是结构性的，不是模型再训一训就能解决的。原因有两个：

1. **训练语料有 cutoff**。今天训的模型，昨天的新闻、刚出的游戏角色，全是盲区。
2. **真实请求是长尾且不断演化的**。论文里有个细节挺震撼：3.1 万个 unique 视觉实体里，**93.1% 只在一条 prompt 里出现过一次**。任何有限数据集都不可能 cover 这条长尾。

## 数据集：12 类失败、22 个领域、20,839 条 prompt

在分析 2 万多条生产 prompt 之后，作者归出 **12 类失败模式**：

| 类别 | 主导模态 | 代表 prompt |
| --- | --- | --- |
| Temporal – Recent | 双模态 | "2025 大阪世博会吉祥物的官方姿势和配色" |
| Temporal – Current | 双模态 | "当前 FIFA 世界杯小组赛积分榜" |
| Entity & IP | 视觉 | "Jingliu from Honkai: Star Rail 手持冰剑" |
| Concept & Symbol | 双模态 | "不丹国旗，含 Druk 龙" |
| Factual & Historical | 双模态 | "温泉关斯巴达方阵，考古级青铜甲" |
| Cultural Specificity | 双模态 | "Oaxacan 传统 alebrije 龙，含地方色彩规律" |
| Visual / UI / UX | 视觉 | "iOS 17 天气 App 雷暴动画截图" |
| Data Visualization | 文本 | "中国朝代时间线，含准确年份和开国皇帝" |
| Text / Typography | 文本 | "新艺术运动海报，含当时典型展示字体" |
| Complex Composite | 双模态 | "DNA 复制流程 Aztec 风信息图" |
| Vague / Abstract | 文本 | "日本小镇雨下午后思乡情绪" |
| Implicit Reasoning | 双模态 | "宫崎骏风格的山区小屋内部" |

光看这张表就已经能感受到"光靠 prompt rewrite 是不行的"——**有的需要视觉参考，有的需要文本事实，有的两个都要**。这直接决定了 Reasoner 必须**模态感知**地选 image search 还是 web search。

数据集还做了个有意思的设计：**answer-first**——先用前沿模型对每条 prompt 标出它对应的知识缺口（reference slot / text knowledge slot / failure mode），并标 critical/important/moderate/minimal 严重度。每个 prompt 平均 **5.2 个知识缺口**，90.5% 的 prompt 至少 3 个。这种"先标答案再合成 prompt"的做法让自动评估有了金标准。

英文 prompt 平均 266 字符（偏描述性），中文 prompt 平均 89 字符（偏浓缩），**不是翻译模板**而是真实用户行为。这个双语分布挺有诚意。

最让我觉得贴心的是 **SearchGen-Corpus-1M**：他们把所有 search session 全都预跑好、缓存好——145,642 个 session、55.9 万独立 URL、37 万缓存下载。**等于说，复现这条流水线根本不用再花钱调 Google Search API**。这套 dataset + corpus 才是这篇论文真正的"基础设施"贡献，方法部分反而像是这套基础设施的一个验证 demo。

---

## 那"加个搜索 Agent 不就行了"？

直觉上，给模型挂个能搜图的工具，应该能解决问题吧？论文说，**没那么简单**。Naive search 不仅没解决问题，还**让模型本来能画对的 prompt 也变差了**。

先看 Figure 2 对比两种范式：

![图2：Prompt Rewrite vs SearchGen 两种范式对比](https://arxiv.org/html/2607.05382v3/x2.png)

*图 2：传统 Prompt Rewrite（左）让 LLM 把短 prompt 改成长 prompt 给生成器，但生成器还是不知道 FIFA 当前积分。SearchGen（右）让 Agent 主动搜网页，把视觉参考（Image 1/2/3）和文本上下文打包成 multimodal context 给生成器。*

看着挺美。但论文做了 BlindSearch（对每个缺口都搜）和 ReasonedSearch（用 Gemini-3-Flash 决定搜不搜、搜什么）两组对照，在 NoSearch prompt 子集（不需要外部知识的 prompt）上：

| 模型 | Baseline（不搜） | Reasoned | Blind |
| --- | --- | --- | --- |
| Qwen-Image-2 | 70.7 | 76.5 | 60.4 |
| Qwen-Image | 67.4 | 75.0 | 59.5 |
| Flux.2-Klein-9B | 57.8 | 66.4 | 52.3 |

**BlindSearch 在 NoSearch prompt 上让所有模型都掉分**。Qwen-Image-2 相对掉了 14.6%。

这背后是两种结构化的失败模式（论文 Figure 7 里有例子）：

- **Concept corruption（概念污染）**：搜索 fire 了一个本不需要搜的 prompt，搜回来的参考图把生成器内部原本正确的知识覆盖掉了。本质是"gating failure"——该不搜的时候没忍住。
- **Copy effect（拷贝效应）**：搜回来的参考图带了太多无关的视觉信息（背景、构图、色调），生成器把它当成拷贝模板，而不是知识补充。本质是"filtering failure"——不该用的细节没过滤掉。

我之前在用 SDXL + LoRA 做文生图项目时也碰到过类似现象：给模型喂参考图时，它会**把参考图的风格和构图也一并吞下来**。那时候我以为是 LoRA 没训好，原来是 reference conditioning 本身的噪声敏感性问题——生成器没有能力区分"哪些是知识、哪些是噪声"。

---

## 核心概念：Knowledge Boundary

论文把上面这些现象抽象成一个**生成器特定、且会随训练移动的"知识边界“** $\mathcal{B}(\theta)$。

形式化定义有点学究气但思路很清晰：对一个生成器 $G_\theta$ 和知识单元 $k$（实体、文化符号、字体等），如果用搜索补足 $k$ 之后，质量提升小于阈值 $\epsilon$，那 $k$ 就属于“**可内化集合**” $\mathcal{K}_{\mathrm{int}}(\theta)$，否则属于“**上下文依赖集合**” $\mathcal{K}_{\mathrm{ctx}}(\theta)$。两者构成知识边界。

说人话就是：**有些知识是稳定、低维的，模型学一次就会了（比如一个游戏角色的标准造型），再搜就纯属浪费还添乱；有些知识是开放、不断演化、或者长尾到训不出来（比如"当前世界杯积分"），那就必须留在外部**。这条边界是**生成器特定**的——Klein-4B 觉得能内化的概念，Bagel 可能就觉得必须搜。

这个观察太重要了。它意味着：搜索策略不应该是个固定规则，而应该**和生成器一起演化**。这也是为什么作者提出 "teach-then-search"——先用 DPO 强行把边界往外推一推（让生成器把能学的学了），再用 RFT 校准 Reasoner 跟上新的边界。

但有个**细节我得挑出来**：这个 $\mathcal{B}(\theta)$ 论文是**先有现象再有形式化**的——实际训练中没有任何信号直接告诉模型"这条边界在哪"。它是从 co-training 的 reward signal 里**涌现**出来的。这一点其实作者也说得很清楚，但容易被忽略。说到底，**它把"边界标注"这个不可能获得的人工标签，用生成器自己产出的图像质量作为 proxy 信号挖出来了**。这个思路我觉得比方法本身更有借鉴价值。

---

## 方法：Co-Training 框架

整个框架长这样：

![图3：Co-Training 框架——先教生成器内化知识，再校准 Reasoner 只搜它学不会的](https://arxiv.org/html/2607.05382v3/x8.png)

*图 3：Co-Training 框架全貌。上半"Teach Visual Generators What it can Internalize"是 Phase 1 的 online DPO，搜索增强后的 prompt 让生成器采样 M 张图，按质量分构造 (chosen, rejected) 偏好对。下半"Learn to Search What Generators Cannot Be Taught"是 Phase 2 的 Rejection Finetuning (RFT)，Reasoner 重新校准到强化后的生成器上。整个 Reasoner 走三阶段 gate→filter→integrate 协议。*

我把它拆成四块来聊：

### Reasoner 的三阶段协议（解决"噪声从哪来"）

噪声会在三个点注入到生成器的输入里：①决定要不要搜、②选哪条参考、③怎么把参考整合进 prompt。**一个 filter 拦不住所有点**，所以论文做了三阶段：

- **Gate（门控）**：Reasoner 先把 prompt 拆解成知识缺口，按严重度排序，只有 critical / important 才触发搜索。最多 3 个 search query，否则 skip。
- **Filter（过滤）**：搜回来的视觉参考要再过一遍，**只留下"刚好补上缺口"的那部分**，把无关的背景、构图、色调过滤掉。这一步主要防 copy effect。
- **Integrate（整合）**：视觉参考不直接以像素形式丢给生成器，而是**转写成自然语言**的"借用指令"。比如"参照 Image 1，把角色袍子改成青金色"。这样保住了知识，又把像素级噪声挡在外面。

这一步很关键，它把搜索噪声**挡在生成器门外**。但它解决的是"如果搜了，怎么搜得干净"——没解决"该搜不该搜"。

### Phase 0：SFT 暖启动 Reasoner

直接 RL 太贵也太不稳定，所以先用约 1 万条专家标注的 (prompt, Task A/B/C 输出) 做 SFT，对应 gate/filter/integrate 三个任务。结果是 Reasoner 能跑出结构化输出，但**还是生成器无感**的——它对任何缺口都会搜，不区分某个具体生成器到底会不会。

### Phase 1：Online DPO 让生成器"把能学的学了"

这是我最喜欢的部分。流程是：

1. 用 Phase 0 Reasoner 给 prompt 生成"搜索增强版" $\tilde{p}$
2. 生成器在 $\tilde{p}$ 上采样 $M$ 张图（$M$ 不小）
3. 用 SearchGen-Bench 的 9 维 rubric 评分，挑出**最高分**和**最低分**的当 (chosen, rejected) 偏好对
4. 在 flow-matching 的速度场上做 DPO（用 ELBO 替代 intractable likelihood，$\beta=100$，EMA 0.99）

为什么这能扩展知识边界？因为 DPO 的 chosen 和 rejected 都在**搜索增强**的输入上，模型被迫**消化参考图里的事实知识**（比如不丹国旗长啥样），而不是把它当噪声忽略。等训完之后，原本需要搜才能画对的概念，很多就内化进参数了。

这个机制还有个**很妙但没被作者明说的副作用**：因为训练信号一直是有噪声的搜索参考图，生成器被迫学会"参考图可以信，但不能全信"——自然就拿到了对不完美输入的**鲁棒性**。这一步直接给 Phase 2 铺路。

### Phase 2：RFT 校准 Reasoner 到新边界

生成器边界外扩了，但 Phase 0 Reasoner 的策略还停在旧边界上，**会继续对已经被内化的概念瞎搜**。Phase 2 用 rejection-sampling finetuning 重新校准：

1. 对每个 prompt，让 Phase 0 Reasoner 跑 $N_{\mathrm{traj}}$ 条轨迹，每条配 Phase 1 强化后的生成器出图
2. 按 SearchGen-Bench 评分，算 group-relative advantage $A_n = (s_n - \bar{s})/(\sigma_s + \delta)$
3. **只保留 advantage > 0 的轨迹**做 SFT
4. 4×8 GPU 小时跑完

这样 Reasoner 自动学到了“**什么时候不搜**”——不需要任何显式边界标签，边界从 reward signal 里涌现。Phase 0 Reasoner "瞎搜"的轨迹评分低被丢，Phase 0 Reasoner "搜对地方"的轨迹评分高被保留。等训完 Reasoner 就**精准匹配新生成器**了。

整个循环的精妙之处是：**两个相对便宜的训练阶段（一个 DPO，一个 RFT）替代了不可能获得的人工边界标注**。这是论文最值的洞察。

---

## 实验：到底是骡子是马

主实验结果在 Table 6，三个 Phase 加上参考基线，跑了 Klein-4B（flow matching 架构）和 Bagel（unified 视觉-语言架构）两种。

| 配置 | NoSearch | Set I | Set II | Set III | Overall |
| --- | --- | --- | --- | --- | --- |
| **Klein-4B** | | | | | |
| Phase 0: Blind Search + Klein-4B | 54.6 | 28.9 | 29.2 | 21.2 | 26.4 |
| Phase 1: Blind Search + Klein-4B-DPO | 54.0 | 31.8 | 31.1 | 24.7 | 29.2 |
| Phase 2: Generator-Adaptive + Klein-4B-DPO | 56.9 | 34.1 | 33.6 | 27.4 | 31.8 |
| No Search + Klein-4B-DPO（基线） | 49.9 | 28.2 | 26.3 | 20.6 | 25.0 |
| Oracle (Gemini-3-Flash) + Klein-4B-DPO | 55.7 | 33.7 | 33.9 | 26.0 | 31.2 |
| **Bagel** | | | | | |
| Phase 0: Blind Search + Bagel | 52.6 | 24.4 | 26.6 | 19.2 | 23.4 |
| Phase 1: Blind Search + Bagel-DPO | 52.4 | 26.4 | 26.9 | 20.7 | 24.7 |
| Phase 2: Generator-Adaptive + Bagel-DPO | 54.3 | 28.3 | 29.3 | 22.6 | 26.8 |

三个关键发现：

**发现 1：单调性（Monotonicity）**。从 Phase 0 → 1 → 2 在两种架构上都严格单调上升，没有任何 stratum 出现回退。Klein-4B 上 Phase 1（纯生成器 DPO）涨 +2.8，Phase 2（Reasoner RFT）再涨 +2.6，两个阶段**独立贡献**。这避免了"靠牺牲 NoSearch prompt 换 Search-Intensive 提升"那种拆东墙补西墙。

**发现 2：选择性（Selectivity）**。在 NoSearch prompt 上，Phase 2 比 No Search + DPO 基线**多拿 7.0 分**（56.9 vs 49.9）。这意味着 Reasoner **真的学会了"什么时候不搜“**——它在生成器已经会画的 prompt 上主动 abstains，把污染挡在外面了。

**发现 3：生成器特异性（Generator-Specificity）**。最有说服力的一组对照：把为 Klein-4B-DPO 校准的 Reasoner 拿回去配 Klein-4B（基线生成器），Overall 从 31.8 跌到 26.8。Set III 跌得最猛，因为 Reasoner 还在为新生成器能消化的概念搜参考。**这直接证明搜索策略是生成器-Reasoner 对的联合属性，不能跨生成器通用**。

最有冲击力的对比还是这一行：**31.8 vs 31.2**。一个 8B 的 RFT Reasoner 配 4B 的生成器，**略超** Gemini-3-Flash 配同一个生成器。**这等于说，生成器特定的轻量校准在等计算预算下能跑赢通用重型推理**。这给"小模型对齐特定生成器"这种范式打了个很响的样板。

但也有天花板：**和 GPT-Image-2（75.0）还差 40 分**。这部分差距论文说不是框架的问题，是 4B 容量本身的上限——co-training 能从固定生成器里"挤出最大价值"，但没法跨越容量鸿沟。这个判断我同意。

![图4：Co-training 单调推进 + 知识边界右移](https://arxiv.org/html/2607.05382v3/x9.png)

*图 4：(a) Co-training progression——三种 co-training 阶段（Reasoner SFT、Generator DPO、Reasoner RFT）在 Set I/II/III 上的得分都单调上升。(b) 知识边界右移——DPO 前后 Klein-4B / Bagel 在 no-search 质量分上的 CDF 整体右移，阴影区域是从 $\mathcal{K}_{\mathrm{ctx}}$ 迁到 $\mathcal{K}_{\mathrm{int}}$ 的"新内化"知识。*

Figure 9 (b) 是**理解整个 framework 最关键的一张图**。它画的是"不搜的情况下，每条 prompt 的质量分"的 CDF。DPO 之后曲线整体右移——**更少的 prompt 拿到低分，更多的 prompt 拿到高分**。这直接证明了"边界外扩"是真实存在的，不是叙事把戏。

---

## 我的批判性判断

**说几个我觉得论文没说透的地方**：

1. **DPO 阶段的"质量评分"本身就是 SearchGen-Bench 的 9 维 rubric，RFT 阶段也是同一个 rubric**。这意味着 generated preference 完全是用 VLM judge 标的，没有人类反馈。**VLM judge 本身的偏差（比如倾向于高分、倾向于某种风格）会通过 DPO/RFT 注入到模型里**。论文没讨论 judge bias 的传染性。

2. **Phase 1 训完的 DPO 生成器，作者没详细测过它"在没有 Reasoner 配合下的 standalone 表现“**。我们只看到 Table 6 里 No Search + Klein-4B-DPO 在 NoSearch 子集上 49.9，比 Phase 0 的 54.6 反而掉了。**DPO 是不是让生成器在 NoSearch 上"变差"了？** 这其实暗示生成器可能轻微 overfit 到了搜索增强输入上，standalone 能力有点回退。值得在更大实验里验证。

3. **”co-training 涌现出知识边界"这个声明我觉得是 soft claim**。论文只能证明"训练后的 Reasoner 行为和强生成器匹配"，并不能直接测出"Reasoner 内部对每条 prompt 的'知识缺口评估'和真实 $\mathcal{K}_{\mathrm{ctx}}$ 对齐了"。这是个间接证据链条，强度有限。

4. **方法本身依赖的 Reasoner 是 Qwen3-VL-8B**——这是个相当强的开源 VLM。换成更弱的 Reasoner（比如 4B 以下），三阶段协议的可靠性会不会掉？论文没给 ablation。

5. **SearchGen-Bench 的 rubric 9 个维度里只有 4 个是 knowledge-sensitive 的，但 DPO 用的是 9 维总分**。这意味着 DPO 不仅在学"知识"，还在学"渲染质量"——会不会因为这个 DPO 也在学渲染，导致 Phase 1 的"知识边界外扩"被部分归因到"渲染能力提升"上？论文没拆开。

6. **数据集里有 5.2 个知识缺口/条 prompt，实验却没测单缺口 vs 多缺口的难度梯度**。三阶段 Reasoner 在多缺口 prompt 上的可解释性是个隐患。

**它做得好的地方**：

- **数据集 + Corpus + Harness 一体化**才是真正杀手锏。SearchGen-Corpus-1M 的 1M 预缓存搜索结果让任何人都能离线复现，**这是这个领域少见的开放诚意**。
- **问题定义本身比解法更重要**。"Knowledge boundary"这个概念把"什么时候该检索"从经验调参问题提升到了机制问题，给后续工作（包括视频、3D、SVG 渲染）留下了清晰的研究空间。
- **DPO 训生成器 + RFT 训 Reasoner 的两阶段循环**，把"边界标注"这个不可能的标签问题，用**质量分作为 proxy signal 挖出来了**。这个思路我觉得比方法本身更值得借鉴。

**和同期工作比**：

- **Diffusion-DPO（Wallace 2024）** 是把 LLM 的 DPO 移植到 diffusion 模型，用 ELBO 替代 intractable likelihood。本文用的就是 flow-matching 版的 DPO 损失，但**关键差别是数据来源**——本文的偏好对是用 SearchGen-Bench 的知识敏感维度评出来的，不是人标的人类偏好。
- **RAG / Atlas / REPLUG** 这些文搜文 RAG 工作也训 searcher-generator 联合，但**研究领域是文本**。本文的"reference 注入噪声"是视觉模态特有，结论不一定能平移。
- **RAG 里的 Self-RAG（Asai 2024）** 也是让模型自己决定要不要检索，但**是 LLM 自反思**，不是和质量分绑定的 RL。Self-RAG 的检索决策**没有演化机制**，而本文的决策是随生成器一起 co-evolve 的。这是更深的差异。

---

## 工程落地建议

如果你也在做 Agent 工具调用或者 RAG 增强生成，这篇论文有几个能直接借鉴的：

1. **先建一个 Knowledge Boundary 评测**，把 prompt 拆成 NoSearch / Search-Intensive 两类子集。先看你自己的基线在两类上的差距——**这能立刻告诉你"要不要加搜索"以及"加多少搜索“**。
2. **搜索策略要分阶段**。Gate/Filter/Integrate 三段拆开训，比一锅端靠谱太多。Filter 阶段对消解 copy effect 尤其关键。
3. **生成器和 Reasoner 的协同训练是个被低估的杠杆**。哪怕你只做一阶段 DPO，让生成器在"带噪声的检索增强输入"上训练，也能拿到一定的鲁棒性，不一定要走完整两阶段。
4. **天花板提示**：本论文和 GPT-Image-2 还有 40 分差距，主要来自生成器容量。如果你用的是 4B 量级生成器，不要指望靠检索就把分数拉到商用 API 的水平。

---

## 收尾

整篇论文给我最大的冲击不是 "Phase 2 超了 Oracle" 那个数据点——而是一句话：

> The central question is not how to build a model that knows everything, but how to build a system that knows what it does not know.

模型永远学不完开放世界。下一代模型只会把"知识边界"往外推一点，但**不会让边界消失**——因为训练数据是有限的，世界是无限的。co-training 给出的是一套**在任意生成器上、任意规模上发现这条边界**的方法。

这套框架后续往**视频生成、3D 生成、SVG 渲染、Agent UI 控制**等领域的扩展性我非常期待。论文 Conclusion 提到的几个开放问题——"边界能不能从模型内部预测出来"、"不同模态下边界是否同构"、"多轮 co-training 是否还能单调推进"——每一个都是值得博士论文的题目。

最后留一个**说给做产品的人**的话：如果你正在考虑"我们该不该给我们自己的文生图 / 文生视频 API 加联网检索"——**别加裸搜索**。先评估你的 NoSearch baseline 崩塌多少，再决定 Reasoner 的粒度。**盲搜会让你的 NoSearch 用户体感直接掉一档**，这往往比"加上搜索"带来的提升更不容易被发现，因为掉分发生在 baseline 强的 prompt 上。

---

觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。
