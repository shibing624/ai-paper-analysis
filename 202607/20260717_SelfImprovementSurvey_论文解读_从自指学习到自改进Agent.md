---
title: "从自指学习到自改进Agent:97页综述把"快慢双循环"讲透了"
date: 2026-07-17
arxiv_id: "2607.13104"
short_name: "SelfImprovementSurvey"
tags: [LLM Agent, Self-Improvement, Survey, Scaffolding, RL]
---

# 从自指学习到自改进Agent:97页综述把"快慢双循环"讲透了

你有没有这种感觉:去年大家还在聊"Agent 能用工具",今年所有人都在说"Agent 能自己改自己"。从 Reflexion、Self-Refine 到 Voyager、Darwin Gödel Machine,这一波工作像是打开了潘多拉魔盒,一边宣称"我们的 Agent 学会自我进化了",一边又把整个领域搅得四分五裂——同一个方法叫 self-play、self-evolve、self-improve 还是 continual fine-tuning,每个人说法都不一样,你看论文的时候是不是经常一脸懵?

arXiv:2607.13104 这篇由 Zhe Ren、Yimeng Chen、Dandan Guo 等 12 位作者(其中包括 Jürgen Schmidhuber 本人)联合推出的 97 页综述,干的正是把这一地鸡毛梳理清楚。它用一套统一的数学语言把"现代自改进 Agent"这件事说透:Agent 就是 $(\theta, \Sigma)$ 的二元组,自改进就是一个作用于该二元组的算子 $\mathcal{U}$,按"改什么"切成两条快慢路径,按"用什么信号改"再细分子类。

我读完之后最直接的感受是:它不只是一份文献分类目录,而是在用工程语言回答"为什么现在的 Agent 进化是 1990s 经典自指学习思想在 LLM 上的合乎逻辑的延续",同时给出了一份能直接拿来设计系统的形式化框架。

---

## 核心摘要

**痛点**:LLM Agent 领域关于"自改进"的术语爆炸——self-play、self-evolve、self-refine、continual fine-tuning,大家各说各话;现有综述要么只覆盖 FM 微调,要么只看静态 LLM 的自学习,要么把基础模型和 Agent 支架混在一起讲,缺少一个统一的"到底改什么 / 用什么信号改"的形式化视角。

**方案**:把现代 FM-based Agent 抽象成 $\mathcal{A}_t = (\theta_t, \Sigma_t)$,其中 $\theta_t$ 是基础模型参数,$\Sigma_t = (p_t, m_t, \mathcal{T}_t, g_t)$ 是包含 prompt、记忆、工具、控制逻辑的运行支架。自改进被形式化为一个自诱导算子 $\mathcal{U}$:
$$\mathcal{A}_{t+1} = \mathcal{U}\big(\mathcal{A}_{1:t}, \mathcal{E}(\pi_{\theta_t,\Sigma_t}; \Sigma_t, \mathcal{C}_t)\big)$$
其中 $\mathcal{E}$ 是 Agent 自己跑出来的学习信号(轨迹/反馈/批评),$\mathcal{C}_t$ 是任务上下文。

**两条路径**:
- **FM Improvement(慢回路)**:直接改 $\theta$,按信号源分成内在生成演示、内在评价反馈、外在探索经验三类
- **Scaffolding Improvement(快回路)**:固定 $\theta$,改 $\Sigma$,按组件分 prompt、memory、tool、full scaffolding 四档

**效果**:该综述系统组织了 2023–2026 三年间 80+ 篇代表性工作,横跨软件工程、网页导航、游戏、科学发现、具身智能、通用电脑控制六大应用域,并给出 22 个评估基准的覆盖矩阵。

**我的判断**:这可能是 2026 年到现在为止,把"自改进 Agent"这件事讲得最系统的一份工程参考。它不发明新方法,但把别人发明方法的思路用一套形式化语言拼起来,让你能在一个共同框架下比较 OPRO、Darwin Gödel Machine、MemGPT 这类表面上看完全不相关的工作。**缺点是体量太大**(97 页),不是速食读物;**另一个问题是综述本身不可避免的"只提优点,少提失败"**——很多方法在报告里光鲜的数字背后是不是有评测漏洞,它没有深究。

---

## 论文基本信息

- **标题**:Self-Improvements in Modern Agentic Systems: A Survey
- **作者**:Zhe Ren, Yimeng Chen, Dandan Guo, Guowei Rong, Tonghui Li, R. B. Xiong, Qingfeng Lan, Wenyi Wang, Li Nanbo, Yibo Yang, Mingchen Zhuge, Jürgen Schmidhuber
- **机构**:涉及 KAUST、AI9Stars、Snowflake、LAMDA 等多家机构
- **链接**:https://arxiv.org/abs/2607.13104
- **规模**:97 页,12 张图
- **项目主页**:https://selfimproving-agent.github.io/
- **论文仓库**:https://github.com/selfimproving-agent/awesome-Self-Improving-Agents
- **提交日期**:2026 年 7 月 14 日

---

## 为什么需要这份综述

先说几个让我印象深刻的数字,作为问题的注脚:

- 该综述追踪了 2023–2026 年间 80+ 个代表性自改进工作,横跨 FM 改进与支架改进两条路径
- 评估部分列出 22 个基准,涵盖 scaffolding-level 和 FM-level 两类
- 自我引用链里,Schmidhuber 本人作为合著者把自己 1987 年、1993 年、2003 年的经典工作全部拉进来

如果你之前一直在追这个领域,大概会有同感:每月都有新的"自改进"方法冒出来,声称在某基准上达到了 SOTA,但你很难判断这些方法彼此之间到底是什么关系,是互补还是同质,是工程改进还是范式突破。

该综述的出发点是,目前关于自改进的文献至少在三个维度上是混乱的:

**维度一:概念不统一。** 同样一个东西,有人叫 self-correction(自纠错),有人叫 meta-prompting(元提示),有人叫 self-play(自博弈)。底层机制其实是同一类,但因为术语不一样,大家读论文时容易"看似不同方法,实则同一思路"。

**维度二:覆盖不完整。** 现有综述要么只看 FM 自进化(忽略支架侧),要么只看支架自进化(忽略 FM 侧),要么把静态 LLM 的自学习(比如 RLHF)当成 Agent 自改进的同义词——但其实"Agent 在环境里跑出来的反馈驱动系统级持久变化"和"RLHF 在固定数据集上拟合奖励"是两件事。

**维度三:历史脉络断裂。** 现代工作很少追溯到 Schmidhuber 1987 年的自指学习框架、1993 年的自修改神经网络、2003 年的 Gödel Machine,但这些思想其实正是今天 LLM Agent 自改进的"精神祖先"。作者直接把这个家谱补上,等于给整个领域立了一块"寻根"碑。

作者还和几份已有综述做了横向比较(下面用 $\checkmark$ 表示覆盖到,$\circ$ 表示部分覆盖,$–$ 表示没覆盖),可以一眼看出本文的定位:

| 维度 | 本综述(2026) | Gao et al.(2026) | Fang et al.(2025a) | Tao et al.(2024) |
|---|---|---|---|---|
| Agent 形式化 | ✓ | ✓ | ✓ | $\circ$ |
| 定义范围 | ✓ | ✓ | $\circ$ | $\circ$ |
| 历史根脉 | ✓ | – | $\circ$ | $\circ$ |
| 信号视角 | ✓ | $\circ$ | ✓ | ✓ |
| 更新基底 | ✓ | ✓ | ✓ | $\circ$ |
| 评估视角 | ✓ | ✓ | ✓ | ✓ |
| 领域覆盖 | ✓ | $\circ$ | $\circ$ | – |
| 展望与问题 | ✓ | $\circ$ | ✓ | $\circ$ |

*表 1:四份代表性自改进综述的覆盖对比(直接引自论文 Table 1)*

你看,这份综述在"Agent 形式化 + 历史根脉 + 领域覆盖"这三项上是补全了现有文献的盲区的。这背后是有信号的——把 Schmidhuber 拉进来合著,等于明牌告诉读者:我们要把"自指学习"的传统跟"现代 LLM Agent"接起来。

---

## 形式化核心:Agent 就是 $(\theta, \Sigma)$

先抛开所有工程细节,看该综述给现代 FM-based Agent 立的形式化定义,这是我读下来最受益的部分。

**第一步,定义一个 Agent 的配置**:
$$\mathcal{A}_t = (\theta_t, \Sigma_t)$$
其中 $\theta_t$ 是基础模型(FM)的参数,$\Sigma_t$ 是运行支架(scaffold)。注意这个"支架"是泛指——包括 prompt、记忆、工具接口、控制逻辑,以及 agent 的"执行 harness"。综述里还特意澄清:业界有时叫 agent harness,该综述统一用 scaffold 这个词,理由是 harness 容易让人想到固定的"约束框架",而 scaffold 强调"可被修改的结构",这是该综述的核心立场。

**第二步,把支架拆成四元组**:
$$\Sigma_t := (p_t, m_t, \mathcal{T}_t, g_t)$$

| 符号 | 含义 | 举例 |
|---|---|---|
| $p_t$ | 提示模板/系统指令 | System prompt, 角色设定 |
| $m_t$ | 记忆机制 | 短期 KV cache, 长期 MemGPT 式向量库, 经验回放 |
| $\mathcal{T}_t$ | 外部工具集 | 浏览器、代码解释器、搜索引擎、API 包装 |
| $g_t$ | 控制逻辑 | 路由、调度、安全约束、多 Agent 编排 |

**第三步,定义自改进算子**:
$$\mathcal{A}_{t+1} = \mathcal{U}\big(\mathcal{A}_{1:t}, \mathcal{E}(\pi_{\theta_t, \Sigma_t}; \Sigma_t, \mathcal{C}_t)\big)$$

其中 $\mathcal{E}$ 是 Agent 自己的执行过程,它能跑出三类学习信号(下面会展开);$\mathcal{U}$ 是更新算子,它把信号"承诺"(commit)到 $\theta$ 或 $\Sigma$ 上。

**第四步,关键是"持久 vs 瞬时"**。综述反复强调一个区分:Agent 在跑任务时也会更新它的瞬时状态 $X_t$(比如 KV cache、临时 working memory、对话历史),但 $X_t$ 不算"自改进"——因为它任务结束就清空了。**只有 $\theta$ 或 $\Sigma$ 的变化才叫自改进,因为这是"跨交互持久"的**。

![图 1:现代自改进 Agent 的整体范式图——把 Agent 抽象成 $\mathcal{A}_t = (\theta_t, \Sigma_t)$,按"改什么"切成 FM Improvement 和 Scaffolding Improvement 两条主路,再按"用什么信号"细分](https://arxiv.org/html/2607.13104v1/x1.png)

*图 1:整篇综述的鸟瞰图。左边蓝色框是 FM Improvement(改 $\theta$),右边绿色框是 Scaffolding Improvement(改 $\Sigma$)。下方那排小图(软件工程/网页导航/游戏/科学发现/具身/电脑控制)是后文 6 个应用域的图示。*

这个框架一立起来,后文所有具体方法都被归到一张二维表里:**横轴是改 FM 还是改支架,纵轴是信号类型**。这种结构化分类的价值,在于让你能快速判断"我手头这个新论文到底属于哪一格,跟谁可比"。

---

## 两条改进路径:慢的 vs 快的

### 慢回路——FM Improvement(改 $\theta$)

形式化地:
$$\theta_{t+1} = \mathrm{IMPROVE}_\theta(\theta_{1:t}; \mathcal{S}_t), \quad \Sigma_{t+1} = \Sigma_t$$

**核心特征**:每次更新都通过梯度下降把改动"烧进"模型权重,代价高、周期长,但收益是能力会被压缩进 $\theta$,跨任务泛化。

按信号源 $\mathcal{S}_t$ 切成三种子范式:

#### 1. 内在生成演示 $\mathcal{S}_t \approx \mathcal{D}_t$

Agent 自己合成训练样本(指令-回答对、推理轨迹、执行日志),然后做 SFT 或 RL。代表方法:Self-Instruct、Self-Rewarding LM、Self-Guide、Magpie、UI-Genie。

工程直觉:其实这就是把"FM 自身的世界知识"作为无穷数据源。但这里有个被很多论文忽略的隐患——**用同一个模型生成的数据再去训练它,容易强化既有偏见,且在生成质量边界处陷入"自循环确认"**。后文讨论部分也会提到这一点。

#### 2. 内在评价反馈 $\mathcal{S}_t \approx e_t$

Agent 自己做 reward / preference / critique 来监督自身。代表方法:RLAIF、Constitutional AI、Self-Rewarding、Self-Critique、PPO/GRPO 类 RL。

注意这里有个关键区别:大多数 RLHF 方法的 reward 来自**外部人类标注**,不构成"自改进";而本类工作的 reward 必须**来自 Agent 自身**——可以是同模型的 critique,也可以是 ensemble 自己的偏好投票。

#### 3. 外在探索经验 $\mathcal{S}_t \approx \tau_t$

Agent 跟环境(真实/模拟)交互,收集轨迹和奖励。代表方法:SER、Sport、WebRL、UI-TARS、DeepEvo、SAGE、RePro、EvoAgent、WorldEvolver。这一类最近两年爆发式增长,游戏、具身、SWE 任务里最常见。

按环境真假又分两小类:**Grounded**(真实任务环境)和 **Simulated**(World Model/模拟器)。这个划分很关键,因为 grounded 的反馈信号真实但贵,simulated 廉价但有 sim-to-real gap。

![图 2:FM Improvement 三大子范式的源-动作-信号流图——(a) 内在生成演示走"Data Pool → Generate → Candidate → Quality Filter",(b) 内在评价反馈走"Output → Evaluate → Feedback → Aggregation",(c) 外在探索经验走"Environment → Interact → Trajectory → Processing"](https://arxiv.org/html/2607.13104v1/x6.png)

*图 2:三种 FM 改进子范式的流程图。注意三条路径的"Source"不一样——(a) 是模型自己内部的 Data Pool,(b) 是模型自己输出的 Response,(c) 是外部 Environment。*

### 快回路——Scaffolding Improvement(改 $\Sigma$)

形式化地:
$$\Sigma_{t+1} = \mathrm{IMPROVE}_\Sigma(\Sigma_{1:t}; \mathcal{S}_t), \quad \theta_{t+1} = \theta_t$$

**核心特征**:只改 prompt、记忆、工具、控制逻辑,不碰模型权重。代价低、可回滚、跨任务定制化能力强,但能力天花板被底层 FM 锁死。

按组件切四档:

#### Prompt 优化 $(p_t \rightarrow p_{t+1})$

这是综述里最细的一节。按学习信号 $\mathcal{S}_t$ 的丰富度又分四小档:

| 范式 | 信号类型 | 代表方法 |
|---|---|---|
| Scalar-Feedback Optimization | 标量分数 | OPRO、PromptAgent |
| Qualitative-Feedback Refinement | 语言批评 | Self-Refine、Reflexion、CRITIC |
| Population-Based Evolution | 种群进化 | Promptbreeder、EvoPrompt、遗传类 |
| Textual Gradient Optimization | 文本梯度(对 prompt 求"微分") | TextGrad、ProTeGi |

> 等等,文本梯度是什么?直白说就是 PromptAgent/TextGrad 那一类方法,让 LLM 自己写一句"如果 prompt 是 $\theta$,loss 对 $\theta$ 的梯度是:这条 prompt 在某场景下应该改成更强调 XXX"——形式上像 gradient descent,实际是用自然语言表达"对改进方向的语言描述"。这个抽象挺巧妙的。

![图 3:Prompt 优化的四范式鸟瞰图——从 Init Prompt 出发,经 Agent Core 执行+评估,产出 Candidate Prompt,通过不同范式迭代](https://arxiv.org/html/2607.13104v1/x7.png)

*图 3:Prompt 优化四范式汇总。右侧四块对应上文表格的四个范式,可以看到 Black-Box Optimization 是最朴素的"试 prompt 看分数",Textual Gradient 是最聪明的"用 LLM 写一句'梯度'指导更新"。*

#### 记忆演化 $(m_t \rightarrow m_{t+1})$

记忆是综述里**信息密度最高的一节之一**。它把 memory 拆成三个轴:

- **Memory Object(存什么)**:processed trails、curated raw content、integrated external knowledge、latent embeddings 四类
- **Memory Structure(怎么组织)**:flat、hierarchical、graph-based、vector retrieval 四种拓扑
- **Memory Processing(怎么操作)**:写入、读取、更新、删除四类原语,加一个"信号驱动循环"(Observe & Detect → Structured Organization → Read On Demand → Plan & Act → Evaluate → Update / Delete)

综述还给了一个**Memory-Object 评分卡(qualitative scorecard)**,在 persistence / fidelity / interpretability / compactness / write cost / auditability 六个维度上对四类 memory object 打 1–5 分,并列出常见失败模式:

| 对象类型 | 持久性 | 忠实度 | 可解释性 | 紧凑度 | 写入代价 | 可审计性 | 典型失败模式 |
|---|---|---|---|---|---|---|---|
| Processed trails(教训/例程/摘要) | 中-高 | 中 | 高 | 中 | 低 | 高 | 摘要偏差、过时启发、弱信用归因 |
| Curated raw content(原始素材) | 高 | 高 | 中 | 低 | 高 | 中 | 上下文膨胀、检索噪音、隐私泄漏 |
| Integrated external knowledge | 高 | 中 | 高 | 高 | 中 | 中 | 接地失败、陈旧不一致、工具脆性 |
| Latent embeddings | 中 | 低 | 低 | 高 | 低 | 低 | 漂移/污染、难调试检索、隐性损坏 |

*表 2:Memory-Object 评分卡(综合自论文 Table 3)*

这个评分卡的"反 AI 套话"程度值得肯定——它没有说"各有优劣",而是直接告诉你**哪类失败模式最致命**。比如 Processed trails 的最大风险是"摘要偏差",意思是 LLM 写总结时容易把边缘信息丢掉,导致 Agent 学到错误经验。这跟我们做实验时碰到的问题一致。

![图 4:Memory 的整体框架图——左半是 Object + Structure 静态分解,右半是 Processing 动态循环](https://arxiv.org/html/2607.13104v1/x8.png)

*图 4:Memory 双视角图。左边紫框告诉你"存什么 + 怎么组织",右边黄圈是"信号驱动的处理循环"。注意最右边的循环是闭环的——Observe → Organize → Read → Plan → Evaluate → Update,本质是 PDCA 戴明环。*

#### 工具治理 $(\mathcal{T}_t \rightarrow \mathcal{T}_{t+1})$

这是综述里**比较抽象但提出一个有意思三角**的一节。综述把工具治理拆成三个互相制衡的能力:

- **Autonomous Tool Creation**:Agent 写新工具(典型如 Voyager 在 Minecraft 里自动写技能函数)
- **Iterative Tool Refinement**:基于使用反馈修改工具(典型如 Toolformer,Chameleon)
- **Dynamic Tool Routing**:在工具库里做动态路由(典型如 AnyTool,ToolLLM,α-UMi)

这三个能力形成一个三角,综述叫它 **Tool Governance Metacognition(工具治理的元认知)**。它的意思是:Agent 不光要会用工具,还要会**反思自己是怎么用工具的**,并且能在运行时重写工具。

![图 5:工具治理元认知三角——Autonomous Tool Creation、Iterative Tool Refinement、Dynamic Tool Routing 三个能力互相支撑](https://arxiv.org/html/2607.13104v1/x9.png)

*图 5:Tool Governance Metacognition 三角。这张图视觉上很"硬核",三个角各代表一种能力,中间是统一的元认知机制。*

#### 全支架改进 $(\Sigma_t \rightarrow \Sigma_{t+1})$

这是支架改进的"集大成"档。综述把它跟前面三个组件级改进区分开,理由是:全支架改进是在更高的抽象层(整个 Agent 代码、整个控制流)上做修改,而不是局部组件。代表方法:

- **Self-Referential Code Update(Darwin Gödel Machine)**:Agent 改自己的源码,跑 benchmark 验证,形成树形 archive
- **Generate-Test-Patch Loop(自演化代码 Agent)**:生成候选 → 测试 → 打补丁
- **Open-Ended Search over Agent Designs(ADAS、AIRA)**:把 Agent 架构本身当作搜索空间

![图 6:全支架自改进的迭代图——从 t-1 到 t 到 t+1,每代都改 Scaffolding($\Sigma$ 但固定 FM $\theta$),能力单调上升](https://arxiv.org/html/2607.13104v1/x10.png)

*图 6:全支架改进的时间线。横轴是迭代轮次,纵轴是能力。每一代 Agent 都是 $(\theta_t, \Sigma_t)$,但 $\theta$ 锁住只改 $\Sigma$。注意这是示意图,实际能力曲线不会真的单调——综述后面会专门讨论"参数压缩会丢掉罕见恢复策略"这个问题。*

---

## 历史脉络:从最小二乘到 Gödel Machine

该综述最让我佩服的一点,是把"自改进"这件事的历史拉到 1790s 末(高斯的最小二乘法)。它把整个发展切成五个阶段:

![图 7:自改进的理论根源时间线——从 1790s 的"早期自改进机器"概念,经符号主义、连接主义,走到今天的 FM-Agent 范式](https://arxiv.org/html/2607.13104v1/x4.png)

*图 7:自改进思想史。1790s-1960s 是概念萌芽(最小二乘、哥德尔算术化、Good 的"智能爆炸"猜想),1960s-1980s 是符号主义下的启发式自修改(von Neumann 自再生自动机、Lenat 的 AM/EURISKO),1980s-2000s 是连接主义下的元学习(Schmidhuber 的自指学习、Hochreiter 的 LSTM 早期工作),2000s-2020s 是形式化与架构级(NAS、Self-play、RL),2020s 至今是 FM-Agent 范式。*

**几个里程碑我必须拎出来说一下,因为这跟该综述的"统一性"立场直接相关**:

- **1987,Schmidhuber 的自指学习**:一个系统能生成并评估自己的"后代版本"——这是现代 self-play 思想的鼻祖
- **1993,Schmidhuber 的自修改神经网络**:网络能修改自己的权重——可计算系统能修改自己这件事的早期实证
- **2003,Gödel Machine**:一个能**在数学上证明自身修改能带来期望效用提升**才进行改写的自指系统——理论上自改进的"天花板"
- **2015,Schmidhuber 的"Learning to Think"**:通用控制器学会动态查询"世界模型"生成推理序列——这被作者视为现代 chain-of-thought / ReAct 的精神祖先

读到这段你会意识到:今天我们觉得惊艳的"Agent 自己改自己"——改 prompt、改工具、改记忆,说到底还是 **1990s 的自指学习思想在 LLM 这个高维语义空间里的合乎逻辑的延续**。FM 的"自然语言统一接口"只是把当时那个"在低维权重空间里艰难搜索"的过程变得可行。

这也解释了为什么把 Schmidhuber 列进作者群——他既是这条脉络的开创者,也是今天所有 LLM Agent 自改进工作的"思想家长"。

---

## 应用域:六个领域,各有什么玩法

综述把自改进 Agent 的落地切成六个域(配合表 5 做了环境特性对比),我挑几个印象最深的讲。

### 1. 软件工程(SWE)

最稠密的反馈源,编译器 + 单测 + CI,所以最自然的两条路径都跑得通:
- **支架改**:Darwin Gödel Machine 直接改自己源码,跑 SWE-bench 验证;Live-SWE-agent 在真实任务里编辑自己支架
- **FM 改**:SWE-RL 把测试结果当 reward 训 RL;SWE-RM 训练专门的 reward model

**我的吐槽**:这个领域现在有个评测陷阱——很多方法专门针对 SWE-bench 的 GitHub issue + 测试格式调优,**有可能"过拟合到 benchmark 的测试协议",而不是真的能解决任意 SWE 任务**。综述承认这一点但没深挖。

### 2. 网页导航

模拟浏览器环境(比如 WebArena、Mind2Web)提供可控反馈,代表方法:WebRL、InfoAgent、WebVoyager。优势是状态可重置、成本可控;挑战是真实网页 DOM 在变,sim-to-real gap 不小。

### 3. 游戏与策略推理

最经典的 self-play 舞台:Voyager(Minecraft)、Cicero(Diplomacy)、AlphaProof(数学证明)、AI Economist(经济策略)。**游戏是少数能给出"长程、稠密、有意义"反馈的域**,所以 self-play 类方法在这里效果最稳。

### 4. 科学发现

执行反馈来自模拟器和领域工具,代表工作:ChemCrow(化学)、Coscientist(自主实验)、LLaMPO(材料)。**这个域最稀缺的是"领域专家级 critique"**,所以很多方法把文献检索和 LLM 自身 critique 拼起来,效果不错。

### 5. 具身 AI 与机器人

主要跑在仿真(MuJoCo、Isaac Sim)+ 少量真实数据,代表工作:RT-2、Pi-0、OpenVLA、HGM。**这个域的核心瓶颈是 sim-to-real gap**,所以"工具 = 真实机器人控制栈"反而比"工具 = 浏览器"更关键。

### 6. 通用电脑控制

这个最硬核——OS-level Agent,代表是 OpenAI 的 Operator、Anthropic 的 Computer Use、UI-TARS。**反馈信号极稀疏**(完成没完成),**失败代价极高**(误删文件),所以这个域更倾向保守的支架改进,FM 改进很少。

![图 8:六大应用域的鸟瞰图——SWE、Web、Game、Scientific、Embodied、Computer Control,每个域对应的环境和典型任务列在下方](https://arxiv.org/html/2607.13104v1/x11.png)

*图 8:六应用域一览。注意图最上方是 FM Improvement 和 Scaffolding Improvement 两条主路径,下面六个方块代表六个域,每个方块里是该域典型的子任务。*

---

## 评估:过程 vs 终点

综述用一个我觉得很到位的形式化定义来谈评估:
$$m_t = \mathbb{E}_{x \sim \mathcal{D}_{\text{eval}}, \tau \sim \mathcal{A}_t(x)}[\Phi(x, \tau)]$$

也就是说,**评估不再是一次性"期末考试"**,而是看 $\mathcal{A}_t$ 在 $t$ 时刻的期望得分,关注**性能轨迹**而不是单一峰值。

按 $\Phi$ 的具体形态分两类:

**Metric-based**:可执行验证器(单元测试、仿真器),给出 0/1 或连续分。SWE、数学证明、机器人仿真都属此类。**问题**:反馈稀疏、噪声大(比如 flaky test),且容易"过拟合到 benchmark 的测试协议"。

**Judge-based**:用另一个 LLM 当 judge(LLM-as-a-Judge),适合开放式任务(创意写作、对话质量)。**问题**:Judge 自己可能被 Agent over-optimize(也就是 reward hacking)。

综述还专门讨论了"机制级 benchmark vs 领域级 benchmark"——前者测 Agent 组件能力(比如 G-Memory 测记忆、ReasoningBank 测推理),后者测端到端任务(SWE、WebArena)。这个区分对工程选型很有用:**你该用机制 benchmark 诊断具体组件,还是用领域 benchmark 测整体能力?答案通常是两者都要跑**。

附录里的图 12 把 22 个 benchmark × 15 个代表性方法的覆盖矩阵画了出来,这是个非常实用的"工具对照表",做技术选型时直接看哪个 benchmark 覆盖你关心的能力,哪个方法支持这个 benchmark。

![图 9:22 个 benchmark × 15 个方法的覆盖矩阵(附录图)](https://arxiv.org/html/2607.13104v1/x12.png)

*图 9:Benchmark 覆盖矩阵。绿色块是 scaffolding-level benchmark,蓝色块是 FM-level benchmark。最容易看出来:大多数自改进方法在 scaffolding-level 评测上覆盖广,但 FM-level 评测(数学/编程/常识)覆盖稀稀拉拉。*

---

## 讨论:三件被反复强调的事

### 1. 快探索 vs 慢巩固

$\Sigma$ 的修改是快回路(改 prompt 几分钟搞定、可立刻回滚),$\theta$ 的修改是慢回路(一次 SFT/RL 跑几小时、难定位哪个改动导致回退)。

**综述给了一个工程建议**:嘈杂环境下,先在支架上迭代验证,确认行为稳定后再蒸馏到 $\theta$。**这条建议我非常认同**——我们之前做项目时也是先在 prompt 层调好再蒸馏,直接对 $\theta$ 动刀风险太大。

但它也指出**参数压缩的有损性**:把复杂轨迹蒸馏到权重里,其实是在"取平均",那些稀有的"错误恢复"策略往往就被丢掉了。这是非线性损失,不是简单的精度下降。

### 2. Critic 作为攻击面

把 judge 嵌进闭环训练,等于给 Agent 留了个"刷分通道"——它会学会让 critic 满意,而不是让 ground truth 满意。**综述因此把 critic 视为"被治理的基础设施"**,而不是中立的评委。

它给的工程对策:**生成器和验证器解耦**;验证器的更新必须受人类审计约束,只能做"单调"变化(比如只能加更严的测试,不能放水)。

### 3. 分层门控(Full Scaffolding 改进的必做项)

**全支架自改进最危险**:Agent 能改自己的源码 / 控制逻辑,意味着一次 prompt 注入可能进化成持久性架构漏洞(被污染的记忆 / 被劫持的工具调用被 commit 进下一版)。

综述给的解法:**任何对 $\Sigma_{t+1}$ 或 $\theta_{t+1}$ 的提交,必须经过验证器门控**——覆盖功能正确性、工具权限边界、对随机状态扰动的鲁棒性。"改进"必须发生在"显式定义 + 持续审计"的安全边界内。

---

## 未来方向:六大研究问题

综述把未来方向切成两大主题、共六条具体问题:

**主题 A:终身自适应的算法范式**
1. **Test-Time Continual Adaptation**——测试时持续适应(部署后还在改)
2. **Active Exploration and Curiosity**——主动好奇心驱动探索
3. **Parametric Distillation and Joint Optimization**——参数蒸馏 + $(\theta, \Sigma)$ 联合优化

**主题 B:开放世界的复杂性与鲁棒性**
4. **Resource-Constrained Improvement Dynamics**——资源受限下的效率提升
5. **Multi-Agent Cooperative Co-Evolution**——多 Agent 协同进化
6. **Surviving Open-World Distribution Drift**——抗开放世界分布漂移

**我自己的判断**:第 1 和第 3 是最值得做的——它们把"自改进"从静态 benchmark 推到真实部署,工程价值最高;第 5 也很有想象空间(多 Agent 共享工具 / 测试套件),但离落地远;第 6 是终极挑战(真实 API 一直在变),目前没有靠谱解。

---

## 我的整体判断:这篇综述值不值得读

**该读谁**:做 LLM Agent 工程的研究者、想用一套统一语言跟同行交流的 PM、想选型"该用 Darwin Gödel Machine 还是 Promptbreeder"的技术决策者。

**亮点**:
- 形式化定义干净,$\mathcal{A}_t = (\theta_t, \Sigma_t)$ 这套语言能直接拿来做团队内的术语对齐
- 历史脉络梳理到位,把 Schmidhuber 那条 1987 年的线跟今天 LLM Agent 接上
- 评估部分不仅讲"测什么",还讲"怎么测才不被刷分"
- 22 × 15 的 benchmark 矩阵是实打实的工程选型工具

**问题**:
- **97 页太长,信息密度不均**——很多方法在表格里被一行带过,需要去翻原文
- **偏重"分类"轻"对比"**——把工作分到格子里后,缺一张"哪个格子的方法在工程上最实用"的横向比较表
- **对"自改进是否真的带来能力提升"这个根本问题没正面回答**——很多被列出的方法在同一 benchmark 上差异其实很小,但综述没有批判性地指出
- **没怎么谈 safety 实操**——分层门控、人类审计这些都讲了,但没给具体可复现的实现细节

**一个你可能想问的问题**:我之前已经在追这个领域了,这份综述对我有增量价值吗?我的回答是——**有,但增量主要是形式化语言**。它不会告诉你哪个新方法在 SOTA,但会给你一个"快速定位任意新方法属于哪个格"的坐标。这点对写 review、做选型、写 proposal 都很有用。

---

## 写在最后

自改进 Agent 这个领域,正处于从"概念百花齐放"向"工程收敛"的拐点。两三年前大家还在讨论"LLM 能不能改自己的 prompt",现在已经能稳定做"持续 30 轮自我训练不掉点"了。下一个瓶颈不在算法,而在评估和治理——你怎么证明这个 Agent 是真的"在变好"而不是"在针对 benchmark 刷分"。

arXiv:2607.13104 这份综述,正好在这个时间点提供了一套"可以拿来吵架"的形式化语言。不管你最后选择哪条改进路径,$\mathcal{A}_t = (\theta_t, \Sigma_t)$ 这套表述都值得刻进脑子里。

---

*觉得有启发的话,欢迎点赞、在看、转发。跟进最新 AI 前沿,关注我。*
