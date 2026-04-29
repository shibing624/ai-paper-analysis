# 当"世界模型"成了一个谁都看不懂彼此的词：一份给Agentic AI的能力分级地图

## 核心摘要

如果你最近在不同社区里聊"世界模型"，大概率会有这样的体验——视觉那边的人在说 Sora 这种生成式视频模型，强化学习的人在讲 DreamerV3、MuZero 的潜在动力学，做 Web Agent 的人在讨论让 LLM 模拟网页跳转，搞自动化科学的人在说 A-Lab 那种自主合成实验。同一个词，四套语境，几乎不互通。

这篇 100+ 页的综述（arXiv:2604.22748，作者来自 HKUST、新加坡国立、牛津、南洋理工、港中文等十所机构的 39 人合作）干的事情很直接：给"世界模型"这个被滥用的概念建一套**二维坐标系**——纵轴是三级能力阶梯（L1 Predictor 一步预测 / L2 Simulator 多步可决策仿真 / L3 Evolver 证据驱动自我修订），横轴是四类规律范畴（物理世界 / 数字世界 / 社会世界 / 科学世界）。论文综合了 400+ 篇工作和 100+ 个代表性系统，最有价值的判断在于：**当下绝大多数所谓的"世界模型"还停留在 L1，能像样跑 L2 多步仿真的不多，真正闭环到 L3 自我进化的，目前几乎只有自主科学实验那一类**（CAMEO、A-Lab 这种）。

如果你正在做 Agent、做强化学习、做视频生成或做科学发现的工程，这套分级值得花时间读一遍——它不会教你新算法，但能让你看清自己手里的系统到底在第几层、缺什么。

---

## 论文信息

- **标题**：Agentic World Modeling: Foundations, Capabilities, Laws, and Beyond
- **作者**：Meng Chu, Xuan Billy Zhang, Kevin Qinghong Lin, Lingdong Kong, Jize Zhang, Teng Tu, Weijian Ma, Ziqi Huang 等共 39 人（含 Mike Zheng Shou、Ziwei Liu、Philip Torr、Jiaya Jia 等多位资深作者）
- **机构**：香港科技大学、新加坡国立大学、牛津大学、南洋理工大学、香港中文大学、香港大学、华盛顿大学、香港科技大学（广州）、新加坡科技设计大学、新加坡管理大学
- **arXiv**：[2604.22748](https://arxiv.org/abs/2604.22748)

---

## 为什么需要又一篇 World Model 综述

说实话，看到"又一篇 World Model 综述"我第一反应是嫌弃的。这两年这个方向 survey 出得已经够多了——Ding et al. (2025) 的 understanding vs. predicting 二分法，Zhu et al. (2024) 围绕 Sora 谈生成式世界模型，Yue et al. (2025) 的 G1-G4 视觉世界模型四代论，Kong et al. (2025) 的 3D/4D 世界建模，机器人操作、自动驾驶各有专门 survey……

这些综述有一个共同的问题：**按模态切，按应用切**。视觉的归视觉，RL 的归 RL，结果就是两个研究者拿着同一个词在说完全不同的事情。视觉那边一个研究员可能用生成视频的视觉保真度来评世界模型；强化学习那边的人则只看这模型跑下去能不能让 task success rate 涨。同一个 word，互不相干的指标。

这篇论文的切入角度不一样——它**按"能力"切，按"规律"切**。能力是垂直维度：会不会预测一步？会不会预测多步还满足领域约束？会不会自己发现错了去改自己？规律是水平维度：你在什么"世界"里建模——是物理这种有解析方程可以验证的，还是社会这种连转移函数都靠共同信念支撑的？

这个视角的好处是，它能把分散在不同社区的工作放进一个统一坐标系。你拿着一篇论文，能立刻判断：它在 L 几？哪个 regime？还差什么？如果你做 robot manipulation 想看看代码 agent 那边有没有可以借鉴的 L3 思路，这套地图就有用——这是我读完之后最大的 takeaway。

![图1：本综述相对其他世界模型与Agent综述的定位。论文位于中央，整合了跨域覆盖与基于能力的分类法 L1/L2/L3 × 四类规律](https://arxiv.org/html/2604.22748v1/x1.png)

*图1：四类相邻综述各自只覆盖了世界模型版图的一部分（具身世界模型、生成式世界模型、语言Agent、AI for Science），这篇综述试图用 L1/L2/L3 × 四类规律的能力坐标系把它们统起来。*

---

## L1 / L2 / L3：三级能力阶梯，是怎么划分边界的

我特别喜欢作者用哲学传统来引出三级划分这个动作——不是装腔作势，而是为了让"能力边界"这件事**有 testable 的判定标准**。这是这篇 survey 值钱的地方之一：每一级都有可以验证的边界条件，不是给你抛个抽象口号。

### L1 Predictor：休谟式的常规联结

L1 干的事情其实就一句话：给定历史观测和动作，预测下一步会发生什么。形式化地，它包含四个局部算子：

- **State Inference**: $z_t = q_\phi(z_t \mid o_{\leq t}, a_{\leq t-1})$ — 把观测历史压成潜在状态
- **Forward Dynamics**: $p_\theta(z_t \mid z_{t-1}, a_t)$ — 核心算子，一步状态转移
- **Observation Decoder**: $p_\psi(o_t \mid z_t)$ — 反向重建观测
- **Inverse Dynamics**: $\pi_\eta(a_t \mid z_{t-1}, z_t)$ — 从前后状态反推动作

这一级最经典的代表就是 Dreamer 系列的 RSSM（Recurrent State Space Model）、MuZero 的 latent dynamics、JEPA 系列的 masked latent prediction，再加上 VAE/VQ-VAE/Diffusion 这些表征学习的 building block。

哲学上，作者把它对应到休谟的 **constant conjunction**——你只是在记录"过去观察到 A 之后总跟着 B"，没有声称你知道为什么。这个比喻很贴切：L1 模型在分布内表现可以非常好，但分布一变就崩，因为你从来没真正建模过因果。

L1 的天然限制就是：**单步预测好，不代表多步组合后还能用**。一步误差 0.1，叠加 50 步直接漂到月球上去了。这个 compounding error 问题是 L1 系统跨不过去的鸿沟，也是 L2 之所以是"另一个能力级别"而不是"更好的 L1"的根本原因。

### L2 Simulator：可以拿来做决策的多步 rollout

L2 的边界条件是这篇 survey 我觉得最有 operational value 的部分——它给出三个**可测的 boundary condition**，过了才算 L2：

1. **Long-horizon coherence（长程一致性）**：rollout 到 H 步还能用，不是一步两步看着合理然后立刻塌
2. **Intervention sensitivity（干预敏感性）**：你改个动作，trajectory 应该有方向上正确的变化，不是无脑收敛
3. **Constraint consistency（约束一致性）**：生成的未来要符合该领域的支配性规律

第三条是最容易出问题的。Sora 看着惊艳，但杯子穿桌、重力反向这种事说出来你也都见过——这就是典型的 L1 强但 L2 差，**视觉看起来真**和**物理上自洽**完全是两回事。作者在文中也直接点了：vbench 系列的 physics consistency 评测下，最好的视频世界模型在 PhyWorldBench 上的成功率才 0.262。

形式化地，L2 把 L1 的局部算子组合起来表示为：

$$\hat{p}(\tau \mid z_0, a_{1:H}, c) \propto \prod_{t=1}^{H} p_\theta(z_t \mid z_{t-1}, a_t)\, \phi_c(\tau)$$

其中 $\phi_c(\tau)$ 是一个**作用在整条轨迹上**的支配律兼容性项——这就是 L2 区别于 L1 的关键：L1 的 trajectory distribution 可以分解成独立的逐步项之积，L2 因为 $\phi_c$ 跟整条轨迹相关而**不可分解**。这个区别看起来抽象，但它直接决定了你需要的训练 loss 和评测协议都不一样。

哲学比喻：作者用 David Lewis 的"最近可能世界"理论——一个有效的 counterfactual 推理系统，能 explore 与现实最相似的、只有最小干预差异的多个世界。但 L2 的可靠性始终依赖于**学到的转移结构本身**，所以它有 epistemic drift 的风险——产生在训练 manifold 内部自洽、但脱离真实环境的轨迹。柏拉图洞穴比喻：再厉害的影子预测器，也仍然被墙的尺寸所限制。

### L3 Evolver：自己发现自己错了，并且改自己

这是这篇论文真正想立的 flag。L3 的形式化定义是这样一个闭环：

$$\mathcal{M}_t \xrightarrow{\text{design}} a_t \xrightarrow{\text{execute}} o_t \xrightarrow{\text{observe}} d_t \xrightarrow{\text{reflect}} \mathcal{M}_{t+1}$$

四步：design 设计实验 → execute 执行 → observe 观察结果 → reflect 反思并更新模型栈。

但作者明确说：**有这个 loop 还不算 L3**。要满足三个边界条件：

1. **Evidence-grounded diagnosis**：失败必须能归因到具体可操作的原因，并且基于可回放的证据
2. **Persistent asset update**：修复要变成可复用的 asset（skill、rule、parser、test），不能只是 in-context 的临时补丁
3. **Governed validation**：更新要通过 regression 和 robustness 门，包括 rollback 和 canary 策略，才能默认启用

这三条很关键——它把 L3 从"agent 反思"这种宽泛的概念，**收紧成必须有持久化资产更新和回归门控的工程系统**。SWE-agent 那种每次 issue 单独解决的，没有累积资产，不算完整 L3；CodeIt 那种把搜索轨迹蒸馏回 LLM 自身参数的，才算闭环。

哲学比喻这次用的是 Lakatos 的"硬核 vs 保护带"：参数微调相当于在调保护带（protective belt），而真正的 L3 升级可能要动硬核（hard core）——架构、归纳偏置、约束模块的改写。Duhem-Quine holism 则解释了为什么 L3 的"诊断"这一步天然困难：错误会在多个模块之间分摊，不靠系统化的 diagnostic 根本定位不到根因。

![图2：从局部预测到证据驱动修订的层级视图。L1 建模经验规律做预测，L2 支持可能世界语义和反事实仿真，L3 引入证据驱动的修订机制](https://arxiv.org/html/2604.22748v1/x4.png)

*图2：三级能力的层级化视图——L1 是模式识别，L2 是时间推演，L3 是基于现实交互的自适应模型演化。这个图把哲学层面的进展和工程层面的能力直接对应起来了。*

---

## 四类世界规律：为什么物理和科学要分开

四个 governing-law regime 是横轴。乍看会觉得"物理世界"和"科学世界"差不多，但作者做了一个我觉得非常关键的区分：**两者的差别在于约束如何被访问**。

- **物理世界**：通常可以用解析方程或物理引擎做验证。一个杯子穿桌、刚体互相穿透、能量不守恒，物理引擎当场就能给你判违例。
- **科学世界**：支配机制只是部分已知，往往要通过实验来 empirically 验证。蛋白质折叠、气候动力学、化学反应，你没法事先写出闭式解。

这个区分在工程上意义重大。物理域的 L2 可以靠 MuJoCo、Brax、Isaac Gym 这种 simulator 当 ground truth 来对齐；科学域的 L2 必须做主动实验、跟现实数据对齐——这就是为什么 GraphCast、NeuralGCM、AlphaFold 跟 DreamerV3 不能放在一个评测框架下来比。

社会世界则有完全不同的特性：**reflexivity（反身性）和 normativity（规范性）**。Agent 关于社会状态的信念会改变社会状态本身（你以为通胀要起来了，于是大家囤货，于是真起来了）；而且转移不仅由"会发生什么"主导，还由"应该发生什么"的共同规范主导。这就是为什么 LLM 当 social agent 的时候 role drift 和 goal forgetting 这类失败几乎是必然的——它根本没建模共同 commitment。

数字世界的特殊性最容易被忽略：**程序语义是确定性、可机械验证的**。一个动作要么满足 API contract 要么不满足，error code、permission denial、UI state machine 全部 loggable、replayable。这就是为什么数字域的 L3（regression-gated software agent）走得相对最快——验证基础设施天然存在。

![图3：四类规律范畴的示意图。从左到右分别是物理世界（人形机器人操作积木）、数字世界（代码与UI界面）、社会世界（带语言行为的多Agent网络）、科学世界（带机器人显微镜与移液器的科学实验）](https://arxiv.org/html/2604.22748v1/x2.png)

*图3：四类规律的代表性场景。每类规律的形式化约束本质不同——物理世界靠分析或仿真验证，数字世界靠程序执行验证，社会世界靠 commitment/norm consistency 验证，科学世界靠实验测量验证。*

---

## 历史脉络：从牛顿定律到 AlphaEvolve

作者画了一张时间线（图4），把"世界建模"这个动作的历史从 1687 年的牛顿《原理》一直拉到 2025 年的 AlphaEvolve。这条线索其实回答了一个问题：**人类一直在尝试建立世界的内部模型，AI 只是最新的载体**。

四个时代的划分：

- **数学原理时代（–1956）**：牛顿、拉普拉斯、图灵——说到底都是在构建可预测的世界模型，只是工具是数学方程
- **符号智能时代（1956–1986）**：STRIPS、CYC，尝试手写规则。Frame Problem 揭示了根本困难——每个动作都要显式说明什么不变。两次 AI winter 把这个路径基本判了死刑。
- **连接主义复兴（1986–2020）**：从反向传播到 Transformer，从手写规则转向学习表征，World Models 论文（Ha & Schmidhuber 2018）和 Dreamer 系列把这条路推到极致。
- **生成革命（2020–至今）**：DDPM、GPT-3 引爆，Sora、AlphaEvolve 这些系统模糊了 prediction 和 simulation 的边界。

![图4：世界建模代表性系统的时间线（2018-2026），按能力级别组织。L1 Predictor 是一步动力学，L2 Simulator 是可决策多步 rollout，L3 Evolver 是完整的证据驱动模型修订](https://arxiv.org/html/2604.22748v1/x3.png)

*图4：70 个代表性系统按"年份 × L 级别"网格放置，每个 pill 用颜色标注 governing-law regime（物理蓝、数字绿、社会橙、科学紫）。可以看到 L3 系统在科学和数字两个 regime 出现得明显更早、更密。*

作者的判断我比较认同：**进步从来不是单纯靠规模——而是靠改变"被表示的对象"、"在长程上可组合的对象"、"可以被证据修订的对象"**。这话说得不漂亮但很硬。

---

## L1 全景：表征 / 动力学 / 解码 / 反向动力学

L1 这一节我其实快速扫过的，因为大家比较熟。几个值得提的点：

**表征学习的两条路**：一条是对比学习（CPC、SimCLR、MoCo、CURL、SPR），另一条是 mask 预测（I-JEPA、V-JEPA、DINOv2）。后者是 LeCun 推的方向，思路是不要在 pixel 空间重建，而是在 latent 空间预测 masked region 的 embedding。

**前向动力学的范式迁移**：从 PILCO 的 Gaussian Process，到 RSSM 的 GRU+stochastic，再到 IRIS 的 VQ-VAE+Transformer、TransDreamer 的 Transformer-XL、DIAMOND 的 Diffusion——一条很清晰的"从连续到离散 token、从 RNN 到 Transformer/Diffusion"的演化。TD-MPC2 一个 317M 参数的 agent 能 master 104 个跨域任务，这个数字现在看也还是蛮 impressive 的。

**Inverse Dynamics 的妙用**：这个算子最被低估。Pathak et al. (2017) 用它当 curiosity-driven exploration——通过预测两个相邻状态之间的动作，编码器自动过滤掉不被 action 影响的视觉噪声（飘动的云、闪烁的背景）。VPT（Baker et al. 2022）更猛：用一小批有动作标注的 Minecraft 视频训了 inverse dynamics model，再去给海量未标注的 internet 视频回填动作标签，最终学会了"挖钻石"这种长程行为——只用被动观看。这个 trick 我觉得比纯 L1 framework 更值得做工程的人记住。

---

## L2 全景：四类 regime 的 simulator 各自长什么样

L2 这一节是论文体量最大的部分，也是真正的 cross-domain synthesis。作者把 L2 boundary condition 在四个 regime 下的具体形态做成了一张表（Table 4）：

| 边界条件 | 物理世界 | 数字世界 | 社会世界 | 科学世界 |
|---------|---------|---------|---------|---------|
| **Coherence** | H步操作中物体持久性、接触稳定 | 多步 UI/code 交互中 DOM/文件系统一致 | 多轮对话中 commitment 与关系稳定 | 实验序列中因果链有效性 |
| **Sensitivity** | 力/位置扰动按比例改变抓取结果 | UI 故障注入引发合适的 replan | 改变一个 agent 策略带动谈判结果变化 | 参数变化产生方向正确的测量变化 |
| **Consistency** | 无相互穿透、能量守恒、运动学可行 | API 契约、类型约束、状态机有效 | 规范遵循、信念一致、社会反身性 | 守恒律、因果图一致、证据链有效 |

这个表的价值在于：**它给四个领域的工程师一个共同语言**。你做物理 simulator 评测时关心的"无穿透"，跟我做 web agent 评测时关心的"DOM 一致"，其实都是 constraint consistency 这个东西的不同实例化。

### 物理域：视觉看起来真 ≠ planner 能用

这一节作者很犀利。Sora、Lumiere、VideoPoet 这些 visual-first 的 video generation 系统，**视觉拟真和决策可用是两回事**。VBench 系列、PhyWorldBench 这些专门测物理一致性的 benchmark 揭示了：最强的视频模型在物理一致性上的成功率才 0.262。

更可怕的是，FVD、SSIM 这种传统的生成质量指标根本测不出来这个问题——distributional realism 高得吓人的同时，重力反向、物体穿透、对象消失这种 planner-killing 的 failure 完全测不到。

作者由此引出一个我特别认同的观点：**a good Simulator does not have to look more like the world; it must look more like the constraints**。一个好的仿真器不需要看起来更像世界，它要看起来更像约束。物理用几何/接触约束、软件用状态机、社会用角色/规范一致性、科学用证据链可证伪性。**让约束变得显式（loggable、replayable、regressable），通常比提升感知保真度更能改善长程稳定性。**

### 数字域：世界模型可以是一段可执行的程序

这块最让我兴奋的是 CodeWM 和 WorldCoder 这类工作。它们的核心 insight 是：**世界模型不一定要是个 neural network，可以就是一段可执行的 Python 程序**。

CodeWM (Dainese et al. 2024) 用 LLM + MCTS 生成 Python 程序作为 18 个环境的 explicit、interpretable world model。WorldCoder 让 LLM agent 通过环境交互**增量地构建** Python 世界模型。CWM (Copet et al. 2025) 一个 32B 的 open-weights LLM 专门为 code world model 训练，在 SWE-bench Verified 上拿到 65.8%。

更进一步的是 Web World Models (Feng et al. 2025)：把世界状态实现成 ordinary 的 web code（TypeScript modules、HTTP handlers、database schemas），逻辑一致性交给 web stack 的确定性执行，LLM 只负责生成 context 和高层决策。这个思路很反直觉——你以为越大的 LLM 越能"端到端建模一切"，结果实际工程里"把不该让 LLM 做的让真程序做"反而更可靠。

### 社会域：1 万 Agent 的城市仿真，还差很远

社会域这块作者比较坦诚——LLM agent 在 Theory of Mind 上的表现还非常弱。FANToM (Kim et al. 2023) 揭示了所有 SOTA LLM 都有"illusory ToM"现象，ExploreToM 测试中 GPT-4o 的准确率低到 9%。

但应用层面的工作进展确实很快。Generative Agents (Park et al. 2023) 的 25 agent 小镇是开端；Project Sid (AL et al. 2024) 上到 1000 agent 还出现了 emergent specialization；OASIS (Yang et al. 2024) 直接拉到 100 万 agent，重现了信息传播和群体极化现象。但这些系统的"社会动力学"基本是涌现出来的，不是被精确建模的——你没法用它做精细的政策推演。

### 科学域：从 GraphCast 到 AlphaFold

科学域的 L2 系统是最"成熟"的一类——GraphCast、Pangu-Weather 在 90% 以上的 ECMWF 预报指标上超越传统数值天气预报；GenCast 用 diffusion 把概率预报推到 97.2% 的指标领先；NeuralGCM 把可学习参数化嵌入到可微 GCM 内部，能涌现出热带气旋。AlphaFold 系列、Aurora（Earth system foundation model）、神经网络势函数（取代 DFT 做分子动力学，速度提升数量级）。

这些工作之所以成熟，**根本原因是科学域有相对干净的 ground truth——观测数据**。你的天气预报准不准，对照实测温度、风速、气压就能算清楚。物理 RL 那种"sim 里好看，real 里崩"的问题在这边相对小一些。

![图5：四类规律范畴的诊断图。横轴反映规律的可形式化和可机械验证程度，纵轴反映状态和约束的可观测性](https://arxiv.org/html/2604.22748v1/x5.png)

*图5：四个 regime 的诊断坐标——横轴是规律的"可形式化与可验证度"（数字世界最高、社会世界最低），纵轴是状态与约束的"可观测度"（物理较高、社会较低）。这个图解释了为什么 L3 在不同 regime 的成熟度排序差别如此之大。*

---

## L3：自我进化的世界模型，目前真的有吗？

L3 是论文力推的概念，也是实践上最稀缺的能力级别。作者按 regime 给了一个清晰的成熟度判断：

- **科学域（已成熟）**：CAMEO（同步辐射光束线的闭环材料发现）、A-Lab（17 天做了 353 个实验，从 57 个目标合成出 36 个化合物）、BacterAI（零先验的微生物代谢图谱探索）、Robot Scientist Adam（最早的自主基因功能研究）。这些系统反复证明了：**只要 instrument 和 feedback 设计得好，闭环修订是可工程化的**。
- **数字域（部分成熟）**：FunSearch 在 cap set 问题上找到了新的构造、AlphaEvolve 在 Strassen 矩阵乘 56 年之后给出了改进、CodeIt 把搜索轨迹蒸馏回 LLM 参数。但很多系统缺**主动信息扩展**这个边界条件——它们有 design 和 observe，但没有真正的 active probing。
- **物理域（emerging）**：AdaptSim 通过少量真实 task data meta-learn 适应策略；Hu et al. (2025) 的视觉 self-model 检测形态变化并重训练。但归因困难是核心瓶颈——感知错了？动力学错了？actuation 错了？环境变了？没法定位就没法修。
- **社会域（aspirational）**：基本还在 paper stage。社会实验有伦理约束，归因本身模糊，行为 ground truth 噪声大。Kumar et al. (2026) 用 LLM 驱动的遗传编程演化 governance rule，比人工设计好 123%——这是少数能算上 social L3 的工作。

![图6：L3 进化循环。一个完整周期经历四个阶段：design、execute、observe、reflect，产生修订后的世界模型栈](https://arxiv.org/html/2604.22748v1/x6.png)

*图6：L3 的四阶段闭环——design 设计干预、execute 执行实验、observe 观察结果、reflect 蒸馏证据并更新模型。这个图本身不复杂，复杂的是工程化每一步：怎么设计高 information gain 的实验？怎么 attribute 失败到具体模块？怎么验证更新没有 regress？*

![图7：L3 在四类规律范畴的演化示例。(a) 物理智能：自适应探测修正接触动力学；(b) 社会智能：规范漂移触发社会模型修订；(c) 数字智能：评估器驱动的程序搜索 + 回归门控；(d) 科学智能：同步辐射光束线的闭环自主实验](https://arxiv.org/html/2604.22748v1/x7.png)

*图7：四个 regime 的 L3 实例化各有自己的味道。物理域靠对接触模型的微扰探测，社会域靠规范漂移检测，数字域靠 LLM 程序搜索 + 自动评估器，科学域则真的有机器人在自动跑实验。这张图解释了为什么 L3 不是一个单一技术，而是 regime-specific 的工程模式。*

我个人觉得 L3 最有意思的判断是这个：**L3 系统的瓶颈通常不是生成候选修复，而是安全地验证它们**。你让 LLM 生成 100 个 candidate fix 不难，难的是哪些可以 default-enable，哪些要 canary、哪些要 rollback、哪些要触发 human review。这个判断也跟我自己的工程经验对得上——SWE-agent 这类系统真正的工程难点不在 agent 本身，在于 patch 能不能过 regression suite。

### Evidence 的可证伪性是关键

L3 的另一个核心论点：**进化质量取决于证据质量**。作者给了一张表（Table 9）按 regime 列出了"什么样的信号能触发 L3 模型修订"，并按 within-domain 的 falsifiability 上色。

- **数字域**（高可证伪）：regression detection、execution outcome mismatch、task completion failure——这些都是 mechanically checkable 的硬证据
- **物理域**（中等可证伪）：kinematic infeasibility、contact dynamics mismatch、morphology change——力/扭矩传感器可以提供量化证据
- **科学域**（中-高可证伪）：hypothesis falsification、prediction-measurement gap、epistemic gap detection——靠实验测量
- **社会域**（低可证伪）：interventional inconsistency、global behavioral drift、individual faithfulness violation——基本要靠间接信号，归因极其困难

作者特别提了一句我觉得很值得抄下来的话："human feedback should not be treated as a single falsifiability class"——人类反馈不是同一个可证伪性等级。**主观偏好反馈是弱可证伪的，专家诊断反馈在能被后续 test/experiment 验证时才是强可证伪的**。这个区分对所有做 RLHF / DPO 的人都有用：你拿来训 reward model 的偏好数据，跟拿来 attribute failure 的诊断数据，根本不是一类东西。

---

## 实验和评测：为什么大家其实都在 L1 上比赛

这一节作者做的事情很犀利：把现在大家在做的 benchmark 拿过来看，**到底测的是 L 几**？

结论让人尴尬：**绝大多数主流 benchmark 测的都是 L1**。Atari 100k、Meta-World、CALVIN、OSWorld、SWE-bench、WebArena、Sotopia——这些 benchmark 在协议层面其实都是单步预测准确率或者端到端 success rate（没有 perturbation injection）。

L2-style 的评测（counterfactual injection、degradation curve、constraint violation detection）原则上是有的，少数 benchmark 有，但**远远不是 field standard practice**。L3-style 的评测基础设施（regression suite、asset validation gate、cross-episode improvement tracking）**除了 autonomous science，基本不存在**。

作者于是提出两个 decision-centric 的 aggregate metric：

**Action Success Rate** ASR：

$$\mathrm{ASR}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\text{task}_i \text{ succeeds under policy derived from } \hat{p}]$$

**Counterfactual Outcome Deviation** COD：

$$\mathrm{COD}(k)=\mathbb{E}[d(\hat{z}^{(1)}_H, \hat{z}^{(2)}_H)]$$

ASR 测"模型支撑的决策好不好"，COD 测"模型对 action 干预敏不敏感"。COD 低意味着模型不响应 action 改变，对 counterfactual planning 完全没用。

我想说的是：这两个指标其实不算多新颖，但它们值得写到工程指南里——**如果你只看 success rate，你测的是 agent，不是 world model 本身**。要真正评测一个世界模型，必须看 counterfactual 行为。

论文还提了一个 Minimal Reproducible Evaluation Package (MREP) 提案：版本锁定、trace logging、failure taxonomy、tail statistics、boundary condition mapping。这是一个比较 actionable 的工程标准建议——对做 benchmark/leaderboard 的人来说，这是个值得参考的 checklist。

---

## 架构落地：representation × dynamics × control

这块作者把世界模型的设计空间拆成三个轴（Table 11）：

**表征轴**：
- 符号/程序状态（VirtualHome）：可解释、能硬约束，但工程量大、状态空间窄
- 潜在连续表征（DreamerV3、V-JEPA2）：吸收高维多模态输入，但长程会语义漂移
- 结构化 3D（RoboOccWorld、PointWorld）：天然契合物理约束，但重建瓶颈
- 离散 token（IRIS VQ-VAE）：组合性强、能用 cross-entropy 精确训练，但 codebook collapse 风险

**动力学轴**：
- Stochastic latent（DreamerV3）：原则性的不确定性建模，但长程 miscalibration
- Deterministic value-aware（MuZero、TD-MPC2）：跟 control 紧密整合，但缺乏显式不确定性
- Autoregressive token（iVideoGPT、LWM）：统一多模态接口，但长程逻辑一致性弱
- Diffusion（Sora、DIAMOND、Genie）：逼真观测，但 multi-step denoising 的延迟和 action controllability 都是问题

**控制接口轴**：
- Online MPC（TD-MPC2、PETS）：每步重新规划，反应快但 compute 重
- Tree search（MuZero、EfficientZero）：可分支的反事实探索，但放大模型误差、容易被 benchmark exploit
- Imagined-rollout policy（Dreamer 系列）：在模型内训练，不需真实交互，但要求动力学非常准
- Offline distillation（GR-1）：部署便宜，但分布漂移脆弱
- Replayable environment（OSWorld、SWE-agent）：把真实环境当 simulator，绕开学到的动力学

不同 regime 对这三个轴的组合有强偏好。物理域基本是 latent/3D + MPC/imagined-rollout；数字域是 symbolic/DOM + replayable environment；社会域受困于 commitment graph 的可扩展性；科学域则普遍是 Bayesian surrogate + 主动学习。

工程上有三个 cross-cutting 原则我觉得值得反复提：

1. **Separate what is learned from what is enforced**：硬约束层（碰撞检测、状态机校验、回归门）应该在 inference 时强制执行，不要靠 training loss 软约束——soft enforcement 没法保证零违例 rollout
2. **Instrument before you iterate**：日志、回放、failure attribution 的基础设施要在系统设计早期就做进去，不然 L3 修订就是 anecdotal 的、ungovernable 的
3. **Match the representation to the planner's query**：表征看起来真不重要，重要的是它有没有暴露 planner 需要的变量——可达自由空间、permission state、反应速率。一个高保真但不暴露这些变量的表征，反而比低保真但暴露了的差

第三点尤其重要——它直接呼应前面那句"a good Simulator does not have to look more like the world; it must look more like the constraints"。

---

## 我的判断：这篇 survey 真正的价值在哪

读完这篇 100+ 页的 survey，我的几个判断：

**它最大的价值不是新算法，是新词汇表**。L1/L2/L3 + 四类 regime 这套坐标系其实没什么 fundamentally 颠覆性的内容——大部分概念在各自社区都已经存在。但这篇 survey 把它们组织成一个**可以跨社区交流的统一语言**，这个本身就值得做。前两年做 World Model 不同方向的人聊起来基本是鸡同鸭讲，有了这套坐标系之后至少能问清楚"你在 L 几？哪个 regime？"

**L3 的 framing 是这篇文章最有 ambition 的部分**。把"自主修订模型"提升为一个独立能力级别，并且给出三个 testable 边界条件（evidence-grounded diagnosis、persistent asset update、governed validation），这个动作很有价值。它逼着工程师区分"agent 能反思" vs "agent 能持久化地自我改进"——这两件事之前经常被混在一起。

**深刻的洞见之一：约束比保真度重要**。物理域的视频世界模型那一节我反复看了几遍——视觉拟真度上去了不代表 planner 能用。这个判断不算新，但作者把它形式化成了 L2 的三个 boundary condition，并且明确提出"a good Simulator does not have to look more like the world; it must look more like the constraints"，对当下整个生成式 AI 圈是一个不小的提醒。

**一些值得吐槽的地方**：

- **它就是一篇 position-driven survey**，作者自己也明说了。L1/L2/L3 的边界条件很多时候在实践里很难严格判定——一个系统部分满足、另一部分不满足的情况比比皆是。这种分级在学术叙事里很有用，但工程落地时你会发现很多系统是"L1.5"或"L2.3"。
- **L3 的工程细节其实没展开太多**。CAMEO、A-Lab 这类系统作为 L3 example 被反复提到，但真正怎么搭这种系统、怎么设计 regression gate、怎么设计 canary deployment，作者基本一笔带过。这块可能需要后续专门的 engineering paper 来补。
- **跨 regime 转移的实操性比较弱**。survey 反复说 cross-domain synthesis 是它的核心价值，但具体怎么把数字域的 regression-gated loop 搬到物理域，作者没有给出可操作的迁移路径。
- **404 个引用里很多是 2025/2026 年的论文**，部分引用的工作我读这篇 survey 的时候还没法验证。survey 的"完整性"在快速迭代领域永远是一个相对概念。

**对正在做这块的工程师，我的具体建议**：

1. 如果你做 Web/GUI agent 或 SWE agent：你目前的系统大概率在 L1.5 到 L2 之间。值得思考的是怎么往 L3 推——具体来说，**怎么把 agent 解决过的问题蒸馏成 reusable asset**（reproduction script、regression test、reusable installation procedure）。
2. 如果你做 robot policy/manipulation：注意"视觉真" vs "可决策"的差距。FVD/SSIM 这些指标对你来说基本是噪声，要切换到 ASR/COD 这种 decision-centric 的评测。
3. 如果你做生成式视频/世界模拟：物理一致性是一个真问题，光看 VBench-2.0 上的 perceptual 分数会被骗。把约束验证（geometry check、object permanence、conservation laws）当 first-class metric。
4. 如果你做 AI for Science：你已经在 L3 frontier 上了，但要小心 surrogate-to-reality gap——光在 simulation 数据上验证 surrogate 是不够的，必须 budget 一部分真实测量来做 calibration。

最后一个我自己持续在想的问题——**meta-world modeling 这个概念**（Section 8.3）。L3 是在已知规律内修订模型，meta-world modeling 则是修订规律本身。比如生物演化、社会规范变迁、气候系统，governing law 本身就在漂移。这种"规律的规律"如何建模？符号表征会不会重新变得重要？作者也承认这块是 open question。但我隐约觉得，这可能是 next big thing——尤其是在 AI for Science 这边。

如果你在做 Agent、做 RL、做生成式仿真、做科学发现，我推荐你花时间精读这篇综述里跟你 regime 相关的两个 section（L2 应用、L3 案例），加上 Section 7 的架构指南。其他部分可以扫读，参考文献当 reading list 用。

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*

