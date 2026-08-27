# 调 Agent 不如调"鞍具"：AutoSaddler 把 Harness 优化变成了 mini-batch 训练

你有没有这种体验：一个 Agent 系统跑起来之后，性能上不去，你盯着它看半天，发现瓶颈根本不在模型——而是系统提示词写得不对、某个工具的参数设计很蠢、agent loop 里少了一个提醒。于是你开始手工调 prompt、改工具描述、加 hook。改一版，跑一遍，看轨迹，再改。一个下午过去了，涨了 2 个点，你也不知道这 2 个点是泛化的改进还是过拟合到你刚看的那几条失败轨迹上。

这个"模型之外的一切"——prompt、工具配置、运行时控制逻辑——最近有个流行的叫法：**Harness**（鞍具，马要跑得快，鞍具得合身）。手工调 Harness 的痛苦，做 Agent 的人都懂。微软和 KAIST 等机构合作的这篇 AutoSaddler（arXiv:2608.23041），干的事情一句话概括：**把 Harness 优化形式化成一个离线 mini-batch 学习问题，让 Agent 自己读失败轨迹、诊断根因、给 Harness 打补丁，而且用 dev 集验证保证补丁是"耐用的"而不是过拟合的**。

## 核心摘要

长时程 Agent 的可靠性问题，很大程度上不是模型问题而是 Harness 问题，但 Harness 设计空间巨大（prompt × 工具 × 控制逻辑），手工搜索又贵又慢。AutoSaddler 的思路是把整个优化过程组织成机器学习里再熟悉不过的 mini-batch 训练循环：在小批次任务上评估当前 Harness → 深度诊断失败轨迹并生成结构化补丁 → 同批验证 → dev 集泛化检查 → 反思提炼教训存入 EvoDAG → 演化出下一代 Harness。在 GAIA2、SWE-Bench Pro、Terminal-Bench 2.0 三个基准上，相对各自的基础 Harness 分别提升 **9.0、9.6、10.0 个百分点**，相对最强自动基线提升 7.4、4.4、6.7 个点，而且达到最优性能只消耗了约 147 条轨迹——是对比方法 Meta-Harness 的十分之一左右。我的判断：这不是一个"新算法"式的突破，而是把 LLM 系统优化这件事的工程纪律做对了——诊断要深、干预要有结构、选择要看泛化。这三条听起来朴素，消融实验告诉你每一条都值好几个点。

---

## 📖 论文信息

- **标题**：AutoSaddler: Automatic Harness Optimization with Durable Updates from Agent Execution Traces
- **作者**：Sungho Park, Wonjoong Kim, Rongyuan Tan, Jue Zhang, Wook-Shin Han, Pengfei Gao, Chanyoung Park, Yongqiang Yao, Rao Fu, Elsie Nallipogu, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang
- **机构**：KAIST、Microsoft、POSTECH 等
- **链接**：https://arxiv.org/abs/2608.23041 （2026 年 8 月 24 日提交，44 页，15 张图）
- **项目主页**：https://aka.ms/AutoSaddler-website

---

## 🎯 问题动机：模型之外的搜索空间，比模型本身还难调

先厘清 Harness 到底是什么。论文自己承认这个词在社区里用法很口语化——"LLM 之外所有让 Agent 行为变好的东西"。AutoSaddler 采用了一个具体的形式化（跟随 LangChain 的 harness engineering 实践），把优化空间写成三元组：

$$\theta = (\theta_{\text{prompt}}, \theta_{\text{tool}}, \theta_{\text{middleware}}) \in \Theta$$

$\theta_{\text{prompt}}$ 是系统提示词和指令，$\theta_{\text{tool}}$ 是可用工具及其接口，$\theta_{\text{middleware}}$ 是运行时控制逻辑（hooks、agent loop 行为）。注意它刻意不管 memory 和 skill 管理——因为设定里任务是无状态、相互独立的。

为什么这事值得自动化？三个原因叠加起来很致命：

1. **搜索空间大**。prompt 措辞、工具参数、hook 逻辑，组合起来是个巨大的离散空间；
2. **评估贵**。一个候选 Harness 好不好，得让 Agent 真的跑几十步任务才知道；
3. **失败分析更难**。长时程轨迹动辄上百步，人工读轨迹找根因，基本是纯手工考古。

这就是经典的"贵且不可微"优化问题。模型权重可以反向传播，Harness 不能。但作者意识到一件事：你想想看，mini-batch SGD 的结构其实不依赖梯度——它依赖的是"小批量评估 → 误差信号 → 参数更新 → 验证"这个循环。误差信号可以是数值梯度，**也可以是文本形式的失败诊断**。这就是整篇论文的题眼。

---

## 🏗️ 方法：把训练循环原样搬过来

### 总体框架

优化目标写得很标准：

$$J(\theta)=\mathbb{E}_{(x,y^{\ast})\sim\mathcal{T}}\mathbb{E}_{(\tau,\hat{y})\sim P_{\theta}(\cdot\mid x)}[\mu(\hat{y},y^{\ast})],\qquad \theta^{\ast}=\arg\max_{\theta\in\Theta}J(\theta)$$

任务集切成 train/dev/test 三份，rollout 预算 $K$ 耗尽后，返回 dev 集经验得分最高的候选 Harness $\hat{\theta}_{\mathrm{AS}}$，然后在 test 集上只评一次——**不拿测试反馈回头改**。这个协议本身就很"机器学习"，值得点赞，后面细说。

![AutoSaddler 的迭代优化循环总览](https://arxiv.org/html/2608.23041v1/main_figure_revised.png)

*图 2（论文 Figure 2）：AutoSaddler 的完整循环。① 当前 Harness $H_n$ 在 mini-batch 上测试；② Diagnosis-Patch Agent 跨 Prompt/Tool/Middleware 三个组件诊断并打补丁（分探索期和精修期两个阶段）；③ 同一 mini-batch 上验证补丁是否有效；④ 若有改进则在 dev 集上进一步评估泛化性；⑤ Reflection Agent 对比补丁前后轨迹、提炼教训；⑥ 教训写入 EvoDAG（Harness 演化历史的有向无环图）；⑦ Evolution Agent 基于 EvoDAG 合成下一代 Harness $H_{n+1}$。*

每一轮迭代走七步。其中最关键的设计映射是：**Diagnosis–Patch–Verification 扮演了文本梯度下反向传播的角色**——文本误差信号不像数值梯度那样能自动校验对错，所以每次更新都要显式地做假设生成（根因是什么）、干预（打补丁）、经验验证（同批重跑）。**EvoDAG + Evolution Session 则扮演优化器的角色**——它不只是存结果，还存"教训"这个优化器状态，并且支持从历史任意祖先 Harness 重组元素，类似进化搜索里的 recombination，用来跳出局部最优。

这个类比不是牵强附会。附录里有一张表把传统 mini-batch 训练的七步和 AutoSaddler 循环逐步对应：前向传播 ↔ 执行当前 Harness（rollout 就是前向计算）；计算损失 ↔ 记录结果和轨迹（任务结果给稀疏信号，轨迹给解释性证据）；应用优化器更新 ↔ 反思 + 写 EvoDAG + 演化。论文甚至把更新规则拆成两个式子：

$$\text{diagnosis} \longrightarrow \text{patch-as-intervention} \longrightarrow \text{same-batch verification}$$

$$\text{dev-set validation} \longrightarrow \text{reflection} \longrightarrow \text{EvoDAG update} \longrightarrow \text{evolution of next harness}$$

第一式负责构造"可信的文本梯度"，第二式负责"提交更新"。分工很干净。

### 三个核心设计：每条都被消融验证过

**① 深度诊断，不是浅层反思。** 大多数自动 prompt 优化（比如后面会谈到的 GEPA）的失败分析是"单次 LLM 调用读一下轨迹，推断失败原因"。AutoSaddler 的诊断是一个完整的 Agent 会话：诊断和补丁生成不分离，Agent 手里拿着失败轨迹**和整个 Harness 代码库**，可以像调试器一样逐步检索轨迹细节、查文件、验证假设，最后才提补丁。数据说话：合并的 diagnosis-patch 会话比"只打补丁"平均多做 6.2 次工具调用、5.8 次文件访问。多花的这些功夫换来了什么，消融实验里看。

**② 结构化干预，不是自由编辑。** 补丁只能落在三类组件上，而且只暴露实现 Harness 功能逻辑的源文件，评估代码和基准数据碰都不让碰。补丁分类法如下：

| 类别 | 子类型 | 类型 | 干什么 |
|---|---|---|---|
| Prompt Patch | Prompt Rule Addition | Steering | 系统提示词里新增行为规则 |
| Prompt Patch | Prompt Rule Modification | Steering | 修订现有规则，解决冲突 |
| Tool Patch | New Tool Addition | Capability | 现有工具干不了就加新工具 |
| Tool Patch | Argument Modification | Capability | 改工具参数增强过滤/选择能力 |
| Tool Patch | Implementation Fix | Capability | 修工具内部 bug 或扩展功能 |
| Tool Patch | Tool Description Fix | Steering | 改 docstring 防止误用 |
| Middleware Patch | PreToolUse Hook | Steering | 特定工具调用前注入即时提醒 |
| Middleware Patch | Infrastructure Change | Capability | 改 agent 配置、迭代预算、环境设置 |
| Middleware Patch | Agent Loop Logic Change | Capability | agent 循环里加预处理或预算提醒 |

两个大组的划分很讲究：**Capability Patch（C）** 改可执行代码或编排逻辑，**Steering Patch（S）** 只做文本编辑。还有一个类似"学习率调度"的设计——**Phased Patch Scheduling**：先跑 Capability 阶段（大刀阔斧改结构），再切到 Steering 阶段（精调文本）。直觉上很合理：你不会在架构还没定型的时候去抠 prompt 措辞。

**③ 泛化感知选择，不是轨迹特异的"热修复"。** 这是论文标题里 "Durable Updates" 的落点。一个补丁在同批验证通过后，还要去 dev 集上评估；Reflection Session 把每个案例分成 fixed / regressed / still-failing / still-passing 四类，带着针对性的反思问题（为什么修好？为什么回归？）提炼教训写进 EvoDAG。只有经得起泛化检验的更新才会沉淀下来。

---

## 🧪 实验：三个基准，全面压制

### 设置

三个基准配三个不同的基础 Harness，覆盖面不错：

| 基准 | 任务类型 | 基础 Harness |
|---|---|---|
| GAIA2 | 模拟手机环境的通用助手，10 个 Universe | GAIA2 默认 ReAct Agent |
| SWE-Bench Pro | 企业级软件工程任务 | SWE-agent |
| Terminal-Bench 2.0 | 89 个真实终端任务（系统管理、ML、安全） | Terminus 2 |

底层模型统一用 Claude Opus 4.6。三个优化器 Agent（诊断-补丁、反思、演化）都基于 Claude Agent SDK 实现。基线有两个：**GEPA**（Agrawal et al. 2025，反思式 prompt 进化，ICLR 2026 oral，以 prompt 为中心）和 **Meta-Harness**（把优化对象从 prompt 扩展到整个 harness 的端到端方案）。

数据划分上特意做了**跨分布泛化**设计：train/dev/test 来自不同任务组。比如 SWE-Bench Pro 训练集是 qutebrowser 仓库，dev 集是 Vuls 和 NodeBB，测试集是 Ansible、Flipt、Element-web——优化时压根没见过测试仓库。每种方法在 train+dev 上跑一次演化，test 集跑三遍取均值，指标是 Pass@1。

### 主结果

GAIA2 测试集（三遍均值 ± 标准差，括号内为任务数）：

| Harness | U21 (107) | U22 (112) | U27 (81) | 平均 |
|---|---|---|---|---|
| Default Agent（手工） | 54.8±4.8 | 51.5±4.9 | 52.7±4.3 | 53.0±1.5 |
| GEPA（自动） | 60.1±3.9 | 47.9±3.4 | 56.4±0.7 | 54.6±2.5 |
| Meta-Harness（自动） | 53.0±1.1 | 51.5±5.2 | 56.0±0.7 | 53.2±2.2 |
| **AutoSaddler（自动）** | **61.4** (2.4) | **60.7** (2.4) | **64.6** (3.1) | **62.0** (1.2) |
| w/o 深度诊断 | 56.7±4.2 | 57.1±3.9 | 60.1±5.0 | 57.8±3.8 |
| w/o 结构化干预 | 58.9±2.5 | 53.3±7.6 | 59.3±3.3 | 56.9±3.8 |
| w/o 泛化感知选择 | 53.3±1.9 | 44.9±6.7 | 54.7±4.0 | 50.6±4.0 |

另外两个基准：

| Harness | SBP Ansible (96) | SBP Flipt (85) | SBP Element-web (56) | SBP 平均 | TB2 (40) |
|---|---|---|---|---|---|
| 基础手工 Harness | 40.6±1.9 | 31.0±3.5 | 41.1±1.7 | 37.3±4.8 | 40.0±0.0 |
| Terminus KIRA（手工专家调优） | — | — | — | — | 47.5±2.5 |
| GEPA | 50.0±1.3 | 32.2±1.4 | 45.2 (0.6) | 42.5±1.2 | 42.5±2.5 |
| Meta-Harness | 36.9±2.9 | 31.3±1.7 | 38.7±0.8 | 35.3±2.0 | 43.3±5.8 |
| **AutoSaddler** | **58.0** (1.8) | **36.5** (1.8) | 43.5±1.6 | **46.9** (1.8) | **50.0** (0.0) |

几个值得停下来看的数：

- GAIA2 平均涨 **9.0 个点**（53.0→62.0），SBP 涨 **9.6**（37.3→46.9），TB2 涨 **10.0**（40.0→50.0）。幅度相当能打。
- TB2 上甚至超过了手工专家调优的 Terminus KIRA（50.0 vs 47.5）。自动方法打赢人工调参，这个信号意义不小。
- GEPA 在 GAIA2 U21 和 SBP 部分仓库上不差（毕竟它只优化 prompt，而 prompt 确实低垂果实多），但**不稳定**——GAIA2 U22 上反而比默认 Harness 还低（47.9 vs 51.5）。Meta-Harness 更惨，两个基准上都没跑赢手工基线。这说明"优化整个 Harness"这件事本身不难想到，难的是怎么不让它学歪。
- 小瑕疵：SBP 的 Element-web 上 AutoSaddler（43.5）反而略低于 GEPA（45.2），Flipt 上也只有 36.5。跨仓库泛化到特定仓库时并不是全胜，论文没展开解释，我觉得值得追问。

### 效率：这个数让我愣了一下

看到 Figure 1 的数据时我停了几秒。AutoSaddler 用约 **1,000 次任务执行**就达到 72.3% 的 dev 准确率；GEPA 和 Meta-Harness 各消耗约 **2,800 次**，却只饱和在 64.6% 和 61.5%。按"优化实际利用的轨迹数"算更夸张：AutoSaddler 只用了 **147 条 traces** 就到最佳，Meta-Harness 用了 **1,400 条**——差了将近 10 倍。甚至 AutoSaddler 在 391 次 rollout 时（67.7%）就已经超过 Meta-Harness 的峰值了。

为什么能省这么多？关键在于每次更新都是"诊断驱动"而非"采样驱动"。进化方法靠大量变异-筛选碰运气，而这里每个补丁都带根因假设，命中率天然高。另外 dev 集过滤把大量"试了白试"的评估砍掉了。

---

## 🔬 消融：三条设计原则，每条值多少分？

这是全文我最喜欢的部分。三个消融变体对应三条设计原则，全在 GAIA2 上跑：

### RQ1：深度诊断值多少？——4.2 个点

把深度诊断换成"单次 LLM 调用读轨迹推断原因"（就是 GEPA 那一套浅反思），Pass@1 从 62.0 掉到 57.8。更细看过程：Epoch 1 结束（迭代 25）时，深度诊断攒下了 13 个被接受的补丁，浅层版只有 5 个。浅反思不是找不到问题，是找不准——根因错了，补丁自然站不住。

论文给的案例挺有说服力：

![日历任务案例研究](https://arxiv.org/html/2608.23041v1/rq1_case_1.png)

*图 7（论文 Figure 7）：一个日历任务的诊断对比。任务是"取消所有标记 Social 或 Personal 的日程，且每个有取消的日期补一条 Volunteer Social Work"。Agent 错在创建了 8 条而非 3 条。浅层诊断（左下，橙色）只能含糊地说"对任务描述理解有歧义"，给不出补丁；深度诊断（右下，绿色）通过多轮文件访问和轨迹分析，定位到第 38 步推理把 "at the same time as the cancelled event"（单数）错误泛化到每个取消事件，并给出具体补丁——给日历 hook 和系统提示词加歧义处理规则。*

### RQ2：结构化干预值多少？——5.1 个点

去掉补丁分类法和分阶段调度、允许无约束编辑后，从 62.0 掉到 56.9。更细的消融（附录 F，在 GAIA2 U22 上）：只去掉 Phased Patch Scheduling 从 60.7 掉到 54.8，去掉全部结构化干预进一步掉到 53.3。

背后的机制很有意思：无约束时补丁严重塌缩到 Steering 类（占比高达 91.5%）——因为改文本最容易，LLM 天然倾向偷懒。但接受率最高的恰恰是 Capability 类的硬干预：**新增工具** 83%、**Loop 逻辑修改** 71%、**基础设施修改** 67%。无约束时这些高价值补丁只占生成量的 4%，AutoSaddler 的结构化设计把它拉到 25% 以上。

还有一个耐用性数据（附录 G）：Capability 和 Steering 补丁的修复率相当（55% vs 58%），但回归率差了一倍多（8% vs 17%）。改代码的补丁更"耐用"，改文本的补丁更容易误伤——这解释了为什么先 Capability 后 Steering 的调度是合理的。

![优化轨迹上的代表性补丁](https://arxiv.org/html/2608.23041v1/analysis_rq2.png)

*图 11（论文 Figure 11）：GAIA2 优化轨迹上的代表性补丁与 dev 准确率曲线。探索期（Exploration）的补丁偏结构性：迭代 11 创建批量查询 API、迭代 13 修复静默的基础设施问题；进入精修期（Refinement）后转向文本调优，比如迭代 27 把一个禁止性 hook（"不要加多余的客套话"）重写为建设性的输出结构指导。曲线呈阶梯式上升——每个台阶对应一个被接受的关键补丁。*

### RQ3：泛化感知选择值多少？——11.4 个点，最大的单一项

去掉 Reflection 和 dev 集评估、只信训练集结果，从 62.0 暴跌到 50.6——比基线还低。细拆（附录 F）：只去 dev 集过滤从 60.7 掉到 50.0，再去掉 EvoDAG 反思掉到 44.9。

为什么掉这么狠？关键不在修复能力——两种设置的 fix rate 几乎一样。差异全在 **regression rate**：AutoSaddler 的回归率随迭代下降（-0.24 pp/iter），消融版在上升（+0.16 pp/iter）。论文给了一个很典型的案例：消融版在迭代 20 引入了一个新工具并给高频工具 send_message_to_user 加了强制重定向 hook，没有反思机制评估附带损害，这个补丁被保留下来，迭代 21 回归率从 8% 飙到 22%。AutoSaddler 在迭代 4 其实犯过一模一样的错，但**被反思机制拦住了**。

这就是"durable updates"的含义：不是修得更多，而是**少拆东墙补西墙**。做在线系统的人对这个直觉应该不陌生——线上热修复不修回归测试，最后全是补丁叠补丁的技术债。

---

## 💡 鲁棒性与迁移性

附录里三组实验值得提：

- **优化随机性**：另跑一次独立优化，GAIA2 上仍达 58.6%（vs 基线 51.5 左右），说明不是单次运气好；
- **训练分布偏移**：换一个训练 universe 优化，测试集仍达 57.4%，比基础 Harness 高 5.9 个点；
- **跨模型迁移**：把底层 LLM 从 Opus 4.6 换成 Haiku 4.5，保留优化好的 Harness，仍比基础 Harness 高 5.6 个点。这条对工程最有价值——Harness 的投资不完全绑定在特定模型上。

---

## 🤔 我的判断

**这篇论文最值钱的地方，不是框架本身，而是它把"LLM 系统优化"这件事的工程纪律讲清楚并且逐一验证了。** 深度调试优于浅反思、定向修改优于自由编辑、泛化筛选优于轨迹热修——每一条单看都像"正确的废话"，但消融告诉你它们值 4.2、5.1、11.4 个点。尤其是第三条，差距大到说明这是成败的关键变量。

跟同期工作比，它的位置很清晰：GEPA 证明了"文本反思可以当梯度用"，但优化对象只有 prompt；Meta-Harness 把对象扩展到整个 Harness，但没有泛化保护，实测甚至跑不过手工基线；AutoSaddler 相当于把两者的想法收进一个带 train/dev/test 纪律的完整训练范式里。说这是"底层突破"谈不上——mini-batch 类比、进化搜索、dev 集筛选都是现成零件——但作为工程整合，它是我见过的 harness 优化工作里协议最干净的一个。

**但有几个地方要泼冷水。** 第一，所有实验的底层模型都是 Claude 系（Opus 4.6 为主），优化器 Agent 也是 Claude Agent SDK——这套方法对强模型能力的依赖有多深？如果诊断 Agent 本身推理不够强，深度诊断的收益还在吗？论文没回答。第二，44 页 15 图，但主实验只有三个基准、每个基准只跑一次优化演化（test 三遍是评估重复，不是优化重复），优化过程的方差只在附录 C 里点了一下。147 条轨迹到最优这个数字很亮眼，但它是单次运行的轨迹，不同 seed 下的分布如何，没看到。第三，设定假设任务无状态独立，memory 和 skill 被排除在优化空间外——而很多真实 Agent 系统的 Harness 恰恰包含这两块。

**对工程的启发很直接。** 如果你在维护一个 Agent 产品，这套东西几乎可以直接抄：把 prompt/工具/hook 的版本管理起来当"参数"，失败轨迹自动诊断 + 结构化补丁空间 + dev 集回归门禁，再配一个记录演化历史的 DAG。就算不引入 LLM 自动诊断，光是"每个 Harness 变更必须过 dev 集回归"这一条纪律，就能拦住大部分越调越烂的死亡螺旋。另外那个 Capability/Steering 分类和先硬后软的调度策略，对人工调 Harness 同样是好的心智模型：先修工具和 loop 逻辑，再抠措辞。

还有一个更本质的问题论文没碰：优化出来的 Harness 学到的到底是"通用好的 Agent 设计模式"，还是"这个基准的怪癖"？跨模型迁移 +5.6 个点暗示前者占相当比例，但 GAIA2 不同 universe 间的波动（U21 已经 61.4，U22 只有 60.7 而基线才 51.5）说明基准特异性也不小。这个问题可能要等 HarnessOptBench 这类专门评测来给答案了。

---

## 📚 相关工作速览

| 方法 | 优化对象 | 选择机制 | 与 AutoSaddler 的关键差异 |
|---|---|---|---|
| GEPA（arXiv:2507.19457） | 仅 prompt | Pareto frontier | 无结构化补丁空间，无 dev 集泛化门禁 |
| Meta-Harness | 整个 harness | 无显式泛化保护 | 无约束编辑，实测不稳定 |
| TextGrad 类 | 文本组件 | 无 | 单次反传，无历史记忆与演化 |
| **AutoSaddler** | prompt+tool+middleware | dev 集验证 + EvoDAG | 三件套：深度诊断、结构化补丁、泛化筛选 |

---

## 📝 结语

AutoSaddler 给我的最大感触是：Agent 系统的优化正在从"手艺"变成"训练"。一旦你把 Harness 当参数、把失败轨迹当训练信号、把 dev 集当守门员，整个 mini-batch 训练的方法论——学习率调度、优化器状态、检查点选择——都能平移过来。这篇论文做对的是纪律，而不是某个单点技术。对做 Agent 产品的人来说，这是好事：纪律比技术好抄。

觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我
