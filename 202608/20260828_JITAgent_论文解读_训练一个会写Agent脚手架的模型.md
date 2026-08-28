# 别再手搓 Agent 脚手架了：JIT-Agent 训练一个 27B 模型，现场给你的任务量身定制 harness

上周调一个 DeepResearch 类的 Agent，模型没换，就把记忆模块从全量历史换成了证据矩阵，分数直接涨了 8 个点。这种"换皮如换刀"的体验，做 Agent 的人应该都有体感——**模型外面的那层脚手架（harness），往往比模型本身更能决定最终表现**。问题是，这层脚手架今天几乎全靠工程师手搓：记忆怎么压缩、规划怎么组织、工具怎么暴露、控制流怎么收敛，每一项都是反复试错。

LV-NUS Lab 这篇 JIT-Agent（arXiv:2608.25593）干的事，就是把手搓脚手架这件事变成一个**可训练的模型能力**：给任务描述，模型现场生成一套专属 harness，跑挂了会修，跑完还能从反馈里进化出更好的设计。

## 核心摘要

JIT-Agent 是一个基于 Qwen3.6-27B 训练的"脚手架智能"模型。它把 agent harness 形式化为记忆、规划、动作、能力编排四个模块组成的可执行代码工件，通过三阶段训练（任务条件定制 → 失败修复 → 在线进化）学会按需生成 harness。效果相当能打：DeepSeek-V4-Flash 套上 JIT 生成的 harness 后，在 DeepSearchQA 上反超 GPT-5.6 **9.1 个点**；已经很能打的 GLM-5.2 还能再涨最多 **20.2 个点**。更狠的是成本控制——对比 Claude Code、Codex、OpenCode 这些成熟 runtime，JIT-Agent 在全部 6 组受控实验里 token 和成本都是最低的，平均省 **36.0%** 的钱。我的判断：这不是又一篇刷榜论文，它把"Agent = 模型 + 脚手架"这个 2026 年的行业共识，第一次落成了一个**有训练管线、有理论框架、能持续进化的完整方案**。值得细读。

## 论文信息

- **标题**：JIT-Agent: Scaling Harness Intelligence via Just-in-Time Harness Evolution
- **作者**：Guibin Zhang, Leo Lu, Fangzhou Xie, Kang Zhu, Junhao Wang, Zhifei Xie, Zhaochen Yu, Zihang Liu, Zhongxiang Sun, Qiankun Li, Yue Liao, Heng Chang, Xiaobin Hu, Qibing Ren, Wangchunshu Zhou, Shuicheng Yan
- **机构**：LV-NUS Lab
- **链接**：https://arxiv.org/abs/2608.25593 ｜ 代码：https://github.com/bingreeky/JIT ｜ 模型：https://huggingface.co/JIT-Agent

![图1：四大 Agent 基准排行榜](https://arxiv.org/html/2608.25593v1/four_benchmark_leaderboard.png)

*图1：四个代表性基准（OfficeBench、AgentIF、DeepSearchQA、DeepPlanning-Shopping）的排行榜。带 JIT 角标的柱子是套了 JIT 生成 harness 的模型——GLM-5.2 和 DeepSeek-V4-Flash 装上 JIT harness 后分别冲上了多个榜单的第一名，而且 DeepSeek-V4-Flash 是个主打推理效率的便宜模型。*

---

## 🎯 问题：AOT 脚手架的根本矛盾

先讲清楚这篇论文反对什么。

2026 年"优化脚手架"已经不是新鲜事了。Stanford 的 Meta-Harness（arXiv:2603.28052）让 agent 读历史候选的源码和执行轨迹来搜索更好的 harness；Google DeepMind 的 AutoHarness、还有 AHE、Harness-R1 这一票工作，都在做类似的事。但它们共享一个假设，论文称之为 **AOT（Ahead-of-Time）**：脚手架是一个**耐久的工件**，在经验流上优化好之后，期望它能泛化到未来的任务上。

这个假设在部署分布稳定时是成立的。但你想想看——深度研究任务需要证据工作记忆，终端任务适合精简的串行 ReAct 循环，编码任务天然要靠文件系统当中介，宽搜索任务受益于并行证据探索。**任务异质性这么高，一个提前编译好的通用 harness 怎么可能处处最优？**

论文的替代方案叫 **JIT（Just-in-Time）**：不提前找通用脚手架，而是训练一个元智能体，看到具体任务后**现场生成**专属 harness。一句话讲，就是把"搜索最优脚手架"这个开销巨大的过程，**摊销进一个模型的参数里**。

这是整篇论文最核心的范式转换：从"优化一个 harness"变成"训练一个会写 harness 的模型"。

## 🏗️ 方法：四模块协议 + 三阶段训练

### 把 harness 变成可生成的结构化工件

让模型直接生成任意 agent 程序是不现实的——空间太大、没法验证。JIT-Agent 的第一步是把设计空间框死：任何 harness 都分解为四元组

$$\mathbf{h} = (\mathbf{M}, \mathbf{P}, \mathbf{A}, \mathbf{F}) \in \mathfrak{M} \times \mathfrak{P} \times \mathfrak{A} \times \mathfrak{F}$$

记忆 $\mathbf{M}$ 决定历史怎么压缩成视图，规划 $\mathbf{P}$ 把视图变成本步意图，能力编排 $\mathbf{F}$ 决定当前暴露哪些工具/技能，动作 $\mathbf{A}$ 消费组装好的上下文、更新状态并发出下一个动作。运行时依赖顺序是 $\mathbf{M} \rightarrow \mathbf{P} \rightarrow \mathbf{F} \rightarrow \mathbf{A}$，所有模块跑在同一个冻结的 backbone 上。

这个分解的表达力如何？论文给了几个对号入座：经典 ReAct 就是 $(\mathbf{M}_{\text{full}}, \mathbf{P}_{\emptyset}, \mathbf{A}_{\text{react}}, \mathbf{F}_{\text{all}})$；Codex、OpenCode 这类工程化变体是压缩记忆加 todo 规划的 ReAct；ROMA、AOrchestra 这类递归 agent 则是子问题隔离记忆 + DAG 分解 + 递归执行。**同一个协议下，市面上的主流 harness 都能写成这个四元组的具体取值**——这就把异构程序变成了可比较的坐标点。

在这个协议上，作者还建了 HarnessFactory，手工复现了 13 个代表性脚手架（ReAct、Plan-and-Execute、ReSum、Flash-Searcher、GAM、MemoBrain、AgentFold、ROMA 等），作为种子库 $\mathcal{B}_0$。这个库既是 Stage I 训练的参考材料，也是 Stage III 进化时的"现任前沿"。

![图2：JIT-Agent 方法总览](https://arxiv.org/html/2608.25593v1/jit_method_overview_final.png)

*图2：方法总览。左边是三类任务（深度研究、产品生成、自动研究）及各自需求；中间 JIT-Agent 接收任务后做任务条件的模块合成 h=(M,P,A,F)；右边是为不同任务生成的三套截然不同的专用 harness——深度研究用证据矩阵驱动定向重规划，产品生成编译成宽度≤2、深度≤3 的 DAG 执行器，自动研究则带确定性验证和进度板。同一个生成器，产出完全不同的执行协议。*

### 三阶段训练管线

![图3：JIT-Agent 训练管线](https://arxiv.org/html/2608.25593v1/jit_training_pipeline_frontier.png)

*图3：三阶段训练。Stage I 用冻结的强教师模型生成符合协议的任务适配 harness，SFT 加价值加权偏好学习；Stage II 把失败的 harness 和诊断报告（编译错误、接口不匹配、运行时异常）转成短程修复轨迹；Stage III 在线进化，候选组与种子库前沿对比，奖励/延迟/成本三路优势解耦归一化后驱动策略更新，胜出者进入 harness bank。*

**Stage I：定制。** 用一个更强的冻结教师模型 $q_\phi$，对每个任务采样 3 个同类型的参考脚手架作为上下文，生成任务适配的 harness，只保留通过协议验证和执行检查的样本做 SFT。光合规还不够，还要偏好学习：候选对 $(\mathbf{h}^+, \mathbf{h}^-)$ 的偏好定义为**奖励更高且延迟、成本不退化**（至少一项严格更优），用参考锚定的 DPO 式目标训练。

**Stage II：修复。** Stage I 里被丢掉的失败样本在这里变废为宝。教师模型对失败的 harness 提出结构化补丁 $\Delta^{(k+1)}$，确定性应用后重新验证，**只保留两轮内修好的轨迹**——目标很明确：部署时面对的是"差一点就能跑"的 harness，学的是少数几次高杠杆的修补，而不是推倒重来。这个设计我觉得挺务实的，工程上 80% 的失败确实是接口层的小毛病。

**Stage III：进化。** 这是最有意思的部分，叫 Evo-GDPO（Evolutionary Group-Decoupled Policy Optimization）。每个在线轮次，从当前 harness 库 $\mathcal{B}_n$ 检索参考集，策略采样一组候选，与现任最优（incumbent）在同一 backbone、同一预算、同一评测种子下执行。三路信号分开归一化：

- 奖励通道是主通道，超过 incumbent 有额外 bonus：$R_i^{\text{rew}} = r_i + \lambda_{\text{evo}}[r_i - b_r]_+$
- 延迟和成本通道只在**奖励不掉**的前提下激活：$R_i^{\text{lat}} = \mathbb{1}[r_i \geq b_r][b_\ell - \bar{\ell}_i]_+$

合并时强制奖励权重主导（$w_{\text{rew}} \gt w_{\text{lat}} + w_{\text{cost}}$），再做一次 batch 级归一化，最后套 PPO 式裁剪目标。和标准 GRPO 的区别在于：奖励的对象**不是组内相对好，而是超越历史前沿这件事**。库的更新也是保守的——新 harness 必须匹配当前奖励前沿且至少在一个维度上严格改进才保留。

说实话，看到 Evo-GDPO 这三路解耦的设计我愣了一下。奖励、延迟、成本三个量纲完全不同的信号硬合在一起训练，数值尺度互相碾压是经典翻车点，这里用分组归一化加门控激活处理得挺干净。这是从 GDPO 改过来的思路，但"超越 incumbent"这个目标设定是真正的新意。

## 📊 实验：数字相当硬

### 主实验：9 个基准，开源 backbone 打前沿模型

评测覆盖四类任务共 9 个基准：深度研究（BrowseComp-Plus、DeepSearchQA、xBench-DS）、日常工作（AgentIF-Oneday、PinchBench）、规划（DeepPlanning-Shopping/Travel）、工作区（OfficeBench、OdysseyBench）。

| 模型 | BC+ | DSQA | xBench | AgentIF | Pinch | Shop | Travel | Office | Odyssey |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.7-Plus | 70.5 | 78.0 | 75.0 | 59.9 | 80.5 | 75.2 | 52.1 | 56.6 | 67.7 |
| GLM-5.2 | 72.0 | 89.2 | 76.0 | 63.0 | 87.0 | 78.2 | 62.8 | 63.0 | 75.3 |
| DeepSeek-V4-Flash | 68.1 | 76.2 | 70.1 | 58.4 | 81.7 | 59.1 | 54.8 | 61.0 | 71.0 |
| GPT-5.6 | 76.9 | 76.0 | 81.0 | 68.0 | 84.2 | 83.7 | **84.9** | 65.3 | 68.7 |
| Gemini 3.5 Flash | 75.0 | 88.0 | 85.0 | 64.0 | 74.2 | 76.2 | 50.3 | 63.3 | 78.0 |
| **JIT + GLM-5.2** | **78.0** | **93.9** | **88.0** | **69.9** | **93.3** | 83.4 | 83.0 | **68.4** | **78.7** |
| **JIT + DS-V4-Flash** | 74.0 | 85.1 | 82.0 | 63.8 | 92.9 | **83.9** | 61.3 | 63.4 | 73.0 |

*表：主实验结果摘录（满分 100）。加粗为该列最优。*

几个值得停下来看的数：

- 18 组 backbone-基准配对**全部提升**，无一例外。GLM-5.2 九项均分从 74.1 涨到 81.8（**涨 7.7 个点**），DeepSeek-V4-Flash 从 66.7 涨到 75.5（**涨 8.8 个点**）。
- 最大涨幅出现在需要持续状态管理的任务上：DeepSeek-V4-Flash 在 DeepPlanning-Shopping 从 59.1 涨到 83.9，**24.8 个点**。GLM-5.2 在 DeepPlanning-Travel 涨 **20.2 个点**。
- DeepSeek-V4-Flash 是个效率向模型，套 JIT harness 后在 DeepSearchQA（85.1 vs 76.0）、PinchBench（92.9 vs 84.2）、OdysseyBench（73.0 vs 68.7）上反超 GPT-5.6，而且在每个基准上都打过了自家大哥 DeepSeek-V4-Pro，平均领先 8.7 分。

便宜模型加好脚手架反超贵模型——这个结果和 Meta-Harness 让 Haiku 4.5 登顶同量级榜单一事互相印证。脚手架智能正在成为一种可以替代模型规模的能力来源。

### 与成熟 harness 的受控对比

这部分是全文我最喜欢的实验设计：**固定 backbone，只换 harness**，同时报告 token 消耗和美元成本。对比对象是 Claude Code、Codex、OpenCode、Hermes、NanoBot 五个成熟 runtime。

| Backbone | Harness | DSQA Perf | DSQA Cost | xBench Perf | AgentIF Perf |
|---|---|---|---|---|---|
| DS-V4-Flash | Claude Code | 79.6 | $0.088 | 75.0 | 66.9 |
| DS-V4-Flash | Codex | 77.8 | $0.107 | 70.0 | 58.5 |
| DS-V4-Flash | OpenCode | 75.9 | $0.258 | 65.0 | 48.1 |
| DS-V4-Flash | NanoBot | 80.4 | $0.131 | 78.0 | 53.1 |
| DS-V4-Flash | **JIT-Agent** | **85.1** | **$0.066** | **82.0** | 63.8 |
| Qwen3.6-Flash | Claude Code | 72.8 | $0.140 | 58.0 | 55.4 |
| Qwen3.6-Flash | NanoBot | 74.2 | $0.197 | 63.0 | 43.5 |
| Qwen3.6-Flash | **JIT-Agent** | 70.3 | **$0.095** | **70.0** | **58.3** |

*表：受控 harness 对比摘录。JIT-Agent 在 6 组设置中 4 组性能第一，6 组全部 token 和成本最低。*

性能上 JIT-Agent 赢了 6 组中的 4 组，输的两组也都有说法：DeepSeek-V4-Flash 上 AgentIF 落后 Claude Code 3.1 分，Qwen3.6-Flash 上 DeepSearchQA 落后 NanoBot 3.9 分，但用的 token 都少得多。**没有任何一个固定 harness 能跨任务通吃**——NanoBot 在 Qwen3.6-Flash 的 DeepSearchQA 上最强，到了 AgentIF 就落后 JIT-Agent 14.8 分。这恰好反过来论证了 JIT 范式的动机：与其全局选一个耐久脚手架，不如按任务生成。

成本上更夸张：JIT-Agent 在全部 6 组里 token 和花费都是最低的，比每组最省的固定 harness 还要再省 14.9%–54.1%，平均省 36.0%。DeepSeek-V4-Flash 的 xBench-DS 上，token 从 527K 降到 212K，性能反而从 78.0 涨到 82.0。**涨分不是靠拉长轨迹堆出来的，而是更短、更精选的执行**——这一点反驳了"提升来自更多推理算力"的常见质疑。

![图4：成本-性能帕累托前沿](https://arxiv.org/html/2608.25593v1/advanced_harness_cost_performance_dsqa_agentif.png)

*图4：DeepSearchQA 和 AgentIF 上的成本-性能散点。实心点是 DeepSeek-V4-Flash，空心是 Qwen3.6-Flash，颜色区分 harness。JIT-Agent（绿色）的点全部贴着左上角的帕累托前沿——DeepSearchQA 上 DeepSeek-V4-Flash + JIT 以 $0.066 的成本拿到 85.1 分，比最强固定 harness NanoBot 高 4.7 分还便宜 49.6%。*

### 跨模型泛化与在线进化

![图5：JIT harness 对 ReAct 的全面胜出](https://arxiv.org/html/2608.25593v1/model_pair_jit_vs_react.png)

*图5：三个模型家族（DeepSeek V4、Qwen 3.6、Mimo 2.5）各两个变体，固定 ReAct harness（虚线）对比 JIT 生成 harness（实线）。24 组配对全部提升，平均 +7.6 分。DeepSeek 家族平均 +10.2，Mimo 家族 +8.6，Qwen 家族 +4.0。DeepSearchQA 单项平均涨 15.2 分，Mimo-V2.5-Pro 最高涨 22.2 分。*

24 组配对全胜，而且增益不是某个 backbone 的特例——这说明 JIT-Agent 学到的确实是"怎么给任务配脚手架"的通用知识，而不是在补偿某个特定模型的缺陷。

还有一个 Figure 6 展示的 streaming 进化实验（该图是论文内联绘制，无独立图片）：对比 Static JIT（每个任务独立生成）和 Streaming JIT（持续把执行反馈滚进 harness 库），后者在 DeepPlanning-Shopping/Travel 和 OfficeBench 三条任务流上累计准确率都更高，且成本和工具调用量没有同步膨胀。经验在库之间传递，参数不动——这正是 Stage III 训练目标的部署形态。

## 🔬 生成物长什么样：两个案例

数字之外，JIT-Agent 实际生成的 harness 才是最有说服力的部分。

![图7：Palimpsest 案例](https://arxiv.org/html/2608.25593v1/palimpsest_flow.png)

*图7：Palimpsest——图规划的工件生产。任务是"收集除 Tom 外的所有联系人卡片、建排序工作簿、发邮件给 Tom"。GraphPlanPlanning 把需求编译成依赖 DAG（发现卡片和读 schema 是过滤的前置，建簿等规范化记录，发送等工件验证），GraphPlanAction 以宽度≤2、深度≤3 的有界并行执行，GraphPlanMemory 把中间结果存成工件存储供下游节点直接消费。*

![图8：Trapdoor 案例](https://arxiv.org/html/2608.25593v1/trapdoor_flow.png)

*图8：Trapdoor——委托即运行时原语。任务是多跳身份考证（线索横跨 1901 年名言、1974 年大学合并、2023 年回忆录和一篇 2010 年论文）。这种证据路径不确定的任务不适合提前定死 DAG，所以生成的 harness 用 DynamicDecomposer 动态拆子问题，并合成了一个 delegate 工具：调用它会被 OrchestratorLoop 拦截，开一个最多 5 步的私有研究子 agent，答案作为普通观测回注主循环，FactGraphMemory 同步抽取键值事实。*

对比一下这两个案例：同一个生成器，给工件生产任务生成"DAG 执行 + 工件存储"，给深度研究任务生成"递归委托 + 事实图存储"。**协议约束的是接口，不是行为**——这句话是全文方法论的点睛之笔。附录里还有 8 个类似案例（层级上下文折叠、完成门控工具访问、阶段条件执行、证据矩阵搜索等），每个 harness 的名字（Origami、Turnstile、Gearbox、Pegboard……）起得还挺有味道。

## 🤔 我的判断

**最值钱的地方**：把 harness 优化从"外部搜索启发式"变成"模型内化的训练能力"，这是真范式区别。Meta-Harness 那类 AOT 方法每次部署都要跑一轮昂贵的搜索循环（一步优化动辄上千万 token 的诊断上下文），JIT-Agent 把这个成本摊销进了 27B 参数，推理时一次生成。三阶段管线里 Stage II 的"失败变修复教材"和 Stage III 的"超越 incumbent"目标，都是可以拿去复用的独立贡献。

**需要泼的冷水**：

1. **first 这个声明要打个折**。论文自称"第一个专为 JIT harness 生成构建的模型"，但 Harness-R1（同样引用了）已经训练生成器加学习修复加在线进化，差别主要在 AOT 测试时编辑 vs JIT 现场生成。把 JIT-Agent 定位成"把这条线推到极致"更准确。
2. **四模块协议的天花板**。论文自己也承认，Claude Code、Codex 这些生产级 runtime 的机制远比四模块实例化丰富。当前的赢面部分来自"紧凑 harness 也够打"这个发现，但当任务复杂度继续上升，协议本身可能需要扩展。作者把这留给了未来工作，坦诚是坦诚，但现在的方案显然不是终点。
3. **评测的时效性存疑**。GPT-5.6、Gemini 3.5 Flash、DeepSeek-V4 这些都是 2026 年中的模型，榜单格局半年一换，+9.1 分的反超能维持多久不好说。另外 Figure 5 用的子集只有 50–100 个样本，统计噪声不小——不过 24 组全胜的一致性多少弥补了单点的不确定性。说实话，这类"harness 工程"论文最缺的其实是真实生产流量的长尾任务，benchmark 覆盖的任务结构仍然偏规整。
4. **生成成本的隐藏账**。JIT-Agent 本体是 27B 模型，每个任务要先跑一遍生成加验证加可能的修复，这部分开销没有算进表 4 的 API 成本里（那里只算了执行 rollout）。对短任务来说，生成 harness 的固定开销可能吃掉一部分效率收益。论文没有给出这个数字，算是个小遗憾。

**对工程的启发**：如果你在做 Agent 产品，这篇论文最直接的 takeaway 不是去复现 Evo-GDPO，而是那套**四模块分解的视角**——把你的 harness 拆成记忆、规划、动作、能力编排四个可替换坐标，你会发现很多"调 prompt"的苦活其实是在错误的模块上用力。再进一步，静态场景下哪怕只做"任务类型 → 手工挑选模块组合"的路由，大概率也能吃到一部分 JIT 的红利。

## 收尾

模型规模的叙事讲了三年，这篇论文和 Meta-Harness 那批工作一起，把叙事权往"脚手架"这边扳了一扳。模型决定上限，脚手架决定你能摸到多接近上限的地方——而脚手架这件事，现在也能训练了。模型-脚手架协同设计（co-design）大概率是接下来一两年 Agent 方向的主线之一。

如果你也在手搓 harness，建议至少去读一下论文的四模块协议和 HarnessFactory 部分，就算不复现，也能帮你理清自家系统的模块边界。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
