# NanoResearch：研究自动化跑得起来不算赢，跑出"你这个研究员要的东西"才算赢

## 核心摘要

The AI Scientist 这类全流程科研自动化系统这两年挺火。从 ideation 到 paper writing，端到端跑通，听起来很美好——本来要一周的工作，几小时跑完。但你真的用过就知道，**那个 paper 跟你"要的 paper"是两码事**。

我之前帮一个朋友试过这类系统。他是做 AI4Science 的，最关心的是"这个方法在真实生物医学场景里有没有用"。结果 system 给他生成了一堆"模型架构创新 + 在 ImageNet 涨 0.3%"的论文——纯 CV 学术圈味儿。对他来说价值约等于零。

这篇 NanoResearch 就盯着这个 gap：**personalization 不是锦上添花，是 research automation 能不能真正用起来的前提条件**。它提出一个 tri-level co-evolution 框架——**Skill Bank（可复用过程知识）+ Memory Module（用户/项目历史）+ Policy Learning（隐式偏好内化）**——三个层次共演化，让系统跑得越久越懂你。

20 个研究主题、7 个领域的实验显示，NanoResearch 在 simulated 和 human evaluator 两套评估下都比 AI Scientist、AI-Researcher、EvoScientist 等 SOTA 好得多，更关键的是**多 cycle 下能持续自我提升**——不是一次性 demo。

读完我的感觉是：这是一篇**问题定义比方法本身更值钱**的论文。"personalization 是 research automation 的必要前提"这个 framing 很有可能成为未来这个方向的共识。

---

## 论文信息

- **标题**：NanoResearch: Co-Evolving Skills, Memory, and Policy for Personalized Research Automation
- **作者**：Jinhang Xu, Qiyuan Zhu, Yujun Wu, Zirui Wang, Dongxu Zhang 等（共 13 人）
- **机构**：上海人工智能实验室、香港科技大学、北京大学、浙江大学、西安交通大学、华东理工大学、香港中文大学
- **arXiv**：https://arxiv.org/abs/2605.10813

---

## 问题：研究自动化的"假民主"

我先吐槽一下现有系统的 framing。

The AI Scientist 那一脉，包括 v2、EvoScientist、AI-Researcher、DeepScientist——它们的卖点都是"能跑通全流程"。但跑通是一回事，**跑出来的东西是不是用户要的**完全是另一回事。

论文里有一段我觉得讲得特别到位：

> AI4Science researchers prioritize whether a method addresses a meaningful real-world need, while core CV researchers value architectural novelty and consistent benchmark gains.

同一个 prompt（比如"研究 vision transformer 的 efficiency improvements"），不同背景的研究员想要的 paper 完全不一样：

- **效率系研究员**想要：FLOPs 减少多少、latency 降低多少、能不能跑在边缘设备上
- **理论派研究员**想要：为什么这种 transformer 更高效的理论解释、收敛性分析
- **应用派研究员**想要：在具体下游任务（医疗、自动驾驶）的 benchmark gain
- **架构创新派研究员**想要：组件设计的 novelty、ablation 上每个 component 的贡献

现在的 system 给所有人输出**同一篇 paper**——通常是"benchmark gain + ablation study + 通用 method writeup"——这个 paper 对每个人都不完美，最后等于对每个人都没用。

![图1：单一流水线 vs NanoResearch 的对比](https://arxiv.org/html/2605.10813v1/x2.png)

*图1：上半部分 (a) 是传统 research automation 系统——所有 user 走同一条流水线，输出同质化的 paper。下半部分 (b) 是 NanoResearch——识别不同 researcher persona，提供个性化的 skills 和 feedback。最关键的区别在右下角——不同 persona 沿着不同的演化路径成长。*

---

## NanoResearch 解决了什么？三个 capability gap

论文把"现有系统做不到 personalization"的原因拆成三块：

### Gap 1：缺乏可复用的过程知识

每次跑一个新 project，从零开始。同样的 debugging pattern、同样的 config 配错、同样的 dataset loading 问题——重新踩一遍。即使是 EvoScientist 这种带 memory 的系统，存的也是 episode-level narrative（"上次跑实验出了什么事"），不是 compact procedural rule（"碰到这种 error 就这么修"）。Transferability 差。

### Gap 2：不积累用户特定经验

上一次的 hypothesis、validated config、推断出的资源约束——session 结束就丢了。下次同样的 user 来，又得从 generic prior 开始 planning，完全没用 user 的历史。

### Gap 3：无法内化隐式偏好

User 可能说"我更喜欢简单的方法"或"多做点 efficiency 分析"。这种 feedback 太 diffuse 没法直接 encode 成 rule，太 nuanced 没法压缩成 memory entry。如果没有 mechanism 把这些 signal 转成 persistent parameter change，那 preference 一过 context window 就忘了。

---

## 三层共演化：Skill / Memory / Policy

![图2：NanoResearch 框架总览](https://arxiv.org/html/2605.10813v1/x3.png)

*图2：核心组件是 Orchestrator $\mathcal{O}$，它协调三个阶段（Ideation → Experimentation → Writing），同时管理 Skill Bank $\mathcal{S}$ 和 Memory Module $\mathcal{M}$。每个 stage 结束 user 给 natural language feedback，feedback 通过 SDPO 算法被 internalize 到 Orchestrator 的 planner $\pi_\theta$ 里。三个层次形成 co-evolution loop——skill execution 充实 memory，memory 改善 planning，preference learning 让系统对齐 user。*

### 三个组件的分工

| 组件 | 存什么 | 怎么用 |
|------|--------|--------|
| **Skill Bank $\mathcal{S}$** | Compact 的过程知识（如 debugging pattern、调用模式）| 跨 project 复用，相当于系统的"长期程序记忆" |
| **Memory Module $\mathcal{M}$** | User/project-specific 经验（如失败的 hypothesis、有效的 config）| Ground planning 到 user 实际历史 |
| **Planner Policy $\pi_\theta$** | User 偏好（通过参数更新内化）| 重塑 coordination 行为 |

这个分层我觉得设计得很合理，**对应了人脑的三种记忆**：
- Skill Bank ≈ 程序性记忆（如何骑车）
- Memory Module ≈ 情景记忆（上周去哪儿吃饭）
- Planner Policy ≈ 习惯/偏好（自动倾向某种处理方式）

---

## 三个 Stage 的细节

### Stage I：Idea Generation and Planning

分两个 phase：

**Ideation phase**：Orchestrator 检索 topic 和 user 相关的 skill 和 memory：

$$\mathcal{S}_I, \mathcal{M}_I = \text{Retrieve}(\mathcal{S}, \mathcal{M} \mid \mathcal{T}, \mathcal{U})$$

然后用 ReAct loop 在 arXiv、Semantic Scholar 上做文献综述，找 research gap。这里有个细节：用 **quantitative evidence extraction**（直接从 paper 里抓性能数字）来减少幻觉。最后做 novelty verification 滤掉重复工作。

**Planning phase**：把最佳 hypothesis $h^*$ 转成 JSON-formatted blueprint $\mathcal{B}$，再过一遍 LLM reviewer 的 peer-review 修正 loop：

$$\mathcal{B}^{(t+1)} = \text{Refine}(\mathcal{B}^{(t)}, c_\mathcal{B}^{(t)}, P_P, E)$$

直到 review 通过或达到重试上限。

### Stage II：Experimental Validation and Optimization

三个 phase：

**Setup and Coding**：基于 blueprint 生成完整代码库 $\mathcal{W}$，包括 model、training、evaluation、cluster submit script

**Execution and Debugging**：投到 SLURM cluster 跑，跑不通就用 skill 和 memory 自动 debug：

$$\mathcal{W}^{(t+1)} = \text{Debug}(\mathcal{W}^{(t)} \mid \mathcal{S}_C, \mathcal{M}_C)$$

**Analysis**：parse 原始 log，生成 analysis report：

$$\mathcal{A} = \text{Analyze}(R_\text{raw}, \mathcal{B}, \mathcal{T})$$

每个 phase 结束都 update skill 和 memory。

### Stage III：Paper Writing and Review

Writing agent section-by-section 写作（避免 context 爆掉），Review agent 不用 skill / memory（保证 unbiased），critique 之后 iterative revise 到达 quality threshold。

每个阶段都把 trajectory 蒸馏成新的 skill / memory。

---

## 关键创新：Label-Free Policy Learning

这是 NanoResearch 最有意思的部分。

**问题**：User 的 feedback 是自然语言（如"我更喜欢简化的实验设计"），不是 reward score 也不是 preference pair。怎么把这种 free-form feedback 转成模型的 persistent change？

**做法**：用 **Self-Distillation Policy Optimization (SDPO)** —— 一种最近提出的、不需要 reward model 或 preference annotation 的对齐方法。给定 Orchestrator input $x$ 和 planner 的 initial trajectory $y \sim \pi_\theta(\cdot|x)$，SDPO 把**带反馈条件的 model** $\pi_\theta(\cdot|x, \mathcal{F}, y_{<t})$ 作为"自己的更好版本"做 distillation 目标，把单个 feedback 实例转成 token-level 的 dense learning signal。

这个 idea 我觉得挺巧的。传统对齐需要 thousands of preference pair，对每个 user 单独训练完全不现实。SDPO 把"读了 feedback 后的自己"作为 teacher 来蒸馏"没读 feedback 的自己"——一次 feedback 就是一次梯度更新，把 preference 内化成参数。

这意味着：**user 不需要标 1000 条数据，跟系统聊几轮就行**。每次反馈都让 planner 微调一次，几个 cycle 后系统真的"懂"这个 user 了。

---

## 实验结果

20 个研究主题，7 个领域，simulated 和 human researcher 两套评估。

### 主要发现

1. **NanoResearch 在所有 baseline 上 substantial gain**。具体分数我懒得搬（论文 Table 2、Table 3），但相对 AI-Scientist-v2、AI-Researcher、EvoScientist 这些 SOTA 都有显著提升

2. **Preference alignment 强**。这是 NanoResearch 独有的评估维度——其他系统输出对所有 user 一样，没法比 personalization。NanoResearch 在 "this paper matches my style" 的人评上明显好

3. **Progressive improvement**。多 cycle 跑下来，质量持续涨、cost 持续降。这是 self-evolve 系统应该有的性质——但很多 baseline 没做到

4. **三个组件都必要，但 isolation 下都不充分**。这是论文消融的关键 message：
   - 只有 Skill 没 Memory：可以处理通用问题，但区分不了 user
   - 只有 Memory 没 Skill：可以诊断问题，但 prevent 不了 recurring failure
   - 两者都有但没 Policy learning：跟不上 evolving user intent

### 我的一点保留

实验整体让人 convinced，但有几个点想吐槽：

1. **"Substantial gains" 的具体数字我从摘要看不太到**——需要看完整 paper 才能判断 gain 的量级
2. **Human evaluator 的数量、背景、bias 没说清楚**。research evaluation 这种主观任务，evaluator pool 的设计是关键
3. **运行 20 个 topic 的 compute cost 没有完整报告**。这套 system 跑 1 个 paper 的 cost 估计不便宜，多 cycle 跑下来 cost 是怎么变的，这点对实用性很关键

---

## 我的判断：值不值得读？

**强推**，特别是如果你在做 agentic system 或 research automation。

**亮点**：

1. **Problem framing 非常 sharp**。"Personalization is a precondition" 这个 framing 我之前没见有人正经说过，但说出来之后觉得显而易见
2. **三层架构对应人脑三种记忆**——Skill / Memory / Policy 分工清晰，对应了不同 nature 的信息存储
3. **SDPO 把 free-form feedback 转成参数更新**这个机制是关键创新。过去对齐方法都依赖 structured signal（reward / preference pair），不适合 research 这种 nuanced feedback 场景
4. **Co-evolution loop** 是一个 clean abstraction：skill 充实 memory，memory 强化 planning，preference 重对齐——闭环。这是很多 multi-agent 系统嘴上说做不到的

**问题**：

1. **System 复杂度爆炸**。Orchestrator + 3 个 stage × 多个 agent + Skill Bank + Memory Module + Policy Learner——这套东西的 engineering complexity 是 nightmare。复现难度很高
2. **Skill / Memory 的"distillation"和"merge"过程**是黑盒。论文用 LLM 做这些操作但没充分讨论 LLM 自己会不会引入 bias——比如某些 user feedback 在 distillation 时被错误归类成 generic skill
3. **Cycle 之间的 user 是同一个吗**？如果是同一个 user 跑多个 paper，那 personalization 很合理；如果是 cross-user 评估（系统对 user A 学完，对 user B 表现如何），那需要更多实验数据
4. **缺乏跟 Memory-Enhanced Agent 这一脉工作的对比**。MemGPT、Letta 等也是做长期记忆，但应用在 general agent 上。NanoResearch 应该明确说一下自己 vs 这些 general memory system 的区别

**对工程实践的启发**：

- **如果你在做长期 agent 系统，先把记忆分层**。Skill / Memory / Preference 三个层次有截然不同的存储要求和更新机制
- **Free-form feedback 不要硬塞进 memory**。它会被压缩或召回时漏掉。要么直接更新模型参数（SDPO 思路），要么用 hierarchical summary
- **每个 cycle 都要 distill 新 skill**。不能让系统每次都从零开始 debug——同样的错误踩两次是工程失职

---

## 收尾

NanoResearch 这篇我觉得在 research automation 这个赛道是一个**重要的 framing reset**。

过去这个领域的 ambition 是"让 AI 自己做研究"——这个 framing 隐含着"AI 的研究 = 一个固定的好研究"。但真实研究是 contextual 的——不同社区、不同资源、不同目标，对"好研究"的定义完全不同。NanoResearch 第一次把这个常识写到论文动机里。

下一步值得追的问题：

1. **Personalization 的 transferability**：A user 学到的 preference 能不能 zero-shot 推断给 similar user？还是必须每个 user 重新跑几轮 feedback loop？
2. **Adversarial preference**：如果 user 的 preference 本身有问题（如喜欢一些 questionable practice），系统是该 internalize 还是 push back？这是个伦理 + 工程的双重问题
3. **Cost-quality Pareto frontier**：多 cycle 持续 evolve 的 ROI 曲线长什么样？什么时候开始 saturate？

这个方向接下来一两年应该会很热闹。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我*
