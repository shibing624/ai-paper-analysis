# HeavySkill：撕掉 Agent Harness 这层壳，里面真正干活的其实是模型自己的"重思考"

最近做 Agent 项目的朋友都会有一个共同感受：那套花里胡哨的 orchestration harness——orchestrator、subagent、skill library、memory、tool use——看起来挺玄乎，真把它拆开来看，能力提升到底是从哪儿来的，其实没几个人说得清。

我自己之前在调一个多智能体系统的时候，就反复怀疑过这个问题。Orchestrator 派活、subagent 并行干、最后再 summary 一下，这套流程跟 Best-of-N 加一个 reranker，本质区别有多大？是这套 harness 真的提供了认知层面的额外能力，还是它只是把模型本来就有的某种潜能，用一种更复杂的方式包装了一遍？

这篇 HeavySkill 给出的答案挺有意思：所谓 agent harness 的核心，可以剥到只剩两步——**并行推理（parallel reasoning）+ 顺序审议（sequential deliberation）**。这两步不是 harness 的产物，而是大模型自己就该内化的一项"内在技能"。把它做成一个 skill 文件，丢到任意 harness 里就能用。

更扎心的一点：在足够强的模型上，这套"重思考"能逼近 Pass@N 的理论上限——也就是说，模型确实在审议阶段做了真正意义上的"跨轨迹综合"，而不是在挑挑拣拣。

---

## 一句话摘要：这是一个"祛魅"agent harness 的工作

**痛点**：Agent harness 越搞越复杂，但真正驱动性能的机制被复杂系统设计糊得看不见。**核心方案**：把 harness 简化成两阶段——K 条并行推理轨迹 + 一个 deliberation 模型做综合。再把这套流程封装成一个纯文本 skill 文件，注入任何支持 skill 的 harness 都能跑。**关键效果**：在 STEM 类任务上 HM@4 全面碾压 Vote@K 这类 BoN 基线，Kimi K2 Thinking 和 GPT-5 Thinking 上接近 Pass@K 的理论天花板；GPT-OSS-20B 在 LiveCodeBench 上从 M@K 69.7% 直接拉到 HM@4 85.5%。**真实定位**：不是底层架构创新，更像是一篇"经验研究+概念整合"的论文，价值在于把多个工业界框架（Kimi K2、PaCoRe、LongCat-Flash-Thinking）背后的共同模式做了系统化的实证研究，并把它"skill 化"。值得做 Agent 系统的工程师读一遍。

---

## 论文信息

- **标题**：HeavySkill: Heavy Thinking as the Inner Skill in Agentic Harness
- **作者**：Jianing Wang、Linsen Guo、Zhengyu Chen、Qi Guo、Hongyu Zang、Wenjie Shi、Haoxiang Ma、Xiangyu Xi、Xiaoyu Li、Wei Wang、Xunliang Cai（共 11 位作者，最后一位 Xunliang Cai 是 Meituan LongCat 团队负责人，论文中也引用了自家 LongCat-Flash-Thinking-2601）
- **arXiv**：[2605.02396](https://arxiv.org/abs/2605.02396)（2026-05-04）
- **代码**：https://github.com/wjn1996/HeavySkill

---

## 一、问题动机：harness 越来越复杂，到底哪一块在真正干活？

回想一下你最近看到的 Claude Code、CodeX、OpenClaw、Hermes 这些 agent harness。它们的描述往往是这样的：

- 一个 orchestrator 负责理解任务、做 planning、派活
- 多个 subagent 并行执行子任务
- 一个 skill library 提供可复用策略
- 一个 memory 模块存上下文
- 一组 tool 接口接外部能力

听起来非常 fancy，但作者其实想问的就是一句话——这堆模块去掉哪些，性能会塌？

他们的判断是：**抽掉所有外壳之后，剩下的核心就是一个两阶段循环**——orchestrator 派 K 个 subagent 并行做，subagent 把答案交回来 orchestrator 再综合一遍。这其实就是测试期扩展（test-time scaling, TTS）里大家熟悉的并行采样 + 综合那一套。

这个判断如果成立，结论就有点扎人——大部分 harness 的"系统创新"，其实是在变着花样实现"重思考"这件事。你看 Kimi K2、PaCoRe、LongCat-Flash-Thinking 这些工作，它们各自都有自己的招牌，但拆开都是"并行轨迹 + 顺序综合"两步。

我的第一反应是：这个观察其实挺扎实的。我之前在做某个 reasoning agent 的时候，反复试过加各种 verifier、加 memory、加 sub-skill，最后发现真正涨点最稳的其实就是把采样数从 1 拉到 K，再加一个像样的 aggregation。所谓 harness 复杂度，很多时候是工程组装的副作用，不是认知层面的突破。

---

## 二、方法核心：极简的两阶段架构 + memory cache + 迭代审议

### 2.1 整体框架：parallel reasoning + sequential deliberation

![图1：HeavySkill 整体框架——并行推理生成 K 条独立轨迹，序列化进 memory cache，再交给 deliberation 模型做综合](https://arxiv.org/html/2605.02396v1/x1.png)

*图1：HeavySkill 的整体架构。左侧是 K 个并行的 reasoner 各自从头独立解题，生成 K 条 trajectory；中间是 memory cache，对 trajectory 做剪枝和打乱顺序；右侧是 deliberation 模型读取 cache 后做综合，并可以多轮迭代。整套流程既可以当作 Python pipeline 跑，也可以蒸馏成 skill 文件让 orchestrator 自动执行。*

形式化一点说：

- 给定问题 $q$，并行推理阶段产出 $\mathcal{T}_{\pi_{\theta}}(q, K) = \{y_1, \ldots, y_K\}$，$K$ 条独立轨迹由模型 $\pi_{\theta}$ 生成
- 顺序审议阶段用另一个模型 $\pi_{\phi}$（默认跟 $\pi_{\theta}$ 是同一个）读取序列化后的 cache $x_c = \mathcal{C}(\mathcal{T}_{\pi_{\theta}}(q, K))$，产出 $K^{(1)}$ 份综合内容

注意一个工程细节：reasoning model 生成的轨迹里，thinking content 通常非常长，全部拼起来直接超 context 长度。所以 memory cache 做了两件事——**剪枝**（保留答案部分，思考过程做摘要）**+ 打乱顺序**（防止模型对位置敏感）。这两点在工程实现里相当关键，作者把它叫做"serialized memory cache"。

### 2.2 迭代审议：让模型反复"回看自己刚才想了什么"

光是一次 deliberation 还不够，作者还加了一个迭代机制。第 $t$ 轮的 cache 是这样构造的：

$$x_c^{(t)} = \mathcal{T}_{\pi_{\phi}}(x_c^{(t-1)}, K^{(t-1)}) \,||\, x^{(t-1)}$$

也就是把上一轮的 deliberation 输出再拼回 cache，继续做下一轮。这个设计灵感来自人类反复推敲想法的行为——你写完一段方案，过几个小时再回头看，往往会发现自己之前没考虑到的点。

这里有个问题我后面会展开：**迭代不是越多越好**。论文的实验里，迭代次数从 1 到 4，HM@K 是涨的，但 HP@K 反而下降——也就是说，"平均质量"在涨，但"理论上限"在被信息冗余拖累。这其实是一个很值得玩味的现象。

### 2.3 Skill 化：从 Python pipeline 到一份纯文本文件

这是论文标题里那个"Skill"的来源。作者把整套 workflow 蒸馏成一份 skill 文档，分四块：

| 模块 | 内容 |
|------|------|
| **Activation Conditions** | 描述什么时候该触发 heavy thinking（复杂推理任务触发，日常对话不触发，避免不必要的开销） |
| **Parallel Reasoning Protocol** | 指导 orchestrator 启动 K 个 subagent，鼓励它们用不同策略（比如代数 vs 几何） |
| **Deliberation Prompt** | 核心 prompt 模板，要求模型 1) 分类查询类型 2) 批判性评估每条思考轨迹而不是盲从多数 3) 当所有 thinker 都错时重新推导 4) 保持语言和格式一致 |
| **Output Constraints** | 最终输出只给答案、不暴露 meta-analysis，按域规范出格式（数学用 \boxed{}、代码用 code block） |

**关键的地方在于"控制权的迁移"**。Workflow 模式下，是外部 Python 在调度；Skill 模式下，是 orchestrator 自己读 skill 文档然后自驱地调度。后者的好处是跨 harness 可移植——同一份 skill 文件可以在 Claude Code 和自定义 harness 里都跑得起来，不需要改框架代码。

我的判断：**Skill 化这个抽象比方法本身更值钱**。因为它把"重思考"从某个特定系统的产物，变成一个能跨系统迁移的、模型可读的能力片段。如果未来主流 agent 框架都支持 skill 装载，那这种"原子能力封装"会成为新的复用单位。

---

## 三、实验：数据说话

实验跨度其实蛮大的——闭源模型用了 GPT-5 Thinking、Claude 4.5 Thinking、Gemini-3 Pro Preview；开源模型用了 R1-Distill-Qwen 系列、Qwen3 系列、DeepSeek R1-0528、GPT-OSS-20B、Kimi K2 Thinking、GLM4.6、DeepSeek V3.2 Thinking 一长串。基本能覆盖到 2026 年初的主流 reasoning 模型。

主实验设置：温度 1.0，top_p 0.95，top_k 10，K 取 8 或 16，$K^{(1)} = 4$（生成 4 份 deliberation 输出取平均），迭代数 $N=1$（也就是不迭代）。

评估指标值得展开一下：

- **M@K（Mean@K）**：K 条并行轨迹的平均准确率，反映"平均水平"
- **P@K（Pass@K）**：K 条里至少一条对的比例，反映"理论上限"
- **V@K（Vote@K）**：多数投票，传统的 Best-of-N baseline
- **HM@K（Heavy-Mean@K）**：跑完两阶段后 deliberation 输出的平均准确率
- **HP@K（Heavy-Pass@K）**：deliberation 输出里至少一条对的比例

### 3.1 STEM 任务：层级关系非常清晰

作者跑了 AIME25、BeyondAIME、HMMT25-Feb、GPQA-Diamond 四个 benchmark，得到一个非常稳定的层级：

$$\text{HP@4} \geq \text{HM@4} \geq \text{V@K} \geq \text{M@K}$$

Table 1 节选关键数据（K=8 配置）：

| 模型 | AIME25 M@K | AIME25 V@K | AIME25 HM@4 | AIME25 HP@4 | AIME25 P@K |
|------|-----------|------------|-------------|-------------|------------|
| GPT-5 Thinking | 92.5 | 96.7 | 96.7 | 96.7 | 100 |
| Claude 4.5 Thinking | 82.5 | 90.0 | 90.0 | 90.0 | 90.0 |
| Gemini-3 Pro Preview | 95.0 | 96.7 | 95.8 | 96.7 | 96.7 |
| R1-Distill-Qwen-7B | 42.1 | 50.0 | 50.0 | 56.7 | 66.7 |
| R1-Distill-Qwen-32B | 53.3 | 63.3 | 63.3 | 66.7 | 76.7 |
| R1-Distill-Qwen3-8B | 76.7 | 83.3 | 85.8 | 90.0 | 90.0 |
| Qwen3-8B | 69.6 | 76.7 | 80.0 | 80.0 | 86.7 |
| Qwen3-32B | 72.5 | 83.3 | 80.8 | 83.3 | 83.3 |

几个我觉得真的能打的点：

**强模型直接逼近 Pass@K**：GPT-5 Thinking 在 AIME25 上 HP@4 直接打到 96.7%，跟 P@K=100 只差 3.3 个点，几乎没什么空间了。这个现象在更难的 BeyondAIME 上更明显——GPT-5 Thinking 的 M@K 只有 69.9%，但 HM@4 拉到 79.5%，HP@4 上 83.0%，比 P@K=86.0 差不到 3 个点。

**弱模型也能涨**：R1-Distill-Qwen-7B 这种 7B 小模型，在 GPQA-Diamond 上 M@K 才 48.8%，HM@4 拉到 51.5%，HP@4 涨到 63.1%，跟 P@K=85.9 还有较大空间。也就是说，小模型的瓶颈不在 deliberation 的综合能力，而在 parallel reasoning 阶段的轨迹质量。

**等等，AIME25 上 R1-Distill-Qwen-7B 的 HM@4 跟 V@K 一样都是 50.0？** 我盯着看了好一会儿。作者解释是"天花板效应"——当模型本身已经在某个 benchmark 上接近其能力极限，deliberation 的边际收益就被压缩了。我倾向于接受这个解释，但更想看的其实是在 BeyondAIME 上的同等比较——果然在那里 HM@4=30.8% vs V@K=32.0%，差距更明显地体现出来。

### 3.2 deliberation 真的在做"跨轨迹综合"吗？

这是论文里我觉得最值得看的一个分析——Figure 2 把不同并行 pass rate 区间的 query 分桶，看 deliberation 后 pass rate 怎么变化。

![图2：不同并行 pass rate 区间下，heavy thinking 的修正能力分布](https://arxiv.org/html/2605.02396v1/x2.png)

*图2：横轴是并行推理阶段的 pass rate 分桶（0.125、0.375、0.625、0.875），纵轴是经过 deliberation 后的 heavy pass rate 分布。最关键的观察是：在 pass rate < 0.5 的"高难度区间"里（也就是 K 条并行轨迹里超过一半都答错的 query），仍然有大约 500 个 query 被 deliberation 救回来了。*

具体数据：用 R1-Distill-Qwen-7B、从 Skywork OR1/DAPO/DeepScaler 采样 10k query、K=16 并行采样后做 deliberation。在 pass rate < 0.5 的 query 里，约 1400 个仍然没救回来，但 500 多个被纠正了。

**这就是 deliberation 不等价于 voting 的关键证据**。voting 只能从"多数派"里选，遇到 pass rate < 0.5 的 query 就废了；deliberation 模型能从少数派的正确思路里识别出来。

但同时作者也老实承认：pass rate > 0.5 的 query 里，大约有 30 个反而被 deliberation 搞错了。这部分性能损失虽然小，但说明 deliberation 不是单调正向的——它有自己的失败模式。

### 3.3 一个反直觉发现：deliberation 模型不需要是最强的 reasoner

![图3：固定 R1-Distill-Qwen-7B 做 parallel reasoning，换不同模型做 sequential deliberation](https://arxiv.org/html/2605.02396v1/x3.png)

*图3：用 R1-Distill-Qwen-7B、R1-Distill-Qwen3-8B 和 Qwen2.5-32B-Instruct 三个模型分别做 deliberation 的对比。注意 Qwen2.5-32B-Instruct 单独跑 AIME25 只能拿 12.8%——比 7B 的 reasoning 模型还差——但它在 deliberation 角色上表现却很可观。*

这个发现挺反直觉的：**deliberation 阶段对模型的要求，跟 reasoning 阶段不一样**。reasoning 需要尖锐的推理能力，deliberation 需要的是 comprehensive 的分析、综合、总结能力。Qwen2.5-32B-Instruct 这种"通用对话模型"虽然不会做 AIME 题，但能把别人做的 trace 看懂、综合出对的答案。

工程含义直接：**生产环境里完全可以让一个便宜的大 instruct 模型做 deliberation，让贵的 reasoning 模型专门跑 parallel 这一段**。两阶段解耦可能是个 cost-effective 的部署模式。

### 3.4 通用任务：tasks-dependent，不是万能药

Table 2 是通用任务上的结果，比 STEM 更复杂一些：

| 模型 | LiveCodeBench M@K → HM@4 | Arena-Hard M@K → HM@4 | IFEval M@K → HM@4 | IMO M@K → HM@4 |
|------|---|---|---|---|
| Qwen3-8B | 55.5 → 56.8 | 26.0 → 25.0 | 85.4 → 85.2 | 50.2 → 50.3 |
| R1-Distill-Qwen3-8B | 56.3 → 56.8 | 20.8 → 18.7 | 35.7 → 69.3 | 47.0 → 47.2 |
| GLM 4.6 | 81.0 → 81.3 | 88.2 → 88.1 | 88.8 → 88.5 | 74.5 → 75.1 |
| Kimi K2 Thinking | 81.2 → 83.7 | 83.5 → 83.1 | 92.5 → 92.0 | 69.1 → 77.2 |
| GPT-OSS-20B | 69.7 → 69.2 | 25.4 → 25.0 | 90.8 → 91.1 | 65.8 → 71.0 |

几个观察值得拎出来：

**LiveCodeBench / IFEval 这种"判定标准客观"的任务上涨得很猛**。R1-Distill-Qwen-32B 在 IFEval 上 M@K 35.7% → HM@4 69.3%（这个数字在 Table 2 的扩展数据里），几乎翻倍。代码和指令遵循这类有客观判定标准的任务，deliberation 的综合能力能稳定收割收益。

**Arena-Hard 这种"主观偏好"任务上反而可能掉点**。Qwen3-8B 在 Arena-Hard 上 M@K 26.0% → HM@4 25.0%，slightly negative。作者的解释是：deliberation 倾向于"找正确答案"，但 Arena-Hard 评的是"风格偏好"，"平均"出来的回答可能跟 judge 的特定风格偏好不匹配。

**IMO 上 HP@4 经常超过 P@K**。比如 GLM 4.6 在 IMO 上 HP@4=86.0% 超过了 P@K=75.1%——这意味着 deliberation 不只是"从已有答案里挑"，而是真的在做"跨轨迹的再推理"，能合成出原本 K 条里都没有的正确答案。这是论文非常想强调的点，也是 RLVR 的切入空间。

---

## 四、迭代审议、工具调用、RL 训练

### 4.1 迭代审议的双刃剑

![图4：不同迭代轮次下 HM@K 和 HP@K 的变化](https://arxiv.org/html/2605.02396v1/x4.png)

*图4：在 R1-Distill-Qwen-7B、R1-Distill-Qwen3-8B、DeepSeek-R1-0528 三个模型上跑 1-4 轮迭代审议。HM@K（平均准确率）随轮次单调上涨，但 HP@K（理论上限）反而下降。*

这个现象作者解释得比较克制：**迭代越多，前一轮的输出污染会累积**，模型的"再思考空间"被前一轮的判断框死了，导致虽然平均质量在涨，但能突破到的最高点反而被压缩。

这块其实让我想到一个工程问题——在生产环境，**到底要不要迭代？** 我的判断是：

- 如果你的目标是"提高输出稳定性 / 平均质量"，比如对话产品的用户体验，迭代 2-3 轮是值得的
- 如果你的目标是"挖掘模型的最强能力"（比如做 hard problem 攻坚，或者作为 RL 训练数据生成器），那应该让 HP@K 保持高位，迭代反而是有害的

### 4.2 工具调用场景同样适用

Table 3：在 AIME25/HMMT25 上，让模型在 parallel reasoning 阶段调用 Python 解释器，最多 50 轮交互。

| 模型 | AIME25 V@4 | AIME25 HM@4 | HMMT25 V@4 | HMMT25 HM@4 |
|------|------------|-------------|-------------|--------------|
| Qwen3-8B | 68.3 | 76.7 | 54.1 | 69.3 |
| Qwen3-32B | 83.3 | 80.0 | 63.3 | 68.5 |
| GPT-OSS-20B | 83.3 | 90.0 | 73.3 | 85.7 |

GPT-OSS-20B 在 AIME25 上 HM@4=90.0% 大幅超过 V@4=83.3%。这说明 deliberation 不只是处理纯文本推理，对带有工具执行反馈的轨迹也能有效综合——这其实是一个让 HeavySkill 跟 agent 场景能对接的重要证据。

### 4.3 RLVR 加 heavy mode：值得做但不稳定

![图5：用 VeRL + GSPO 在 R1-Distill-Qwen-7B 上做 heavy mode 的 RL 训练](https://arxiv.org/html/2605.02396v1/x6.png)

*图5（论文 Figure 6）：蓝色曲线是 K=8 配置，绿色是 K=16。前 100 步两组都在涨，HM@4 涨了约 10%。但 100 步后 K=16 出现明显的熵塌缩（entropy collapse），训练崩了；K=8 配置则一路保持稳定。作者认为是因为 R1-Distill-Qwen-7B 的最大序列长度限制——K=16 时序列化的 cache 太长，超出模型有效处理范围。*

这个 RL 实验我个人觉得还在探路阶段——结论是"RLVR 在 heavy mode 上确实能涨点"，但 K=16 直接训崩这件事说明，**heavy mode 的 RL 训练对序列长度、稳定性的要求比单 trajectory RL 要苛刻得多**。如果想真的把"宽度"和"深度"同时通过 RL 拉起来，可能得在 base model 的 context window、reward shaping、KL 控制上做更多工作。

### 4.4 轨迹挑选：Diversity 没那么神

![图6：四种轨迹选择策略——Random、Max-Diversity、Max-Length、Max-Answer-Num](https://arxiv.org/html/2605.02396v1/x5.png)

*图6（论文 Figure 5）：用 R1-Distill-Qwen3-8B 在 AIME25 和 HMMT25-Feb 上，每个 query 生成 256 条轨迹，从中选 K 条进入 deliberation。结果是 Max-Answer-Num（高频答案优先）最好，Random 和 Max-Diversity 差不多，Max-Length（长答案优先）最差。*

附录里这个实验我觉得很有信息量。它告诉我们三件事：

1. **加大 K 始终有收益**，不论怎么选
2. **explicit diversity 没什么用**——温度采样自带的 diversity 已经够了，专门去优化 diversity 提升甚微
3. **长度不等于质量**——啰嗦的答案不仅没用，反而引入噪声

第二点其实跟 self-consistency 的老观察是吻合的，但作者把它放在 deliberation 框架下重新验证一遍，是有意义的。

---

## 五、我的判断：这篇论文的真实位置

先把好话说完：

**第一个值钱的地方是 abstraction**。把 Kimi K2、PaCoRe、LongCat-Flash-Thinking 这些工业界框架的共同模式抽出来叫"heavy thinking"，再把它从某个 harness 的产物降维成模型自己的 inner skill，这个视角其实挺干净的。它给了一种思考 agent 设计的新方式——不是"加更多模块"，而是"识别哪些是模型本来就有/应该有的能力"。

**第二个值钱的地方是规模化的实证**。11 个模型、4 个 STEM benchmark、4 个通用 benchmark、tool use、RL 训练、迭代消融、模型解耦消融、轨迹选择消融——这套实证规模足够撑得起结论。不是那种只在一两个模型上跑跑的论文。

**第三个值钱的是 deliberation ≠ voting 的实证**。Figure 2 那个 pass rate 分桶分析，是这篇论文里我觉得最干净的一个实验设计——它直接回答了"deliberation 到底有没有跨轨迹综合能力"这个根本问题，结论是肯定的（虽然不强）。

接下来是我觉得需要打问号的地方。

**"首次" 这个说法值得 push back**。论文摘要里说 "We are the first to conduct the comprehensive empirical study"——但 Kimi K2 的技术报告、PaCoRe 的论文、LongCat 自家的报告其实都在做类似的实证。HeavySkill 的贡献更准确地说是"系统化、跨多模型的对比研究"，而不是"首次"。说实话看到"first to"这种字眼我都会习惯性地警觉一下，这篇也不例外。

**Skill 化这一抽象，新颖性其实有限**。Claude Code 的 skills library 本身就已经在做这件事——把能力封装成可读文档，由 orchestrator 动态加载。HeavySkill 是在这个机制上加了一个具体的 skill 内容，理论贡献并不大。论文把"skill 化"放在标题里，多少有点"借力"Claude 那套术语的味道。

**RL 实验过于初步**。K=16 直接训崩、只在 R1-Distill-Qwen-7B 一个 backbone 上跑、只跑 100-300 步——这部分如果作为 future work 提一下还好，作为正文实验来呈现，说服力不够。

**Arena-Hard 上掉点这件事被一笔带过**。论文承认了在主观偏好任务上 deliberation 不一定有用，但没有进一步分析——这其实是 heavy thinking 一个挺严肃的局限。如果一个 agent 同时要处理"找正确答案"和"产生符合用户偏好的回答"这两类任务，heavy thinking 应该用什么策略来调度？这个问题没回答。

---

## 六、对工程实践的几点启发

如果你也在做 agent 系统或者 LLM 推理产品，我觉得有几条能落地：

1. **先把 harness 简化到极致再加东西**。如果你的系统拆完之后核心就是"并行采样 + 综合"，那大部分 orchestration 的复杂性可以砍掉。先把简单版跑通，再决定要不要加 memory、要不要加 skill library。

2. **deliberation 模型可以跟 reasoning 模型解耦**。生产环境里，让一个便宜的大 instruct 模型做综合可能比让 reasoning 模型自己 deliberation 更划算。Qwen2.5-32B-Instruct 那个反直觉发现值得复现一下。

3. **memory cache 要做轨迹剪枝和 shuffle**。直接拼 K 条完整 thinking content 几乎一定会超长，而且位置敏感性会让结果不稳。剪枝 + 打乱位置是低成本但关键的工程动作。

4. **不要迷信 diversity 显式优化**。温度采样的自带 diversity 已经够用，去做 explicit max-diversity selection 提升很小。把这块复杂度省下来。

5. **对客观判定任务用 heavy thinking，对主观偏好任务慎用**。代码、数学、指令遵循这类任务上 deliberation 收益大；对话风格、开放问答这类任务收益不稳，甚至可能掉点。

6. **K 选 8 而不是 16，除非你有理由**。RL 训练阶段 K=16 直接训崩；推理阶段虽然 K=16 有点收益，但成本翻倍。绝大多数场景 K=8 是 sweet spot。

---

## 七、收尾：harness 是壳，能力在模型里

回到一开始那个问题——agent harness 这层壳里，真正干活的是什么？

这篇 HeavySkill 给的答案我是认同的：**是模型自己的"重思考"能力**。Harness 只是一层调度框架，真正能力封装在两件事里——能不能产出 K 条有效的并行轨迹，能不能在审议阶段做跨轨迹的综合。这两件事都是模型层面的内功，不是系统层面的把戏。

延伸一下，这其实给了一个很重要的方向判断：**未来一段时间，提升 agent 能力的边际收益，会越来越多地落在"内化技能 + 模型 RL"，而不是"加更多 orchestration 模块"上**。Claude Code、CodeX 这些 harness 已经给出了 skill 这个抽象的雏形，接下来的赛跑可能是——谁能把更多原本靠外部 orchestration 实现的复杂能力，直接训练到模型参数里去。

如果这个判断对，那 HeavySkill 这篇文章的位置就很清楚了：它不是一个划时代的方法论突破，而是这条主线上一个重要的"概念整合"工作——把分散在多个工业界框架里的"重思考"模式拎出来，做系统化的实证，并把它 skill 化便于复用。对做 agent 工程的人来说，值得花时间细读一遍。

最后多说一句：作者把 prompt 模板和完整 skill 文件放在 Appendix C 里了（Figure 7-10），如果你想直接拿来用，可以去 [GitHub 仓库](https://github.com/wjn1996/HeavySkill) 看一下完整的 skill 实现。比起读论文，照着 skill 文档动手跑一次，可能体感会更准。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我*
