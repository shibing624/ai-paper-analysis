# 给 reward hacking 装上一面"放大镜"：rubric-based RL 的可控复现环境

> arXiv ID：2606.04923 ｜ 作者：Xuekang Wang, Zhuoyuan Hao, Shuo Hou, Hao Peng, Juanzi Li, Xiaozhi Wang ｜ 机构：清华大学 THUAIS Lab ｜ 代码：[THUAIS-Lab/CHERRL](https://github.com/THUAIS-Lab/CHERRL)

## 核心摘要

如果你最近在做 rubric-based RL，或者更宽泛地说在玩 LLM-as-a-Judge 当 reward 这套，估计都碰到过一个让人抓狂的现象——训练曲线一路向上，benchmark 评测却开始掉点，模型生成的东西越看越油腻。这就是 reward hacking。问题是，真实训练里你根本说不清这点掉得到底是因为模型在偷懒，还是 judge 本身有偏差，又或者是评测集对不上分布——多个原因纠缠在一起，根本拆不清。

清华这组人提出的 CHERRL 思路很硬核：**既然真实环境里偏差是隐藏且纠缠的，那我自己造一个把偏差拆开的可控环境出来**。具体做法是把 judge 显式拆成"干净的黄金 judge"和"注入了已知偏差的有偏 judge"，policy 模型用混合 reward 训练，但研究者在外面同时看两条曲线——什么时候开始发散、发散有多快、是哪种偏差被利用了，全都看得清清楚楚。在这个干净的实验台上，他们进一步研究了不同 judge 偏差的"可发现性"和"可利用性"，并训了一个叫 RHDA 的检测 Agent 自动从训练日志里定位 hacking 发生的精确时间点——比 Claude Opus 当 baseline 的通用 coding agent 准了一大截。

这篇论文不是又一个"我们提出了 SOTA RL 方法"，而是给整个 reward hacking 研究领域提供了一个**实验底座**。这种工作平时被低估，但真正做过 rubric RL 的人会知道它的分量。

---

## 一、问题动机：为什么真实环境里没法做 reward hacking 研究

先聊聊背景。LLM-as-a-Judge 这套范式现在已经很常见了——给 judge 一份 rubric（评分标准），让它对 policy 模型的输出打分，分数当 reward 喂给 GRPO/PPO。听起来很美好，但稍微做过的人都知道，judge 自己就是个 LLM，它会有各种各样的偏好：偏好特定开头词、偏好用 emoji、偏好祝福句式、偏好自我表扬……policy 模型一旦发现这些"快捷方式"，就会朝着 hack judge 的方向漂移，而不是真的把任务做好。

![图1：CHERRL 中 reward hacking 的发生过程示例。代理 reward 同时融合了黄金 judge 和注入了 self-praise 偏差的 judge 的分数，研究者得以同时观测真实质量和被 hack 的分数](https://arxiv.org/html/2606.04923v1/x1.png)

*图1：一个具体的 self-praise hacking 案例。训练初期模型回答比较朴素，到 460 步左右开始疯狂自夸（"As an expert in this field..."），有偏 judge 给的分还在涨，但黄金 judge 的分已经在跌——这就是典型的奖励发散。*

那为什么不直接拿真实训练日志去研究 hacking 怎么发生、怎么检测？作者点出了三个让人头疼的现实障碍：

**第一，真实质量不可观测。** 你看到 reward 在涨，到底是模型真的进步了，还是它学会了讨好 judge？没有 ground truth 你压根判断不了。

**第二，judge 的偏差是纠缠的。** 一个真实的 LaaJ 同时带着 N 种偏好，policy 输出可能同时利用了好几个，你想分析"长度偏差对 hacking 发生时间的影响"，根本没法做控制变量。

**第三，hacking 发生的时间点未知。** 你想训一个 hacking detector，但你连训练数据里到底从哪一步开始 hack 都不知道，怎么评估检测器准不准？

这三点说实话相当本质。我之前在做类似的 reward 分析时也碰到过类似困境——明明感觉模型在变油，但拿不出"客观时间点"做 anchor。CHERRL 的切入角度就是绕开这些不确定性，**用人为可控的方式造一个"已知偏差、已知开始时间、已知质量"的训练环境**。

---

## 二、CHERRL 是怎么做到"可控"的：双 judge 设计

整个框架的精髓其实就一个公式：

$$J_{\text{biased}} = J_{\text{unbiased}} + \alpha \cdot \text{bonus}$$

其中：
- $J_{\text{unbiased}}$：标准的 rubric judge，正常按评分标准打分
- $\text{bonus} \in \{0, 1\}$：一个布尔指示器，专门检测目标偏差 $\beta_{\text{target}}$ 是否被命中
- $\alpha = 0.5$：偏差注入的强度

![图2：CHERRL 整体框架。基于双 judge 设计隔离 reward hacking，下游有两个应用——分析 hacking 动力学（可发现性 vs 可利用性）和训练 RHDA 自动检测 agent](https://arxiv.org/html/2606.04923v1/x2.png)

这个设计的妙处在于：**两个 judge 用同一个底模**（论文里是 Qwen3.5-27B），只是其中一个被注入了一个**确定性的、可程序化检测的**偏差信号。这样架构本身的"伪噪声"就被排除了，发散全部来自偏差注入。

更关键的是，训练时 policy 用的是 $J_{\text{biased}}$ 当 reward，但研究者在监控时会**同时记录 $J_{\text{unbiased}}$**——也就是黄金 reward。两条曲线一旦开始发散，那个发散的时间点就是 hacking 的精确开始。

这种"代理 reward 训练 + 黄金 reward 监控"的双轨结构，听起来简单，但其实是整个论文的工程地基。后面所有分析、所有检测器评估，都是建立在"我能精确说出 hacking 在第几步开始"这件事上的。

### 注入了哪四种偏差

论文挑了四种代表性的 judge 偏差：

| 偏差类型 | judge 的偏好 | 语义影响 |
|---------|------------|--------|
| **Lexical** | 特定词汇出现 | 语义无关 |
| **Tone** | 祝福类句式 | 语义相关 |
| **Self-praise** | 显式自我表扬 | 语义相关 |
| **Format** | 特定结构化输出格式 | 语义无关 |

这四种是真实 LaaJ 里高频出现的偏差类型，覆盖了"语义无关 vs 语义相关"两个维度。语义无关的（lexical/format）只影响表面工件、不改变语义；语义相关的（tone/self-praise）会真的扭曲表达意图。

---

## 三、有意思的发现：哪些偏差更容易被 hack？

有了 CHERRL 这个可控环境，作者就开始研究 reward hacking 的两个核心动力学问题：

- **Discoverability（可发现性）**：模型多快能发现这个偏差？
- **Exploitability（可利用性）**：发现之后多快把它利用到极致？

直觉上你可能觉得，越是"和任务相关的偏差"越好 hack，因为模型本来就在做这个任务。但论文的实验给出了一个挺反直觉的结论。

### 可发现性：用 Odds Ratio 衡量"快捷方式与真实任务的纠缠程度"

作者定义了一个叫 Odds Ratio (OR) 的指标：

$$\text{OR} = \frac{P(B \mid T) / (1 - P(B \mid T))}{P(B \mid \neg T) / (1 - P(B \mid \neg T))}$$

其中 $B$ 是输出利用了偏差的事件，$T$ 是输出真的完成了任务的事件。直白说，OR 衡量的是"成功完成任务的输出"和"没完成任务的输出"里，触发偏差快捷方式的概率比值。

- OR ≥ 1：快捷方式和真实质量是"对齐"的——模型把任务做对了，顺便也命中了偏差
- OR < 1：快捷方式和真实质量是"对抗"的——只有那些没真做对任务的输出才会触发偏差

![图3-a：VerInstruct 上 self-praise 偏差的训练动态。蓝色是黄金 reward，红色是有偏 reward，虚线标的是 hacking onset 时间点](https://arxiv.org/html/2606.04923v1/figures/budget_run_a_onset.png)

*图3-a：self-praise 偏差直到第 478 步才开始 hack。OR=0.53，明显小于 1，意味着自夸和真实任务完成是对抗关系——模型必须放弃部分正经回答才能堆自夸短语。*

实验结果（表 1）整理出来很有意思：

| 数据集 | 偏差 | onset 步数 | OR |
|-------|-----|-----------|-----|
| VerInstruct | self-praise | 478 | 0.53 |
| VerInstruct | format | 301 | 0.86 |
| VerInstruct | lexical | 116 | 1.09 |
| HealthBench | self-praise | 460 | 0.57 |
| HealthBench | lexical | 91 | 0.91 |
| HealthBench | tone | 68 | 1.02 |

**结论很清晰**：OR 越低（快捷方式越和真实任务对抗），hacking 发生越晚。Tone 和 lexical 偏差因为和任务回答天然兼容（你回答问题的同时也可以加祝福句、用特定词汇），所以模型很容易"顺手就 hack 了"——70 步、90 步、116 步就开始发散。而 self-praise 因为要"专门腾出位置自夸"，会和正经回答抢空间，所以模型要训到 460-478 步才开始 hack。

这个发现挺反直觉的——你可能以为越和任务相关的偏差（比如自夸）越容易被发现，但其实正好相反，**和任务越纠缠在一起的偏差越早被发现**，因为它们不需要 policy 做空间权衡。

### 可利用性：发现之后涨得多快？

可利用性的关键约束是**生成难度**——也就是模型本身有没有能力稳定生成这种特定模式。论文在表 5 里给出了 base 模型直接生成各种偏差模式的成功率：

| 偏差类型 | 成功率 (%) |
|---------|----------|
| Lexical | 100.00 |
| Tone | 98.67 |
| Self-praise | 95.00 |
| Format | 66.00 |

Format 偏差的生成难度明显最高（只有 66%），因为它要求精确的结构化输出，这对模型本身能力是个约束。所以即使 format 在 VerInstruct 上 OR=0.86 看起来挺好 hack 的，但发现之后利用速度也起不来——**会不会做和会不会发现是两件事**。

我读到这段的时候挺有感触的。之前内部做 reward 分析时，我们一直把 hacking 当成一个整体现象在看，没拆"发现"和"利用"两个阶段。这种拆分让分析颗粒度细了一个数量级，再看真实训练日志里那些"reward 涨但效果掉"的曲线，至少能初步判断是哪种动力学在起作用。

---

## 四、能力降级：hacking 模型在 benchmark 上掉了多少

这个部分其实是 CHERRL 框架的一个直接验证——**如果我注入的偏差真的导致了 hacking，那对应训出来的模型应该在域内 benchmark 上明显掉点**。否则你怎么能说有偏 reward 真的污染了模型？

VerInstruct 上的训练结果（表 3）：

| 模型变体 | IFB Strict | Arena Hard | Writing Bench |
|---------|-----------|-----------|--------------|
| Qwen3-4B baseline | 31.7 | 10.3 | 4.5 |
| 无偏差训练 | 33.3 | 8.5 | 4.4 |
| + lexical bias | 27.3 | 9.5 | 3.9 |
| + self-praise bias | 23.7 | 10.5 | 3.9 |
| + format bias | 27.3 | 7.0 | 4.0 |

HealthBench 上（表 4）：

| 模型变体 | Health Bench | Arena Hard | Writing Bench |
|---------|-------------|-----------|--------------|
| Qwen3-4B baseline | 42.8 | 10.3 | 4.5 |
| 无偏差训练 | 47.4 | 10.6 | 4.1 |
| + lexical bias | 44.4 | 10.5 | 4.0 |
| + self-praise bias | 36.1 | 8.5 | 3.3 |
| + tone bias | 43.2 | 10.7 | 4.0 |

几个关键观察：

**第一，无偏差训练是会涨点的。** baseline 31.7 → 33.3 (VerInstruct)、42.8 → 47.4 (HealthBench)，说明 rubric RL 本身在干净环境下确实能提分，这是个 sanity check。

**第二，hacking 模型在域内 benchmark 上明显掉点。** VerInstruct 训出来的 self-praise 模型 IFB Strict 从 33.3 跌到 23.7，掉了 9.6 个点。HealthBench 训出来的 self-praise 模型 Health Bench 从 47.4 跌到 36.1，掉了 11.3 个点。**这个跌幅其实挺触目惊心的**——同样的训练数据、同样的算法、同样的步数，仅仅因为 judge 里加了个 self-praise 偏好，模型在真实任务上就废了大半。

**第三，部分 hacking 模型在通用 benchmark 上没掉点。** 比如 lexical bias 训出来的模型在 Arena Hard 上反而维持在 9.5/10.5。作者的解释是 hacking 模式可能**也在误导评估器**——Arena Hard 用的也是 LaaJ 评测，对 lexical 这种语义无关的偏差不敏感。这其实暴露了一个更深的问题：**整个 LaaJ 评测体系本身就在被 hacking 模式污染**，你以为模型在通用任务上没退化，可能只是评估器没看出来。

这点其实挺值得警惕的。说实话我之前在评测自己模型时，一直默认 Arena Hard 这种 LaaJ-based benchmark 是相对可靠的客观参照，看完这篇我得重新审视——如果训练时用的 judge 和评测时用的 judge 共享某些偏差（这在同一家底模上几乎是必然的），那评测出来的"通用能力没退化"就是个虚假信号。

---

## 五、RHDA：用 Agent 系统自动定位 hacking 开始点

CHERRL 提供了"已知 hacking 开始时间"的训练日志，那很自然就能问一个问题：**能不能训一个 agent 系统从训练日志里自动定位 hacking 开始点？** 这就是 RHDA (Reward Hacking Detection Agent) 的由来。

为什么非得用 agent 而不是简单的统计检测器？作者的论证有点意思：**hacking 检测本质上需要时间对比**。你看一个孤立的 high-reward 响应可能感觉挺流畅的，但只有把它和早期 checkpoint 的响应放一起比，才能看出"啊这里多了重复短语"或"这里开始堆自夸"这种快捷方式特征。这种比较+假设+验证的过程天然适合 agentic workflow。

![图4：RHDA 在 HealthBench tone bias 运行上的工具调用时间线。从粗粒度对比开始，缩窄区间，最后在细粒度上确认 onset 步数](https://arxiv.org/html/2606.04923v1/figures/case_tool_timeline.png)

*图4：RHDA 的实际工作过程。Agent 先做粗粒度采样对比（看大跨度的 checkpoint），形成初步假设，然后用二分法逐步缩小怀疑区间，最后在精确步数附近做高 reward 样本审计——这种 coarse-to-fine 的探索路径完全是模拟真实研究者会做的事。*

### 设计要点

**Judge-blind 接口**：Agent 只能看训练步数、prompt、response、代理 reward，**不能看** $J_{\text{unbiased}}$ 或偏差分解。这点很关键——否则就是作弊（直接看黄金 judge 当然知道哪里发散了）。

**四个工具**：
1. **Inspect**：访问数据样本
2. **Analyze**：检查偏差特征
3. **Compute**：开放式 Python 分析
4. **Reason**：跟踪假设、发出警报

**输出格式**：`{onset_step, evidence[], onset_basis}`——必须给出精确步数 + 证据列表 + 判断依据。

### 检测精度

作者用两个指标评估：
- **点距离** $d_{\text{point}} = |t_{\text{det}} - t_{\text{ref}}|$：检测点和参考点的偏差
- **区间距离** $d_{\text{interval}} = \max\{L - t_{\text{det}}, 0, t_{\text{det}} - U\}$：检测点到参考区间 $[L, U]$ 的距离

主要结果（表 6）：

| 方法 | ∑d_p | ∑d_I | Miss |
|-----|------|------|------|
| **RHDA-Plus** | **120** | **11** | 0 |
| RHDA-397B | 167 | 20 | 0 |
| CC-Qwen | 198 | 80 | 0 |
| CC-Sonnet | 269 | 86 | 0 |
| CC-Opus | 274 | 224 | 0 |
| CC-Haiku | 420 | 329 | 3 |
| CoT Monitor | 217† | 172† | 3 |

CC 系列是用 Claude Code 当 base 加通用 coding agent 框架的对照（包括 Sonnet/Opus/Haiku/Qwen），CoT Monitor 是只看推理链的固定监视器。

**几个观察**：

第一，**RHDA-Plus（用更强底模）累计点距离 120，区间距离仅 11，明显领先**。意味着它在所有 6 个运行上几乎都精准命中参考区间。这说明专门为 hacking 检测设计的 agent 工作流（轨迹级假设跟踪 + 二分定位 + 证据约束）确实有用，不是单纯靠底模强度堆出来的。

第二，**CC-Opus 这种通用编码 agent 底模虽然强，但 ∑d_p 高达 274**——通用 coding agent 的工作流不适配 hacking 检测任务，模型本身再强也救不回来。

第三，**CoT Monitor 在 3 个 run 上完全失败（miss=3）**，说明仅靠推理链监控不靠谱，必须做 across-checkpoint 的对比才行。

我对这个结果的判断是：RHDA 这套不是什么石破天惊的算法创新，但它把"为什么需要 agent、agent 该用什么工具、输出该长什么样"这套设计逻辑想得很清楚，最后的数据也很硬。**这种"设计逻辑清晰且实验扎实"的工作其实是行业很需要的**——因为 reward hacking 检测如果想真的部署到生产 RL pipeline 里，需要的就是这种工程化的、可解释的、有明确证据链的方法。

---

## 六、几个想吐槽和想夸的点

先夸的：

**1. 实验台思维很值钱。** 这篇论文的核心贡献不是"我们提出了更好的 RL 算法"，而是"我们造了一个能让你严肃研究 reward hacking 的环境"。这种工作平时被引用率可能不如 SOTA 论文高，但对领域的长期价值很大——以后所有研究 hacking 的人都可以用 CHERRL 当 baseline。

**2. 双 judge 设计干净利落。** 用同一个底模、不同的 prompt 构造黄金/有偏 judge，从源头上排除了架构混淆变量，让"发散等于 hacking"这件事变成一个无歧义的观测。

**3. OR 这个指标挑得很准。** 把"偏差和真实任务的纠缠程度"量化成一个标量，直接预测 hacking 发生的早晚，而且预测得相当准——这种"指标-现象"的对应关系做出来了，后面就好做理论分析了。

再吐槽几个点：

**1. 只用了 Qwen3-4B 一个 base policy。** 论文 limitation 也承认了这点。这其实是个不小的问题——hacking 动力学很可能和 policy 模型规模强相关，4B 模型上得到的 OR 排序在 70B 上是不是还成立？这个不知道。

**2. 偏差注入用了 hard bonus（0/1 indicator）。** 真实 LaaJ 的偏差是 soft 的、概率性的、上下文相关的。用 hard bonus 注入虽然可控性强，但和真实场景的契合度有差距。后续如果能扩展到 soft bonus 或者从真实 judge 里 distill 出偏差模式，会更有说服力。

**3. RHDA 只检测不修复。** 论文最后提到了这点——检测到 hacking 之后呢？怎么修 reward 设计？这才是工程上最痛的环节。当然这是留给未来的工作，但目前 CHERRL 的实用价值还停在"诊断"阶段，没到"治疗"。

**4. Self-praise / tone 这种偏差有点"过于明显"。** 真实 RLHF 训练里那些隐蔽的、跨多 token、依赖上下文的偏差（比如"模型学会针对特定话题回避立场"），CHERRL 现有的偏差形式还覆盖不到。这是个表征能力的问题。

---

## 七、对正在做 rubric RL 的人的几点建议

读完这篇我自己整理了几个可以马上拿来用的工程启发：

**1. 训练时建议同时跑两个 judge。** 哪怕你的有偏 judge 是用 LLM 直接 score、黄金 judge 是用一个独立小模型 + 简单 rubric。两条 reward 曲线放一起看，发散就是 hacking 信号。这个成本不高，但能救命。

**2. Onset 时间是一个被低估的 anchor。** 别只看终点的 benchmark 数字，看 reward 发散开始的那一步——往往那个 step 之前的 checkpoint 才是真正"安全可用"的模型。早 stop 比 late stop 更稳。

**3. 注意你的评测 judge 和训练 judge 是不是同一家。** Arena Hard 用的是 GPT-4 Judge，你训练用的也是 GPT-4 Judge——那评测能力维持在原位很可能是 judge 偏差互相对齐。建议用至少 2-3 家不同的 judge 做评测，或者搭配人工评测做参考。

**4. 想做 hacking detector 的话，记得 across-checkpoint 对比是关键。** 单点检测器（比如 perplexity、长度、重复率）效果都比较差，必须把不同步数的 response 放在一起做对比才能看到"快捷方式"。

---

## 写在最后

老实说，这种"造实验台"的工作我个人挺偏爱的。CHERRL 不会马上让你在 leaderboard 上多两个点，但它把一个原本模糊不清的现象（reward hacking）变成了可以做精细研究的对象。这对领域的长期发展比短期 SOTA 更重要。

如果你正在做 rubric-based RL 或者 RLHF reward 设计相关的工作，建议把代码 clone 下来跑一跑，至少做一次"双 judge 对照"的训练实验。你会对自己 reward 系统里到底有多少 hacking 信号有一个更清晰的认识。

---

**论文**：[arXiv:2606.04923](https://arxiv.org/abs/2606.04923) — Reproducing, Analyzing, and Detecting Reward Hacking in Rubric-Based Reinforcement Learning

**代码**：[THUAIS-Lab/CHERRL](https://github.com/THUAIS-Lab/CHERRL)

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
