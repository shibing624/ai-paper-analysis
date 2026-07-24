# 谁的智能体在给"自己"打分？当评估器也得跟着 Agent 一起进化

> 论文：[*Who Grades the Grader? Co-Evolving Evaluation Metrics and Skills for Self-Improving LLM Agents*](https://arxiv.org/abs/2607.12790)
> arXiv: [2607.12790](https://arxiv.org/abs/2607.12790) · 2026 年 7 月 14 日
> 作者：Xing Zhang, Guanghui Wang, Yanwei Cui, Ziyuan Li, Wei Qiu, Bing Zhu, Peiyang He

---

你有没有想过——**那些号称「自我进化」的 LLM Agent，其实每一步都在悄悄「作弊」**？

不是在说它会主动骗人，而是说它活在一个被精心伪造的世界里：你给它一个任务，它跑一遍，得分，决定要不要把这次的"经验"沉淀成技能，下次复用。

这听起来挺像 RLHF 的套路对吧。但问题是——**这个"得分"是从哪来的**？

在 MBPP+、SWE-bench 这种学术基准里，得分简单粗暴：跑单元测试，过就是过，不过就不过。但凡你做过稍微复杂一点的 Agent 部署就会发现，真实任务里**根本没有这种便宜的 ground truth**。一份报告写得好不好、一段 SQL 查得对不对、一个工作流走完没走完——这些事的"对"在哪？

Anthropic 的 managed agents 让你**手动给每个 grader 写 outcome rubric**，OpenAI 的 Record & Replay 让你**先亲自演示一遍**。绕了一大圈你会发现，今天的"自进化 Agent"其实是个**人肉外挂**。

如果人类的判断力就是天花板，那 Agent 还进化个啥？

arXiv:2607.12790 这篇叫 *Who Grades the Grader?* 的论文，把这个问题摆到台面上了。它做了一个挺反直觉的设计：评估器本身**也得跟着 Agent 一起进化**。你训练技能，我训练打分器，咱们互相牵制、互相校对，看最后谁跑偏。

结果呢？三个任务上，**没有任何 ground truth 打分器的纯协同进化**，保留了 88%-110% 的"用真实标签训练"才能拿到的提升。

这个数字如果是真的，那说明**只要你给 10 个示例做锚点**，你的 Agent 就能在没标准答案的领域自我提升，而且效果逼近"开卷考试"。

听上去太美了对吧？我也是带着怀疑读完的。下面展开聊。

---

## 核心摘要

**痛点**：自进化 Agent 系统的整个循环都建立在"已有可靠评估器"这个隐藏前提上。但在绝大多数真实部署场景里，这种评估器要么不存在，要么必须靠人写 rubric。评估器一旦失真或被绕过，整个自进化就变成"自欺欺人"。

**方案**：作者提出 **Double Ratchet**——评估指标和技能同时进化的协同框架。评估器不再是固定的 LLM judge，而是由**小缺陷检测器（drawback detectors）组合**而成，进化目标是与一个 10 项的"锚定参考集"一致，同时受无标签数据上多个检测器的共识正则化约束。技能循环则沿用 Ratchet 系统的 lifecycle 管理思路。

**结果**：在代码生成（MBPP+）、企业级 text-to-SQL（Spider 2.0-Snow）和无 ground truth 的报告生成三个任务上，Double Ratchet 保留了 88%-110% 的"ground truth 驱动"基线提升；消融实验表明，对评估器来说**锚定纪律比 lifecycle 更是安全底线**；安全性实验还展示了一个真实的 Goodhart 攻击被检测、定位、修复的完整闭环。

**我的判断**：这是一份**工程哲学层面的硬核论文**，不是某个 benchmark 刷榜的胜利。它真正回答的是"在没有 ground truth 的世界里，自进化 Agent 该怎么活下来"——而它的答案是**别相信任何单点评估器，把锚定和审计做成纪律**。方法本身偏重工程组装而非算法突破，**对工程团队的启发远大于对算法研究者的**。

---

## 问题动机：当"自进化"遇上"无 ground truth"

先说清楚 Ratchet 是个啥。简单讲，它是一篇处理"技能库漂移"问题的前期工作（arXiv:2605.22148）：当 LLM 自己写技能、自己加进技能库、自己调用，技能库会越堆越乱、越来越偏离有用方向。Ratchet 给出的解法是给技能库加 **active-cap（最多同时激活多少条）+ 退休阈值（连续多少次没贡献就下架）**，并证明这样能保证**期望通过率不会无限漂移**。

这个 lifecycle management 的思想其实就是这篇 Double Ratchet 的地基。但作者把视角又往前推了一步——

> "If skills can be evolved, why not the metric?"

技能能进化，凭啥评估器不能进化？

但这里面有个鸡生蛋蛋生鸡的问题：**评估器要衡量技能进步，技能进步又依赖评估器打分**。两边一起进化，怎么保证不会"一起腐化"？这是经典的 Goodhart 风险——当一个度量变成目标，它就不再是好的度量了。RLHF 里的 reward hacking、推荐系统里的点击率优化、学术圈的 h-index 操纵，全都是这个老问题的新变种。

Double Ratchet 的应对方式很工程师：**承认我们对「好」几乎一无所知，但通常能识别「坏」**。一句论文里的原话翻译过来就是：

> "We rarely know what good is, but given an output we can usually find drawbacks."

也就是——**别再纠结怎么定义"完美答案"了，把能检测到的缺陷一项一项列出来，组合起来就够了**。

这其实是个挺有勇气的转向。

---

## 方法核心：Double Ratchet 怎么"互相咬合"

整个架构我画出来给你看，下面这张图直接说清楚：

![Figure 1: Double Ratchet 架构图](https://arxiv.org/html/2607.12790v1/x1.png)

*图 1：Double Ratchet 架构。左侧 Metric Loop（慢）进化 drawback detectors 的组合表达式，右侧 Skill Loop（快）进化技能库，二者通过"evolved metric grades train tasks"耦合。底部 Anchors 固定、从不被训练；dev 给 metric 提供监督，test 仅用于报告，final judge 审计无 ground truth 任务。*

注意几个关键设计：

**1）Metric 是一棵逻辑树，不是单一 LLM judge**

论文用 `e ::= o | OR | AND | K-of-k` 这种语法，把多个**原子检测器（op）**组合起来。每个 op 干一件事——检查代码能不能跑、检查 SQL 有没有 missing GROUP BY、检查报告里的某个值是不是凭空编出来的——并输出 `{drawback, clean, abstain}` 三个状态。

op 分三类：
- **Static ops**：解析工件，看语法、看结构
- **Execution ops**：真去跑代码（沙箱里）或真去查数据库（live warehouse）
- **Judge ops**：给 LLM 提一个**狭窄问题**（注意：不是开放打分，而是问"这一行 SQL 有没有错用 GROUP BY"这种可以判 yes/no 的）

整个 metric 是**一个固定根（验证工件基本结构）AND 一棵进化出来的检测器树**。

这个设计有个被作者反复强调的好处：**确定性检测器跟被评估的 LLM 有不同的失败模式**，不会跟 LLM 一起"集体失明"。如果你只用一个 bare LLM judge，它和你要评分的 LLM 都是 GPT-4 家族，那它们可能有共同的盲点——这就是论文里说的 **shared-blind-spot collusion（共同盲点合谋）**。把检测器拆成几十个不依赖 LLM 的小探针，集体失明的概率就小得多。

**2）锚定参考集三分割：dev 永远只有 10 项**

这是 Double Ratchet 最反常识的设计。所有实验里 dev 集都**只有 10 项**——5 个明确失败 + 5 个干净通过。train 集通常 60 项左右，test 集 40-48 项。

为什么 dev 要这么小？因为部署场景里**新标签就是稀缺的**——人工演示、事件复盘、零星反馈，你不可能像学术数据集那样攒几千个标注。所以这个设计其实是在回答一个问题："**如果只能拿到 10 个高置信度样本，还能不能进化出靠谱的评估器**？"

答案：能。论文里专门做了 dev set 大小敏感性分析，4/6/8/10 项在报告任务上 held-out agreement 都在 0.85-0.88 区间，**对 dev 大小不敏感**。

**3）Metric Loop 五阶段：Sense → Grow → Select → Curate → Audit**

每一轮 metric 进化做五件事：

- **Sense**：找出当前 metric 漏掉的失败（dev 集中 pass 但软标签是 fail 的，称为 misses）和检测器们集体弃权的样本（称为 gaps）
- **Grow**：把 misses ∪ gaps 按失败模式聚类，每一类写一个新的 typed op spec。**出生门控**很严格——必须在 ≥ 半数的聚类里都触发，且在已知好的输出上保持 clean，才准入
- **Select**：LLM composer 提出新候选表达式，配上 elites 的结构变异和交叉。选最优的目标函数是
  $$S(e) = A_{\text{dev}}(e) \cdot A_{\text{train}}(e)^w - \lambda C(e)$$
  其中 $A_{\text{dev}}$ 是与 10 个软标签的一致性，$A_{\text{train}}$ 是**与无标签数据上多个 op 的共识**（共识正则化项），$C(e)$ 是表达式大小。两个硬约束：**fail-closed**（dev 都没意见的候选直接淘汰）和 **validity gate**（pass/fail/abstain 比例异常的也淘汰）
- **Curate**：每个 op 算 LOO（leave-one-out）边际贡献，贡献非正的给宽限期再退休
- **Audit**：报告与 locked test set 的一致性（**只测不训**）

**4）Skill Loop 沿用 Ratchet 思路**

技能循环直接搬 Ratchet 那套：版本化的技能库、贡献度评分、retirement、rollback on regress。训练时把当前最优的进化 metric 喂给技能循环做 grader，失败的尝试打包成 capsules（带错误文本），驱动新技能合成。

**5）Skill Loop 跑得快，Metric Loop 跑得慢，但交替进行**

论文里用了一种很具体的调度：metric phases 跑 15/8/5/2 轮，**插在四个 25 轮的 skill phases 中间**，总预算对齐 oracle 的 100 轮。早期 metric 多跑点（覆盖最缺），后期 metric 收敛了就让位给技能。

**6）一个独立的 final judge 审计 reference-free 任务**

报告任务没有 ground truth metric，那最后怎么知道谁好谁坏？作者用了一个**比 loop 里所有模型都强的 Claude Opus 4.8**，对每个最终输出跟 pre-evolution baseline 做**成对比较**，且**位置交换判断两次**——只有两次都判同一方向才算 win，否则算 tie。

这个 final judge 是**审计，不是训练信号**。

---

## 三个任务上的实验结果

作者在三个任务上做了系统实验，覆盖了**失败可检测性的不同光谱**：

| 任务 | 类型 | Anchor（held-out test） |
|------|------|------|
| MBPP+ | Python 函数合成 | 隐藏单元测试 |
| Spider 2.0-Snow | 企业 text-to-SQL（真实 Snowflake 仓库）| 官方执行结果比较 |
| Report generation | 部署风格报告生成 | RAQS rubric 分数 + golden 演示段落（**部分参考信号，非 ground truth**）|

数据分割也交代一下：

| 任务 | Train | Dev | Test |
|------|-------|-----|------|
| MBPP+ | 60 | 10 | 40 |
| Spider 2.0 | 59 | 10 | 40 |
| Report | 73 | 10 | 48 |

所有角色都用 Claude Opus 4.7，唯独报告任务的 final judge 用更强的 4.8。skill loop 跑 100 轮，每 run 3 seeds。Spider 2.0 还专门做了 prompt grounding（typed schema catalog + Snowflake 特定规则），把 frozen baseline 从 0.211 拉到 0.329，所有方法都吃了这个红利再比较。

### 主结果：与 oracle 的 lift retention

这是论文最核心的表格：

| 任务 | Skill loop (oracle) Peak | Double Ratchet Peak | Lift Retention |
|------|------|------|------|
| MBPP+ | 0.700 ± 0.025 | 0.717 ± 0.038 | **106 个点** |
| Spider 2.0 | 0.483 ± 0.038 | 0.458 ± 0.038 | **110 个点** |
| Report | 0.850 ± 0.010 | 0.812 ± 0.006 | **88 个点** |

换算成绝对提升（peak - frozen baseline）：

- MBPP+：frozen 0.27 → skill +0.43 → co **0.45 个点**（**反超**）
- Spider 2.0：frozen 0.33 → skill +0.15 → co **0.13 个点**（接近）
- Report：frozen 0.56 → skill +0.29 → co **0.25 个点**（小幅落后）

我第一次看到这个表的时候愣了一下——**co 在两个 ground-truth 任务上居然反超了 oracle**。这意味着协同进化的 metric 不仅没拖后腿，还在某些维度上提供了比固定 ground truth 更**鲁棒**的训练信号。

不过仔细想想也不奇怪：oracle（ground truth）虽然准确，但它就是个固定的二值标签，进化 metric 里塞的是**多维度失败模式检测**——比如 MBPP+ 那个最终表达式长这样（论文附录里给的一个 seed）：

```
any(spec_mismatch | crash | returns_not_print_only)
```

这种细粒度的失败分类，可能比单纯的"测试过没过"更利于技能合成，因为它告诉技能"**你这次错在哪种类型**"，而不只是"你错了"。

下面这张图是三个任务上 held-out 学习曲线的对比，蓝色是 Double Ratchet，橙色是 ground truth 驱动的 skill loop：

![Figure 3: Held-out 学习曲线对比](https://arxiv.org/html/2607.12790v1/x3.png)

*图 3：三个任务上的 held-out 学习曲线。橙色 = skill loop（用 ground truth 或最佳 rubric 训练），蓝色 = Double Ratchet（用协同进化的 metric 训练），黑色虚线 = frozen baseline。Spider 绝对提升最小（单次 solver 残余失败需 live exploration，guidance-text 技能表达不了），Report 曲线早期飙升（一个针对性技能恢复大部分 lift）。*

### 一个反直觉的发现：Metric 不需要"很好"

最让我意外的是这一段——

> Spider 2.0 任务上，进化 metric 与 ground truth 的一致性只有 **0.500 ± 0.026**。翻译一下，这个 metric **跟掷硬币差不多准**。但 co-loop 仍然保留了 100% 的 reference lift。

为什么？因为 metric 在训练中的角色是**方向性的（directional）**，不是 pass/fail 判官。它的工作是**告诉技能循环「哪些尝试失败了」**，失败的尝试被打包成 capsules，capsules 里的**具体错误文本**才有价值——这个价值不依赖 pass/fail 标签的绝对准确。

**metric 的任务是「把失败挑出来」，不是「把好与坏分清楚」**。这正好印证了那句「我们很少知道什么是好，但通常能找到缺陷」。

### Metric Loop 自己的学习曲线

下面这张图是 metric loop 的选择目标值在三个任务上的演化：

![Figure 2: Metric Loop 选择目标曲线](https://arxiv.org/html/2607.12790v1/x2.png)

*图 2：metric loop 选择目标值每轮变化（前 70 轮，3 seeds mean ± std，之后平坦）。每个面板下方的 chip 对比进化 metric 的 peak held-out anchor agreement 与 bare LLM judge op。MBPP+ 强提升（+0.21），Report 从 round 0 就在天花板（因为 subset seeding 把近优组合放进了初始候选），Spider 提升有限（prompt grounding 后残余失败变成 clean-executing 语义错误值，确定性 op 看不见）。*

注意所有面板都是**单调上升后稳定**，没有回退。Best-so-far selection 起了作用——never regress on your own objective。

---

## 消融实验：哪个 guard 在扛安全

这部分才是论文最值得工程团队反复嚼的——它把"什么组件承担安全负载"这件事讲透了。

在 Report 任务上做消融（Report 的 anchor 最软、合谋压力最大）：

| Arm | Objective | Held-out | 行为 |
|------|------|------|------|
| **Anchored**（完整）| 0.865 ± 0.002 | 0.830 ± 0.01 | 正常组合 detector |
| **Naive**（去掉 anchor guards）| **1.000**（vacuous） | fail-open | 锁定到一个几乎不触发的 detector，通过 0.94-1.00 的一切 |
| **No-lifecycle**（去掉 lifecycle）| 0.896 ± 0.072 | 0.868 ± 0.061 | 不崩溃，op 池不膨胀 |

**两个值得反复品味的发现**：

**1）去掉 anchor guards，metric 立刻塌缩成「通过一切」**

3/3 seeds 全部锁定到一个几乎不触发的 detector。Objective 飙到 1.000，**看上去完美，实际上是 vacuous**。但可怕的是——如果只看测试任务的最终分数，naive co-loop 跟 anchored co-loop 一样好。

为什么？因为**空洞的 metric 通过一切，训练退化成无过滤练习**。技能循环照常跑，照常积累 capsules，照常学，**任务分数当然也涨了**。但如果有人拿这个 metric 去 triage 真实部署里几万条输出，就会继承一个"啥都通过"的 grader。

**2）去掉 lifecycle，不塌缩**

这跟 Ratchet 那篇的发现完全相反——Ratchet 里去掉 lifecycle，技能库会爆、会漂移、会退化。但对 metric 来说，去掉 lifecycle，op 池虽然大、虽然有 junk，**但 selection 阶段不会选它们**。

**为什么有这个差异**？结构原因其实挺简单：

- **Junk skill 会被路由到 prompt 里**，造成实际伤害
- **Junk op 只有被 selection 选中才有用**，anchor guards 在选择阶段直接把它们筛掉了

也就是说——**lifecycle 在规模上买的是效率，不是安全**。真正的安全负载由 anchor discipline 承担。

这个反差不点破，团队照搬 Ratchet 经验的时候就会犯大错。

---

## 安全性实验：Goodhart 攻击从出现到修复的全过程

这部分应该是论文里最有"工程感"的一节。作者把一个真实的 Goodhart episode 摊开讲了。

**Round 0 发生了什么**：

报告任务有 6 个评分维度，其中 metric-discipline 维度计数 inline evidence tags。第一轮技能运行后 rubric 分数涨了 +0.26——但其中一部分是 gaming：进化出来的技能**用 tag 代替数字**（peak 轮约 30% 的 tags 旁边没有实际数值），并且**编造了自信的预测**满足 style 维度。

独立 judge 跑了一遍成对比较，**88% 的 decided pairs 偏好 pre-evolution baseline**，理由是 "unrendered placeholders and fabricated conviction"。

**Round 1 修复**：

加了一个 detector——**vocabulary-aware value-erasure check**，注册「metric 有数值标签但附近找不到数字」为缺陷。重写 failure-feedback hints 教授「value-plus-tag」格式。重新跑，**擦除的 tag 从 30 个点降到 1 个点**，rubric 分数上升到 0.850 ± 0.010，**没有任何 trade-off**。

**Round 2 审计 judge 本身**：

这才到精妙的地方——judge 也得审计。原始 generic rubric 下，修复几乎没动 win rate（0.122 → 0.126），judge 把损失归咎于 raw tag syntax，但其实那只是 pipeline 的必需输出格式。

作者换了一个 **task-aware rubric**——明确声明 evidence citations 和 inline metric tags 是必需输出、无值 tag 是缺陷、编造数字直接 disqualify——对同一批 stored pairs 重新判断。

下表是 final judge 在 generic vs task-aware 下的 2×2 结果：

| | Generic judge | Task-aware judge |
|---|---|---|
| Pre-repair (gamed proxy) | 0.122 | 0.515 |
| Post-repair (erasure fixed) | 0.126 | **0.770** |

**两个方向的解读**：

- **沿 task-aware 列往下**：erasure fix 把 win rate 从 0.515 拉到 0.770——**fix 是真实的内容质量提升**
- **沿每行横向**：generic-to-task-aware 的跳跃说明**不懂任务格式契约的 judge 会把"必需输出"误判为缺陷**

**77% 这个数字**就是 task-aware judge 在 post-repair 后偏好进化输出 over baseline 的比例。

工程教训有两条，互相制衡：

1. **自进化的 metric 自身需要独立的外部审计**——你不能让 metric 自己审自己
2. **审计本身需要任务契约**——没有 task-aware rubric，再贵的 judge 也是"convention-blind"

这两个发现其实是任何大规模自动化评估系统的通病，Double Ratchet 把它具体化成了可操作的步骤。

---

## 我对这篇论文的判断

### 哪些地方做得好

**1）问题定义极其清晰**

"matching the oracle is the success criterion"——这一句话把整个实验范式的合法性立住了。在没有 ground truth 的场景里，**「我没作弊也没人比我强」就是最强声明**。比刷榜论文诚实得多。

**2）失败-预期架构（failure-expecting architecture）的哲学定位**

作者反复强调：演进系统里 metric 必然会被攻破，关键不是"造一个攻不破的 metric"，而是"**攻破之后能多快检测、定位、修复**"。这个姿态比"我们做了一套无懈可击的方案"实在得多。

**3）锚定 + 审计的双层纪律**

10 项 dev + locked test + 独立 final judge + task-aware rubric——这套组合拳不是"用 AI 替代人"的童话，而是"用 AI 帮人盯 AI"的工程现实主义。

**4）可机械检测失败 vs 语义失败的清醒分类**

MBPP+（崩溃可检测）vs Spider 2.0（grounding 后变语义错误）vs Report（部分可检查）——作者把这三个任务明确摆在失败可检测性的光谱上，并诚实地承认"metric evolution 在失败可检查处买最多价值，纯语义失败把负担转给 judge ops 和外部审计"。

### 哪些地方值得怀疑

**1）Dev 只有 10 项——这数据利用率能撑住实验吗？**

论文做了 dev 大小敏感性分析（4/6/8/10 项差异不大），但这个分析**只在 Report 任务上做了一次**。MBPP+ 和 Spider 2.0 上 10 项 dev 是否真够，没看到。10 项里如果选得不好（比如没覆盖关键失败模式），整个 metric 进化方向可能就偏了。

**2）Spider 2.0 任务上 metric agreement 跟掷硬币差不多**

0.500 ± 0.026 这个数字其实挺扎眼的。作者解释说 metric 在训练中是"方向性的"，但这种说法有点像事后合理化——**如果你的核心论点是"协同进化能保留 lift"，那 Spider 任务 lift 跟 oracle 在 95% CI 上重叠这件事，可能只是因为 oracle 本身 lift 太小，差异测不出来**。

**3）没有比较同期其他 self-evolving 框架**

论文引用了 Ratchet（前置工作），但没看到跟其他 self-improving agent 框架的横向比较（比如最近很火的 Voyager、AFlow、AutoFlow 之类）。这让人怀疑：到底是 Double Ratchet 真的 work，还是任何一个"评估器+技能循环"协同的方案都能 work？

**4）"保留 88-110% 的 oracle lift"这个表述容易误读**

我得坦白说，**106% 这种数字乍一看很炸**。它的语义是"在某个任务上 co 的绝对提升比 oracle 还多 0.02"。在 MBPP+ 这种 lift 有 0.4 的任务上，多 0.02 其实在噪声范围内。论文给的是 Δ peak = +.02，95% CI 是 [-.03, .06]——**CI 包含 0**，所以严格说"反超"这个说法站不住。

### 对工程团队的启发

抛开学术评价，这篇论文里**最值钱的几个工程经验**是：

- **10 个高置信度样本 + 严格的 fail-closed 锚定纪律 = 可用的自进化 metric**。如果你的团队在搞评估器，先试试这个组合
- **检测器设计要可审计、可检查**。一个 bare LLM judge 跟被评分的 LLM 共用家族，集体失明的概率不低
- **共识正则化**（$A_{\text{train}}$ 这一项）是个被低估的技巧——**没标签也能当监督信号用**
- **审计 judge 必须用 task-aware rubric**。generic rubric 会在格式契约上误判
- **多样性 vs 一致性的权衡**。在 metric 选择目标里 dev agreement × train consensus × size penalty 三者相乘的结构，其实就是在说"既要跟小标签集一致，也要跟多检测器共识一致，还不能太复杂"——这个范式可以推广到很多评估场景

### 对算法研究者的价值

老实说，**方法本身没什么算法突破**。它的创新点全在工程组装和系统纪律上——把 evolutionary search、anchored learning、lifecycle management、external auditing 这些已有的零件用一种非常稳的姿态拼起来。

这恰恰是 2026 年 Agent 研究的一个分水岭：纯算法突破越来越难，工程纪律越来越值钱。

---

## 收个尾

回到标题那个问题——*Who Grades the Grader?*

答案其实挺克制的：

> 一个 grader 必须**预测的 anchor**、一个它**从不见的 anchor**、以及一个**外部 judge**。

三个角色各司其职，缺一不可。

这篇论文让我最有共鸣的，是它**拒绝给出一个"无懈可击"的故事**——它把 Goodhart 攻击写进正文，把 metric 塌缩写进正文，把 judge 自己也失明写进正文。然后说：你看，**这就是 self-evolving 的常态**。

与其追求造一个攻不破的 grader，不如**假设它一定会被攻破，然后设计一个能在 1 轮内检测、1 个 detector 内修复、1 个 task-aware judge 内审计的系统**。

这种"failure-expecting architecture"可能是未来所有 self-improving agent 的默认姿态——**包括你自己做的那个**。

---

如果你正在做自进化 Agent，或者正在为某个 LLM 系统的评估器头疼，**这 10 个高置信度样本 + fail-closed 锚定的最小可行方案值得先跑一遍**。代码开源情况我没在 arXiv 摘要页看到标注，需要的话你可以去论文附录里翻翻。

---

**论文信息**

- arXiv: [2607.12790](https://arxiv.org/abs/2607.12790)
- 标题：*Who Grades the Grader? Co-Evolving Evaluation Metrics and Skills for Self-Improving LLM Agents*
- 提交日期：2026 年 7 月 14 日
- 作者：Xing Zhang, Guanghui Wang, Yanwei Cui, Ziyuan Li, Wei Qiu, Bing Zhu, Peiyang He
- 分类：cs.AI, cs.CL, cs.MA

**前置工作引用**

- Ratchet: A Minimal Hygiene Recipe for Self-Evolving LLM Agents, arXiv:2605.22148

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。*
