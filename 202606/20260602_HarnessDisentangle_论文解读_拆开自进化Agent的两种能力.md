# Harness 更新 ≠ Harness 受益：把自进化 Agent 的两种能力彻底拆开

> 一个反直觉的发现：让 9B 的小模型当 evolver，写出来的 skill 跟 Claude Opus 4.6 写的几乎一样好。但反过来，弱模型自己拿到一份高质量 harness，却几乎用不起来。

## 核心摘要

最近这一两年，self-evolving agent 已经被吹得很响——给 agent 配一套可改写的 harness（prompts、skills、memories、tools），让它从执行轨迹里反过来更新这套 harness，看起来是绕开权重训练的一条便宜路径。但所有这些工作的评测，几乎都在问同一个**端到端**的问题：这套自进化方法有没有让 agent 变强？

这篇论文的作者们觉得这个问题问错了。端到端的提升里其实混着两种完全不同的能力：**evolver 写出有用更新的能力**（harness-updating），和 **agent 用上这些更新的能力**（harness-benefit）。它们应该被分开测，而且很可能根本不跟模型基础能力同步。

把这两个能力剥开之后，结论挺让人意外的：**harness-updating 跟基础能力几乎没关系**——9B 小模型当 evolver，效果跟 Opus 4.6 当 evolver 差不到哪去（任意 benchmark 上 evolver 之间最大 gap 只有 3.1 个点）；**harness-benefit 跟基础能力非单调**——中档模型受益最多，强模型撞天花板，弱模型则因为"不会调用 skill"或"调用了不照做"两种失败模式，几乎吃不到红利。

这个 finding 直接改写了 self-evolving agent 系统的设计直觉：**别在 evolver 上烧太多钱，把 capability budget 留给 task-solving agent**，而且 agent 训练的重点应该挪到 harness 调用 + 长程指令跟随这两件事上。

---

## 论文信息

- **标题**：Harness Updating Is Not Harness Benefit: Disentangling Evolution Capabilities in Self-Evolving LLM Agents
- **arXiv**：[2605.30621](https://arxiv.org/abs/2605.30621)（v1，2026/05/28）
- **作者**：Minhua Lin, Juncheng Wu, Zijun Wang, Zhan Shi, Yisi Sang, Bing He, Zewen Liu, Tianxin Wei, Zongyu Wu, Zhiwei Zhang, Dakuo Wang, Xiang Zhang, Benoit Dumoulin, Cihang Xie, Yuyin Zhou, Suhang Wang, Hanqing Lu
- **页数**：24 页正文 + 附录，9 figures，12 tables
- **代码**：https://github.com/A-EVO-Lab/a-evolve/tree/release/harness-evolution

---

## 一、为什么这个问题值得拆？

先把场景说清楚。所谓 **agent harness**，指的是包在一个固定权重的 LLM 外面那一整层"非参数"的东西——system prompt、可调用的 skills 库、长期 memory、工具接口。模型权重不动，但这些 harness 可以在 agent 跑任务的过程中被改写。这就是所谓的 **harness self-evolution**：agent 跑完一批任务，evolver（也是一个 LLM）从轨迹里抽出失败教训和成功套路，写回 harness，下一批任务用更新后的 harness 再跑。

![图1：Harness self-evolution 流程概览。模型权重 f 始终固定，evolver e 根据 agent 在第 t 步的执行证据 D_t 把 harness 从 H_{t-1} 更新到 H_t，下一轮 agent 拿着新 harness 继续跑。](https://arxiv.org/html/2605.30621v1/x1.png)

*图 1：Harness self-evolution 协议示意。本质上是一个固定权重、只更新外部 harness 的循环。*

这条路线最近爆了——Reflexion、Self-Refine、ACE、GEPA、SkillRL、EvoSkill、MemMA 一连串方法。它们的实验报告通常长这样：「我们的方法把 SWE-bench 上的 pass rate 从 X% 提到 Y%」。

**问题就在这里**。这个 Y - X 到底来自哪？

可能来自三件事，端到端分数全部混在一起：
- **基础能力**：agent backbone 本身在初始 harness 上的水平
- **evolver 写更新的能力**：evolver 从轨迹里提炼出来的 skill 是否真的有用
- **agent 用更新的能力**：哪怕 skill 写得很好，agent 在新 harness 下能不能正确调用、忠实执行

之前的 paper 默认把这三件事打包评测，结果就是**你不知道 capability budget 应该花在哪**。是把 evolver 换成 GPT-5 收益更大，还是把 agent backbone 升级成 Opus 4.6 收益更大？没人说得清。

这篇 paper 把这三件事拆开了。具体定义两个能力：

- **harness-updating capability** $\Delta_{\text{update}}(e)$：固定一组 anchor agent，看用 evolver $e$ 产出的更新带来的平均 gain
- **harness-benefit capability** $\Delta_{\text{benefit}}(f)$：固定一组 anchor evolver，看 agent backbone $f$ 在更新后的 harness 上能拿到的最大 gain

加上**基础能力** $M_{\text{base}}(f) = J(f, H_0)$——也就是初始 harness 上 agent 直接做任务的 pass rate。

剩下的事就是看：这两个 evolution 能力，到底跟基础能力有没有关系？

---

## 二、两个核心发现，一张图就说完了

![图2：本文核心结论。左：harness-updating 在基础能力上 flat——不同 tier 的 evolver 产出的更新 gain 高度接近。右：harness-benefit 在基础能力上非单调——中档模型受益最多，强模型撞天花板，弱模型反而最少。](https://arxiv.org/html/2605.30621v1/x2.png)

*图 2：两个发现的可视化总结。这张图是整篇论文的灵魂——左右两条曲线一旦看进去，后面所有实验都是在解释和补强这两条线。*

让我们一块块拆。

### 发现 1：写更新这件事，跟模型大小几乎无关

实验设置很干净：固定 task-solving agent 用 Opus 4.6 / Sonnet 4.6 / Qwen3-235B 当 anchor，把 evolver 换遍 7 个 backbone（从 9B 的 Qwen3.5-9B 一路到 Opus 4.6），看每个 evolver 带来的平均 gain。

![图3：每个 evolver 的 harness-updating 能力。三个 benchmark 上 evolver 之间的最大 gap 都不超过 3.1pp，且最佳 evolver 跨 benchmark 频繁换人——Qwen3-235B 在 SWE 第一，到了 MCP 直接垫底。](https://arxiv.org/html/2605.30621v1/x3.png)

*图 3：三个 benchmark 上 evolver 的 harness-updating gain。横轴是不同 evolver，纵轴是平均 gain（pp）。*

几个结论值得停下来咂摸一下：

- **Spread 极窄**：任意一个 benchmark 上，最好和最差的 evolver gap 不超过 3.1 个百分点。3.1 个点。这就是说，你把 evolver 从 9B 升级到 200B+，下游 agent 的 pass rate 提升空间就这么大点。
- **跨 benchmark 没赢家**：Qwen3-235B 在 SWE-bench 上 lead（8.2pp），到了 MCP-Atlas 上反而垫底（0.6pp）。这种重排现象意味着 evolver 的"水平"是高度任务相关的，没有一个普适最强的 evolver。
- **最反直觉的一条**：在 SkillsBench 上，**最小的 9B 模型 Qwen3.5-9B 写出的 skill 带来的 gain 最大**（3.8pp），超过 Opus 4.6（2.3pp）和 Qwen3-235B（1.5pp）。

我看到这个数的时候确实愣了一下。一个 9B 的开源模型，在一个 frontier 模型 anchor 当 agent 的设置下，写出来的 skill 带来的下游收益反而**比 Opus 4.6 自己写的还高**？

作者顺着这个反常做了个非常细的 case study。

![图4：在 SkillsBench 的 flink-query 任务上，对比 Opus 4.6 agent 在三种条件下的轨迹：(左) 没有 skill，得分 0.67；(中) 用 9B evolver 写的 skill，得分 1.0；(右) 用 Opus 4.6 evolver 写的 skill，得分 1.0。两个 skill 在程序步骤上几乎是同构的。](https://arxiv.org/html/2605.30621v1/x4.png)

*图 4：9B 写的 skill vs Opus 4.6 写的 skill。两边给 agent 修复了同一个 bug——遗漏 FINISH-event filter——只是表层措辞和啰嗦程度不一样。procedural 内容一致。*

这个发现把"写 harness 就是个推理大活、必须用最强模型"的直觉打掉了。**写一段把"成功路径"固化下来的 skill，不是一个非要顶级模型才能完成的任务**——它更像是"总结刚才发生了什么"，9B 模型完全够用。

继续看 evolver-side 第二个观测：

![图5：MCP-Atlas 上 anchor agent 拿到 7 个不同 evolver 写的 harness 后的 post-evolution 分数。每个 anchor agent 内部 7 个 evolver 散点之间的方差，远小于不同 anchor agent 之间的差距。](https://arxiv.org/html/2605.30621v1/x5.png)

*图 5：每个 anchor agent 的 within-agent spread（最大 5.1pp，Qwen3-235B 上）远远小于 between-agent gap（最高 36.0pp）。*

**这就是关键**。决定 post-evolution 最终分数的，主要不是你选了哪个 evolver，而是 task-solving agent 本身有多强。把 evolver 从 9B 换成 Opus 4.6，对最终结果的影响大约 5pp 量级；但把 agent 从 Qwen3-32B 换成 Opus 4.6，影响是 36pp 量级。

如果你正在做 self-evolving agent 系统，这个数据应该直接写进设计文档里。

### 发现 2：用更新这件事，中档模型最赚

发现 1 把 evolver 这一边讲清楚了——大家差不多。那 harness-benefit 这一侧呢？

实验设置反过来：固定一组 anchor evolver，把 task-solving agent 换遍 6 个 backbone，看每个 backbone 拿到 evolved harness 后能榨出多少 gain。

主表 Table 1 拿到的数据如下（重排为 Markdown）：

| Backbone | SWE base / Δ_benefit | MCP base / Δ_benefit | SB base / Δ_benefit |
|---|---|---|---|
| Claude Opus 4.6 | 高 / 较小 | 高 / 较小 | 高 / 较小 |
| Claude Sonnet 4.6 | 中高 / 中等 | 中高 / 中等 | 中高 / 中等 |
| **GPT-OSS-120B（中档代表）** | **中 / 最大** | **中 / 最大** | **中 / 最大** |
| Claude Haiku 4.5 | 中低 / 中等 | 中低 / 中等 | 中低 / 中等 |
| Qwen3-235B | 中 / 中等 | 中 / 中等 | 中 / 中等 |
| **Qwen3-32B（弱档代表）** | **低 / 接近 0** | **低 / 接近 0** | **低 / 接近 0** |

> 注：原文 Table 1 给出每个 backbone 在每个 benchmark 上的具体 base 分数和 $\Delta_{\text{benefit}}$ 数值，跨 benchmark 一致呈现"中档最赚"的非单调形态。这里给出趋势版本以便快速消化，准确数值参见 [arXiv:2605.30621](https://arxiv.org/abs/2605.30621) 原文 Table 1。

把这个关系画到图上更直观：

![图6：SWE-bench 上 Δ_benefit 与 base pass rate 的关系。横轴是 base 能力升序排列，纵轴是 Δ_benefit。曲线呈倒 U 形：弱档底部贴近 0，中档拱起最高，强档随基础能力上升又回落。](https://arxiv.org/html/2605.30621v1/x6.png)

*图 6：Δ_benefit vs base pass rate（SWE）。一个干净的非单调形态：中档拐点在 GPT-OSS-120B 附近。MCP 和 SB 的对应图在附录 D.2，形态一致。*

强档模型受益小，这个还能用"上限论"解释——Opus 4.6 base 已经很高，evolved harness 顶上去也涨不动。**但弱档为什么也少？**

Qwen3-32B 的 base 分数最低，按理说提升空间最大，结果它的 Δ_benefit 反而趋近于零。这就不是天花板能解释的了。

作者把弱档的失败原因深挖了一下，归到了两个非常具体的失败模式上。

---

## 三、弱模型为什么用不上 harness：两个具体的失败模式

![图7：Qwen3-32B 在 SkillsBench 上的两个 harness-benefit 失败模式。左 (threejs)：harness activation failure——Qwen3-32B 用了无效的 multi-key load action，根本没把 skill body 加载进 context；右 (pg-essay-to-audiobook)：harness adherence failure——skill 已加载但 agent 没按里面的步骤走。](https://arxiv.org/html/2605.30621v1/x7.png)

*图 7：两个失败模式的真实轨迹。这两张图很值钱——把抽象的"弱模型 Δ_benefit 低"具象化成了"调用 API 用错"和"看了规范不照做"。*

### 失败模式 A：Harness Activation Failure（不会调用）

第一个失败模式更基础：**agent 根本就没把相关 skill 加载到 context**。

threejs 这个例子里，Qwen3-32B 试图用一个"multi-key load action"一次性加载多个 skill，结果调用格式不合法，整个加载操作直接失败。skill body 压根没进 context，agent 就开始硬做任务。

这一类问题听起来很傻——不就是工具调用错了吗？但当你的 harness 有几十个 skill、需要根据任务动态选择和加载时，**对调用语法和 retrieval 路径的精确遵循**就成了一个真正的能力门槛。弱模型在这上面全军覆没。

### 失败模式 B：Harness Adherence Failure（看了不照做）

第二个失败模式更微妙：**skill 已经在 context 里了，但 agent 跑着跑着就跑偏了**。

pg-essay-to-audiobook 这个例子里，skill 写得很清楚——读 essay、分段、用 TTS 合成、拼接音频。skill body 完整加载到 context 了，Qwen3-32B 在前几步还能跟上，但在长程序列里逐渐遗忘最初的指令，最后输出了一个不符合 skill 规范的结果。

作者还做了一个非常细的 phase-level adherence 分析（Table 2 / Table 3），把每个任务拆成多个 phase，然后看不同 tier 的模型在哪几个 phase 开始跑偏。结论是：**弱模型在中后期 phase 的 adherence 直线下滑**——这在 long-horizon 任务里几乎是致命的。

### 一张表把弱模型的问题量化

附录里有一组叫做 SLR / HFR / LPR 的指标，直接量化了上面两个失败模式：

- **SLR (Skill Load Rate)**：模型的轨迹里有 skill 被加载到 context 的比例。低 = activation failure
- **HFR (Harness Following Rate)**：在 skill 被加载的轨迹里，agent 真的按 skill 指令走的比例。低 = adherence failure
- **LPR (Loaded Pass Rate)**：在 skill 被加载的轨迹里，最终通过的比例

把这三个指标和 Δ_benefit 对照，你会非常清楚地看到：**弱模型的 Δ_benefit 低，几乎就是 SLR + HFR 同时拉跨的结果**。中档模型 SLR 和 HFR 都不错，所以 Δ_benefit 拱起来；强模型 SLR 和 HFR 都很高，但本身 base 已经撞顶了，所以 Δ_benefit 也回落。

我看到这个分解的时候挺受触动的。**弱模型用不上 harness，不是因为它"理解不了"那个 skill 在讲什么，而是因为它的"调用动作"和"长程跟随"这两种能力没练出来**。这其实给 agent 训练指了一条非常具体的路。

---

## 四、对工程实践的几个直接启发

读完这篇，有几条建议是可以直接抄回去用的：

### 1. 别在 evolver 上烧钱，钱留给 agent backbone

如果你正在搭一个 self-evolving agent 系统，发现 1 的含义非常清楚：**evolver 用一个中等水平的开源模型就够了，capability budget 应该全力堆在 task-solving agent 上**。

quantitatively，evolver 之间最多 3.1pp 的差距，agent 之间能拉开 36pp。这个比例让选择变得没有悬念。我之前一直默认 evolver 也得用最强模型——毕竟它要做"反思"和"提炼"——这篇论文给了我一个反例。

### 2. Agent 训练的优化方向应该挪一挪

如果你在训练 agent 模型，传统的训练目标可能围绕"能不能直接做对任务"。但发现 2 + 失败模式分析告诉你：**当 agent 部署在一个有 harness 的环境里时，"能不能正确调用 harness 工件"和"能不能在长程任务里持续跟随指令"才是更值钱的能力**。

这点在 GPT-OSS-120B 这个中档模型的表现上看得最清楚——它的 base 能力远不如 Opus 4.6，但 harness adherence 做得不错，所以 evolved harness 把它的下游表现拉得很高。这是一个很有性价比的 sweet spot。

### 3. 评估 self-evolving agent 别只看端到端数

之前那一波 self-evolving agent 方法的评测，端到端的分数提升被简单解读成"我的方法 work"。这篇 paper 实际上在质疑这种解读：**端到端提升至少有三个来源在打架，你不知道你的方法贡献了哪一块**。

如果你在做这个方向的研究或工程评估，把 base / Δ_update / Δ_benefit 三个量分开报告，不仅会让 reviewer 服气，也会让你自己更清楚下一步该往哪儿改。

---

## 五、几点保留意见

这篇论文打动我的地方是它把一个被打包评测的能力强行拆开，并且拆得很彻底。但有些地方我觉得可以再追问。

**第一，evolver 之间的 3.1pp 差距，可能跟数据集的"自进化空间"有关**。SWE-bench、MCP-Atlas、SkillsBench 这三个 benchmark 的 evolution budget 都是固定的——agent 跑完一定数量的任务，evolver 收到的执行轨迹也就这么多。在这个固定预算下，evolver 能榨出的信息量本身就有上限，再聪明的 evolver 也只能写出"信息量上限以内"的 skill。如果 evolution 过程更长、轨迹更丰富，强 evolver 的优势可能会被放大。

**第二，"flat in base capability" 这个结论**有可能依赖于"7 个模型 + 3 个 anchor agent"这个采样规模。如果把 evolver 放到更宽的能力分布上——比如加上 GPT-5、Gemini 3、DeepSeek 系——3.1pp 的 gap 还守得住吗？我倾向于认为大体上守得住，但论文确实没办法完全排除采样导致的窄区间。

**第三，关于"中档最受益"的工程含义**。这个结论很漂亮，但实际部署时你不会单独按 Δ_benefit 选模型——还要看绝对分数。Opus 4.6 即使 Δ_benefit 小，post-evolution 的绝对值也比 GPT-OSS-120B 高一截。所以"capability budget 留给 agent"是对的，但具体留给"哪一档 agent"，还得看你的成本/性能 trade-off。

**第四，论文没探讨的事**：harness 本身的"互相干扰"。当 harness 里堆了几十上百个 skill 之后，新 skill 的加入是不是会让 agent 在 retrieval 阶段更容易走错路？这其实跟 activation failure 直接挂钩。如果 harness 的规模继续涨，weak-tier 的 activation 问题会不会进一步恶化？这可能比 9B 写不写得出好 skill 更值得后续工作跟进。

---

## 六、收个尾

我觉得这篇论文最值钱的不是某一个具体的实验数字，而是它提供了一个**思维模型**：当你看到一个端到端的 self-evolving 方法宣称把分数从 X 提到 Y 时，你应该立刻问——这 Y - X 里，evolver 占多少？agent 占多少？

这个思维模型一旦内化，很多看似炫目的 self-evolving 工作就会被还原成它本来的样子：要么是 evolver 写出了好 skill（这件事便宜，9B 都能做），要么是 agent backbone 本身在一个更友好的 harness 上发挥得更好（这件事贵，需要真正的强 backbone）。

至于"弱模型靠 harness 翻身"这个故事，看完这篇你会很冷静——**弱模型不是没有上限，而是它连用 harness 的基本功都还没练出来**。投资 self-evolving 系统之前，先看看你的 agent 能不能正确加载 skill、能不能在 50 步之后还记得最初的规范。

如果连这两件事都做不到，evolver 写得再好也是白搭。

---

**arXiv ID**：2605.30621 ｜ 代码：https://github.com/A-EVO-Lab/a-evolve/tree/release/harness-evolution

觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。
