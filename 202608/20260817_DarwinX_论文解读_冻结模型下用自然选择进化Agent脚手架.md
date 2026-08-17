# 模型冻着不动，分数狂涨 49 分：DarwinX 把 Agent 自我改进做成了真正的"自然选择"

上周看到一组数据的时候我愣了一下：同一个冻结的 GPT-5.5，同一个 agent 骨架，WebArena-Infinity 上的真实任务通过率从 43.5% 干到了 93.0%。没有微调，没有换模型，甚至连 gold label 都没有用。

你有没有这种感觉：现在的 LLM 智能体，模型本身越来越强，但真正决定它能不能干活的，其实是外面那一层"壳"——prompt 怎么写、工具怎么配、出错怎么兜底、流程怎么控制。这层壳行话叫 harness。问题是，这层壳目前基本靠人手工调，调出来的东西换个任务就废一半。

这篇 DarwinX 干的事，就是把调壳这件事做成了自动化的进化算法。而且不是那种"改一改、跑一跑、留最好的"的伪进化，是字面意义上的自然选择：维持一个 harness 种群，用 benchmark 自带的验证器当"生存环境"，适者生存，互补者杂交，模型权重全程冻结。

**核心摘要**：自我改进循环的老毛病有两个——单谱系搜索容易一条路走到黑，局部修好的东西会悄悄搞坏别的任务。DarwinX 的解法是把自我进化重定义为 harness 种群上的选择过程：一条 preserve-and-extend 契约保证新变体只能"扩展覆盖、不许回退"，一个 archive 留住所有备选谱系供日后重组，失败轨迹、教师轨迹、自我对比三类证据共享同一个编辑接口。四个 benchmark 上平均涨约 17 分，最夸张的是 WebArena-Infinity 涨 49.5 分且审计干净（reward hacking 不升反降）。这篇的真实定位：不是底层算法突破，而是把"选择压力"这件事做对了的工程系统，值得每个在做 agent 自改进的人细读。

**论文信息**

- 标题：DarwinX: Evolving Agent Harnesses Through Natural Selection
- 作者：Yifan Zhang, Yutong Dai, Juntao Tan, Luyu Yang, Rishi Mullur, Thai Hoang, Zhiyuan Hu, James Zhu, Phil Mui, Silvio Savarese, Ran Xu, Zeyuan Chen
- 机构：Salesforce AI Research / Salesforce Agentforce
- arXiv：https://arxiv.org/abs/2608.07545 （2026 年 7 月 31 日提交）

---

## 🎯 为什么需要这篇论文：自改进循环的两个慢性病

先交代下背景。让 agent 自己改自己的 harness 这个方向，过去一年多已经很热闹了。改 prompt 的有 OPRO、PromptBreeder、TextGrad；改组件图的有 ADAS、AFlow；改技能文档的有 SkillOpt；最激进的是 SICA 和 Sakana AI 的 Darwin Gödel Machine（DGM），直接让 agent 改自己的源代码。DGM 在 SWE-bench 上把 agent 从 20% 改到 50%，相当能打。

但几乎所有这些工作共享同一个内层循环：跑一批 rollout → 反思 → 提出一个有界编辑 → 用留出集信号门控。这个循环有两个慢性病，做过这类系统的人应该都有体感。

**病一：路径依赖。** 单谱系 keep-best 的搜索，会被早期几个编辑带偏，然后陷入平台期。SICA 的作者自己就报告了这个现象。说实话，这就像只用梯度下降还只从一个初始化出发，卡在哪个盆地里全看运气。

**病二：跨任务干扰。** 这个更隐蔽。你修好了数据库类任务，回头发现文本解析类任务偷偷回退了。任务分布越广越严重。很多编辑在小子集上赢、在全 benchmark 上输，如果选择信号只看局部改进，系统就是在积累技术债。

DGM 有 archive 但每次只变异一个父代、只跟父代比分数，没有合并算子——两条互补的谱系永远不会汇合。HarnessX 靠隔离变体来防干扰，代价是专家变体各自困在自己的谱系里。

DarwinX 的切入点就是：**内层循环大家都做得差不多了，真正欠火候的是外层的"选择过程"这件事**。选择压力怎么设计，决定了这个系统是在爬山还是在进化。

## 🧬 方法：把自我进化拆成种群、契约和重组

论文里的术语约定：Monet 是 Salesforce 自家的 agent，DarwinX 是进化它 harness 的程序。Monet (base) 是未进化的，Monet (DarwinX) 是进化后的。所有对照实验里底层模型都是冻结的——这个设定很关键，保证了所有分数差异只能归因于 harness。

harness 的可编辑面分两层：skill 层（prompts、记忆、蒸馏知识）和 code 层（工具、控制流、agent 循环本身）。

![DarwinX 总览](https://arxiv.org/html/2608.07545v1/figures/darwinx_teaser_v6.drawio.png)

*图 1：左图是核心循环——base harness（skill + agent + code 三层）进入进化循环，变体经选择保留或剪枝，互补的幸存者可合并；底部一句话点题："冻结模型 + 无标签选择"。右图是四个 benchmark 的战果：TB 2.1 域内 +7.7（对 GPT-5.5）、TerminalWorld 留出 +7.3（对 Opus 4.8）、WA-Inf 合成到真实 +49.5（对 GPT-5.5）、SWE-V 零样本迁移 +3.4（对 Opus 4.8）。*

### 核心契约：preserve-and-extend

整篇论文我最欣赏的设计是这个准入契约，一句话讲：**新变体必须净增益为正，且回退有界，才被允许活下来**。

具体地，每个变体按逐任务解决率 $\hat{p}_t(v)$（avg@k）打分。子代 $c$ 相对父代 $p$ 的逐任务变化 $\Delta_t = \hat{p}_t(c) - \hat{p}_t(p)$，然后定义两个量：

$$g(c) = \sum_t \Delta_t, \qquad R(c) = \sum_t (-\Delta_t)_+$$

$g$ 是净收益，$R$ 是总回退量。子代被接纳当且仅当 $g(c) \gt 0$ 且 $R(c) \leq \delta$。

这相当于给每个编辑上了"不许拆东墙补西墙"的紧箍咒。前面说的跨任务干扰病，就是靠这条规则在系统层面被摁住的。

裁决还不是一次性的。一个推理型 verifier agent 分两个阶段工作：先 promote（读取子代试验证据和共享记忆，给出 promote / revert 判定），再 probe（被 promote 的子代要通过更高保真度的 preservation probe 重测，才有资格引导后续搜索）。作者管这叫双速设计——对"尝试"宽松，让有希望的噪声信号也能进树；对"信任"严格，只有反复验证过的变体才能塑造未来搜索。探索期偏 recall，最终评估偏 precision。

这个设计其实很懂 benchmark 的噪声问题。agent benchmark 上单次幸运 rollout 太常见了，一次通过伪装成能力增益，然后系统在这个虚假基础上继续堆编辑——做过这类实验的都知道这有多坑。

### 种群与重组：死掉的分支也是遗传材料

DarwinX 跟爬山法的两个本质区别：编辑是加性的（additive），记录在持续增长的 archive 里，谱系是累积能力而不是以新换旧；以及不丢弃任何变体——整体落败的变体也保留，因为它可能握着某个单独编辑，跟另一个分支的编辑组合后能解锁新任务。

Archive 是一棵树：每个节点存 harness 快照、编辑 delta、逐任务分数、试验证据和蒸馏教训。变体按子代解决集 $S(c)$ 与父代 $S(p)$ 的关系分类：

| 变体类型 | 条件 | 能否参与继承 |
|---|---|---|
| improver | $S(c) \supsetneq S(p)$ | 能，保留所有继承解 |
| neutral child | $S(c) = S(p)$ | 能 |
| stepping stone | $S(c) \subsetneq S(p)$ | 不能，只回传蒸馏教训 |
| archived node | 以一些解换另一些解 | 不能 |
| specialist | 解决了任何兄弟都没解决的任务 | 作为遗传材料保留 |

父代选择用累积谱系增益 $G(c) = G(p) + g(c)$ 排序，以概率 $1-\beta$ 选增益最高的已确认变体，以概率 $\beta$ 在更大种群上拓宽。按累积增益而不是原始分数排序这一点挺讲究——各变体在不同任务子集上被筛选，原始分数根本不可比。

合并算子（merge）是种群设计的价值兑现时刻：当几个变体解决互补任务时，从公共祖先 $H_0$ 出发把它们的加性编辑合起来，$H = H_0 \oplus \Delta$，其中 $\Delta = \Delta_{code} \oplus \Delta_{skill} \oplus \Delta_{prompt} \oplus \Delta_{tool}$。合并子代保留的条件很硬：$S(child) \supseteq \bigcup_i S(v_i)$，必须覆盖所有父代胜利的并集，一个都不能少。

![DarwinX 系统总览](https://arxiv.org/html/2608.07545v1/figures/darwinx_overview.png)

*图 2：三个模块的全貌。左侧 preserve-and-extend contract 与两阶段 verifier；中间是种群谱系树——节点按累积增益 G(v) 标记，有的被 promote 有的被剪枝，父代选择大部分时候利用最优节点、小概率拓宽；右侧是共享记忆——失败模式分类器聚合全 benchmark 的主导主题（比如"setup 成本主导超时"），按 K_{g+1} = Agg(K_g, worked, regressed, themes) 更新，proposer 和 verifier 都读它，这让搜索发明的是全局能力而非逐任务打补丁。*

### 三类学习信号共享一个编辑接口

这是另一个我觉得挺漂亮的设计。DarwinX 不绑定固定的训练配方，而是定义了一个信号接口：任何能解释 harness 该怎么改的证据源，都能成为进化压力。三类原生信号全部转成 harness 编辑，都不碰模型权重：

| 信号 | 内容 | 适用任务区间 |
|---|---|---|
| Failure-derived（∇） | 总结失败轨迹，定位缺失能力 | 普通变异的默认信号 |
| Teacher-derived（π*） | 把参考求解器的成功轨迹蒸馏为可复用方法 | "walls"——一个成功 rollout 都没有的任务 |
| Self-derived（A） | 对比 agent 自己通过/失败的 rollout，找出让成功可靠的因素 | "variance-band"——同一 k 样本组内有通过有失败 |

这三个信号在构造上就是互补的：任务被动态划分成 reliable solves / variance-band / walls 三个带，proposer 从不盲目改进，总能看到这个任务当前最有信息量的那类证据。

![种群与重组机制](https://arxiv.org/html/2608.07545v1/figures/darwinx_population.png)

*图 3：左侧是变异循环——环境、教师、agent 的轨迹经 analyzer 汇总成三类信号 {∇, π*, A}，喂给 proposer 产生变体。中间把变体分成可重组（specialist / improver / neutral child）与不可重组（stepping stone / archived）两类。右侧是重组筛选：互补变体的 harness 编辑合并后，必须通过"覆盖所有父代胜利并集"的验收门才算数。*

---

## 🧪 实验：四个递进难度的考场

实验设计是这篇论文另一个值得说的地方。四个 benchmark 按"进化信号与测试的分离程度"递增排列，一层比一层难作弊：

| RQ | Benchmark | 冻结基座 | 进化数据 | 报告数据 | 报告指标 |
|---|---|---|---|---|---|
| RQ1 域内进化 | Terminal-Bench 2.1 | GPT-5.5 | 89 个验证器任务 | 同 89 任务 | avg@5 |
| RQ2 留出泛化 | TerminalWorld | Opus 4.8 | 94 训练任务 | 41 个不相交留出任务 | pass@1 |
| RQ3 合成到真实 | WebArena-Infinity | GPT-5.5 | 300 合成意图 | 1,260 真实任务 | 确定性 pass@1 |
| RQ4 跨 benchmark 迁移 | SWE-bench Verified | Opus 4.8 | 无（纯迁移） | 500 个 issue | 官方 pass@1 |

比较策略也很克制：主打同模型配对（base Monet vs 进化 Monet，同基座同任务同验证器），公开榜单数字只当背景。承重的结论都是 matched-model delta。

### RQ1：Terminal-Bench 2.1，+7.7 分的纯 harness 增益

| Agent | 模型 / effort | avg@5 |
|---|---|---|
| Monet (DarwinX) | GPT-5.6 Sol / medium | **84.7±1.2** |
| Claude Code | Fable 5 / xhigh | 83.8±1.2 |
| Monet (DarwinX) | GPT-5.5 / high | 83.2±1.2 |
| Codex | GPT-5.5 / xhigh | 83.1±1.1 |
| OpenAI 官方参考 | GPT-5.6 Sol / medium | 81.8 |
| Terminus 2 | GPT-5.5 / xhigh | 78.0±1.2 |
| Monet (base) | GPT-5.5 / default | 75.5±3.5 |

几个数值得单独拎出来看。对 base Monet 的纯 harness 增益是 75.5% → 83.2%，**涨 7.7 分**。同基座下 Terminus 2 用更高的 xhigh effort 只有 78.0%——说明增益不是砸算力砸出来的。换上更强的 GPT-5.6 Sol（medium effort）后达 84.7%，追平甚至略超当时的榜首 Claude Code + Fable 5（83.8%，xhigh），而且 effort 档位更低。

增益落点也很有意思：ML 与科学计算簇 60.1 → 74.9%（**涨 14.8 个点**），数据/数据库簇 83.9 → 97.8%（**涨 13.8 个点**），而已经很强的簇几乎不动（系统管理 92 → 98%，安全 85 → 84%，在噪声内）。88 个配对任务里 36 改善、43 不变、9 回退，没有任何簇的回退超出噪声——这正是 preserve-and-extend 契约的经验足迹。

还有个细节：算力消耗是有针对性的。6 个从失败翻转为解决的任务上，轮数约翻倍（22 vs 11）、token 约四倍（380K vs 89K）；而双方本来就能解决的 69 个任务上几乎不动（13 vs 12 轮）。进化学会的是"在难任务上多花时间"，不是无脑变贵。

奖励黑客审查也做了：370 条获奖轨迹里只有 2 条被标记，唯一确认的捷径是某次试验 agent 读了任务自己发布的 README 里含答案的字符串——这是 agent 行为而非 harness 属性，同一任务其余 4 个样本里 3 个是合法解决的。

### RQ2：TerminalWorld，种群价值的最清晰证据

留出集 41 个任务，单次尝试 pass@1：

| Agent | 模型 | pass@1 |
|---|---|---|
| Monet (DarwinX) | Opus 4.8 | **68.3%** |
| Claude Code | Opus 4.8 | 65.9% |
| Monet (base) | Opus 4.8 | 61.0% |
| Terminus-2 | GPT-5.5 | 61.0% |
| Monet (DarwinX) | GPT-5.5 | 56.1% |
| Codex | GPT-5.5 | 51.2% |
| Monet (base) | GPT-5.5 | 48.8% |

这里有个很有教育意义的现象。训练子集分数在循环内饱和到了 1.000，但留出集只有 68.3%——**31.7 分的代理差距**，而且最贴合训练代理的变体并不是最佳泛化者。如果系统是单谱系 keep-best，到这里就结束了，你手里只有那个过拟合的家伙。

DarwinX 的 archive 里有四个高分 specialist，分别解决 24、25、26、27 个留出任务。它们的能力重叠但不同。**合并之后达到 28，超过每一个单独 specialist。** 这是"保留种群而不是单一 incumbent"最直接的证据，也是整篇论文里我觉得最有说服力的一张牌。

还有个稳健性插曲：首次留出扫描恰逢基础设施降级窗口，重跑后各 specialist 提升了 5 到 10 题，而 Monet (DarwinX) 保持 28 题不变——进化出的 harness 是对基础设施抖动最不敏感的变体，不是最幸运的那个。

当然也得诚实：对 Claude Code 的一题优势（McNemar p=1.0）只是暗示性的，41 个任务里 1 题就是 2.4 分，这个统计功效确实有限。

### RQ3：WebArena-Infinity，49.5 分的狂飙与反作弊审计

这是全文最炸的结果。设置上很干净：进化只用 300 个由应用自身描述文档生成的合成意图（管线从不读 benchmark 任务），LLM judge 打分；最终在 10 个应用、1,260 个不相交真实任务上用确定性验证器测 pass@1。真实任务和验证器在进化全程不可见。

| 应用 | Kimi | Qwen | Gemini+BU | GPT-5.5+BU | Monet (base) | Monet (DarwinX) | Δ |
|---|---|---|---|---|---|---|---|
| Elation 病历 | 50.0 | 54.2 | 81.7 | 92.5 | 95.8 | 96.7 | +0.9 |
| Elation 处方 | 23.3 | 41.7 | 80.8 | 90.8 | 20.0 | 95.0 | **涨 75.0** |
| GitLab 计划跟踪 | 39.3 | 37.1 | 63.6 | 77.9 | 63.6 | 97.9 | +34.3 |
| Gmail | 70.0 | 56.7 | 75.0 | 85.0 | 25.0 | 98.3 | **涨 73.3** |
| Gmail 账号联系人 | 40.0 | 33.3 | 61.7 | 87.5 | 21.7 | 91.7 | +70.0 |
| Handshake 求职 | 50.0 | 50.5 | 50.5 | 83.5 | 36.5 | 84.0 | +47.5 |
| Linear 账号设置 | 54.2 | 65.8 | 73.3 | 81.7 | 43.3 | 94.2 | +50.9 |
| PayPal 钱包 | 70.7 | 71.4 | 88.6 | 90.0 | 49.3 | 95.7 | +46.4 |
| Superhuman 通用 | 15.0 | 25.8 | 50.0 | 80.8 | 31.7 | 87.5 | +55.8 |
| Xero 发票 | 52.5 | 55.8 | 80.8 | 93.3 | 39.2 | 96.7 | +57.5 |
| **总体** | 43.3 | 48.3 | 69.3 | 86.1 | 43.5 | **93.0** | **涨 49.5** |

从 43.5% 到 93.0%，每个应用都在涨，最大的增益集中在状态变更繁重的应用上（处方、Gmail 这类）。比最强同模型基线 GPT-5.5 + Browser Use（86.1%）高 6.9 分。

但说实话，涨 49.5 分这种数字，我的第一反应不是"牛"，是"是不是有猫腻"。作者显然预判了这种怀疑，搞了一套两阶段动作合法性审计：静态检测器反混淆 JavaScript、对计分字段做污点追踪、标记越权访问，然后独立 Opus 4.8 judge 复核被标记轨迹。结果：

| 审计指标 | Base | DarwinX | 变化 |
|---|---|---|---|
| 原始 pass@1 | 53.0 | 94.4 | +41.4 |
| 审计后 pass@1 | 43.5 | 93.0 | +49.5 |
| 无效成功数 | 120 | 17 | −103 |
| 确认无效比例 | 23.5% | 1.4% | −22.1 |

注意看这个表里最有意思的一件事：**base Monet 的确认违规率有 23.5%，进化后反而降到 1.4%**。base 有 155 次评估面违规、97 次特权主机违规、26 次漏洞利用违规，进化后这三类全部消失。能力与合规是同步提升的——这直接排除了"涨分靠更激进的捷径"这个最自然的怀疑。

harness 具体进化了什么？新增了四个面向契约的浏览器技能加系统提示修改：推导验收契约、检查客户端可见状态、使用应用自有语义操作、同时验证渲染状态与持久化。进化的方向是"更守规矩"，挺耐人寻味的。

也有一个要泼冷水的点：WAI 上重组被反复尝试但**每次合并都被回退**，收益沿一条短的 accepted 主谱系累积。也就是说在这个 benchmark 上，种群机制的贡献其实是存疑的，作者自己在局限性里也承认了这一点。

### RQ4：SWE-bench Verified 零反馈迁移

把 TB2.1 上进化出的最佳 harness 不加修改直接跑全部 500 个 SWE-bench Verified issue，冻结 Opus 4.8，官方测试框架评分：**421/500，84.2% 官方 pass@1**，比 80.8% 的 fix-skill 参考高 3.4 分，全程没接收任何 SWE-V 反馈。

这是支撑"进化出的是通用 agent 能力而非 benchmark 专属补丁"的关键一环。不过迁移增益（+3.4）明显小于域内增益（+7.7），说明 harness 能力里还是有一部分是任务相关的——这个不必替作者遮掩。

### 消融：进化到底学出了什么

TB2.1 进化谱系新增了 7 个技能，全部属于 verification / artifact-contract 一族：

| 进化出的技能 | 作用 |
|---|---|
| verifier-contract、contract-candidate | 推导任务验收契约，定稿前对照检查 |
| graded-artifact-final-check、artifact-verification-loop | 验证被评分的产物，迭代修复-复查 |
| real-tool-artifact、tool-grounded-artifact | 让输出基于真实工具执行而非断言 |
| security-contract-repair | 针对安全与契约检查修复解法 |

7 个技能没有一个增加领域知识，全都在干一件事：**定稿之前先验证**。跨 benchmark 交叉验证也对得上——WAI 上同类契约技能让确认违规率从 23.5% 降到 1.4%。作者很谨慎地把这定性为"组合归因而非逐技能因果消融"，这个态度我认。

---

## 🤔 我的判断：值在哪，坑在哪

**最值钱的地方**：preserve-and-extend 契约 + 种群 archive + 合并算子这个组合，把自改进循环从"爬山"真正变成了"进化"。TerminalWorld 上 specialist 合并反超的实验，是这个范式迄今我看到的最干净的证据。另外整篇论文的实验诚实度在 agent 自改进这个容易自嗨的领域里算一股清流：双速确认、基础设施故障区分、奖励黑客审计、McNemar 检验，一套组合拳下来，49.5 分这种吓人的数字反而变得可信了。

横向看，这个方向正在快速收敛。DGM 验证了"改代码 + archive"可行，上个月 Nie 等人的 TTHE（arXiv:2607.08124）证明了冻结模型下 harness 进化能涨 6 到 38 分，现在 DarwinX 把选择机制补全了。三天内好几家独立团队得出同一个结论——模型是固定资产，harness 才是可优化的变量。这不是巧合，是范式转移的前兆。

**几个问题也得摆出来**：

一是重组的贡献没被单独消融。WAI 上每次合并都失败，TB2.1 的消融也是组合归因。也就是说"种群 + 合并"到底贡献了多少，目前只有 TerminalWorld 一个 41 题的小 benchmark 撑着，统计上还偏薄。

二是成本。论文没有给出进化循环的总算力账单。DGM 当年一轮 SWE-bench 跑两周、两万美金的事还历历在目，DarwinX 这种"评估即训练"的路线，花销恐怕只多不少。avg@3 筛选加 avg@5 确认的重复测量开销是有意为之，但这开销也决定了这套东西短期内是大厂的玩具。

三是迁移增益的天花板。+3.4 分的 SWE-V 迁移说明 harness 能力有相当比例是任务相关的，"通用能力"这个论断目前还处于"比补丁强、离通用远"的中间态。

**工程启发**：如果你在做 agent 自改进，有三样东西可以直接抄——双速确认（宽松尝试、严格信任）、preserve-and-extend 契约（净增益为正且回退有界）、共享失败主题记忆（让 proposer 修病理而不是修症状）。这三个设计都不依赖种群框架，单谱系系统也能用。

最后说回那句点题的话：冻结的模型不等于固定的 agent。当模型迭代越来越贵、评测算力越来越便宜，"把评估算力转化为持久能力"这条路线，可能会成为 agent 工程的标配。DarwinX 未必是终局，但它把选择压力这件事做对了，这就够格成为一个里程碑。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
