# 不动模型权重，只改"外壳"：StarHarness 把企业智能体成功率拉高 20-35 个点

模型在企业环境里翻车，真的是模型不行吗？

上周看到 ServiceNow 和 Mila 的这篇论文（arXiv: 2608.24804），我第一反应是怀疑——他们声称不动一个参数，只改智能体外面的那层"脚手架"（harness），就能在三个企业基准上把成功率拉高 20 到 35 个百分点。说实话这个数字大到有点反直觉。但看完方法和实验设计，我被打动了。这篇文章值得认真聊聊。

---

## 🎯 核心摘要

**痛点**：把 GPT 级别的模型丢进真实企业环境（Kubernetes 故障诊断、ServiceNow 工单系统、财务 SaaS 工作流），成功率经常只有两三成。问题往往不在模型笨，而在工具接口、prompt 框架、上下文管理这些"外围设施"和环境不匹配——模型–环境错配。

**方案**：StarHarness 把 harness（prompt、工具 schema、skills、子智能体结构、agent 循环配置）当成可优化对象，用一个提案器模型读失败轨迹、写 git diff 补丁，配合"搜索集/选择集/留出集"三层隔离的防作弊机制，迭代演化 harness。模型权重和基准任务全程冻结。

**效果**：三个环境分别接受 4、12、5 个补丁后，ITBench SRE 从 40.0% 涨到 75.0%，EnterpriseOps-Gym 从 23.3% 涨到 43.7%，AutomationBench Finance 从 57.1% 涨到 83.2%。更狠的是，演化出的 harness 不换模型直接套到 GPT-5.4-mini、GPT-5.5 甚至 Qwen 上，增益基本保留——Qwen3.5-27B 在 ITBench 上从 25.6% 直接干到 70.0%。推理成本还降了 17%-53%。

**我的判断**：这不是底层突破，是工程方法论层面的扎实工作。但它踩中了一个被低估的真理——在 agent 时代，"外壳工程"的投资回报率可能比换模型高得多。做企业 agent 落地的人都该看看。

---

## 📖 论文信息

- **标题**：StarHarness: Evolving Harnesses with Stratified Search for Enterprise Environments
- **作者**：Esakkivel Esakkiraja, Denis Akhiyarov, Vikas Yadav, Sai Rajeswar, Patrice Bechard, Sridhar Nemala, Sagar Davasam
- **机构**：ServiceNow、Mila、Université de Montréal
- **链接**：https://arxiv.org/abs/2608.24804 （代码开源：github.com/ServiceNow/StarHarness）
- **提交日期**：2026 年 8 月 25 日

---

## 🤔 为什么这篇论文值得看

做 agent 落地的人应该都有体感：同一个模型，在干净的学术 benchmark 上表现亮眼，一进企业真实环境就拉胯。原因很杂——工具文档不全、参数 schema 有坑、环境里有大量"只可意会"的操作惯例（比如 ServiceNow 里 priority 和 impact/urgency 要联动更新，这种规则从来不写在 API 文档里）。

传统的解法有两条路：微调模型，或者换更大的模型。两条路都贵，而且治标不治本——环境里的隐式知识很难靠训练数据覆盖全。

这篇论文换了个视角：既然模型权重不想动，那就把**模型外面那层东西**当成优化对象。他们管这层东西叫 harness——prompt 与任务框架、工具定义与 schema、参数预处理、skills、MCP providers、子智能体结构、上下文管理、验证与结束逻辑，全算在内。你想想看，这其实是 agent 系统里工程含量最高、但又最少被系统优化的部分。大部分人都是手写一版 harness 就不再动了。

论文问的三个问题也很实在：只用一小撮任务演化 harness，会不会过拟合？演化出来的东西能迁移到别的任务、别的模型吗？它到底修复了哪些失败？

---

## 🏗️ StarHarness 是怎么工作的

![StarHarness 总体框架](https://arxiv.org/html/2608.24804v1/figures/starharness_overview.png)

*图 1：StarHarness 总体框架。模型权重和基准任务全程冻结，只有 harness 在演化。演化池被分成三层：搜索集（proposer 可见 traces）、选择集（隐藏，只给分数）、留出集（封存，最后才打开）。每一代提案器读记忆账本、写一个有边界的补丁，过验证门和隐藏评估后才决定是否接受。*

形式化地说，这是一个外循环优化问题：

$$h^* = \arg\max_{h\in\mathcal{H}} J(h; \mathcal{D}_{\mathrm{holdout}})$$

$h$ 是 harness，$J$ 是平均任务得分。注意优化目标写的是留出集上的得分——但实际搜索时留出集是封存的，这个写法只是在强调"我们要的是泛化，不是刷分"。

整套系统有两个角色要分清：**优化器**是一个基于 Pi 变体的 coding harness，里面跑着提案器模型（GPT-5.4）；**被修改的对象**是一个独立的叫 Stirrup 的 agent harness，也就是真正下场做任务的智能体。提案器不直接做任务，它只做一件事——读失败轨迹，然后给 Stirrup 写补丁。

### 三层任务隔离：防止"偷看答案"

这是我觉得设计得最讲究的部分。从每个基准里采样约一半任务构成演化池（$K \approx N/2$），采样时按三个 baseline 描述符分层：失败模式（wrong_tool、context_loss、missing_evidence、premature_conclusion 这些类别）、baseline 得分、验证器通过率。分层采样保证演化池和全集分布一致，不会专挑简单题。

然后演化池再一分为二：

| 集合 | 提案器能看到什么 | 干什么用 |
|---|---|---|
| 搜索集 $D_{\mathrm{search}}$ | 完整 traces 和结果 | 诊断失败、提出假设 |
| 选择集 $D_{\mathrm{select}}$ | 只看聚合分数，traces 全隐藏 | 决定补丁接不接受 |
| 留出集 $D_{\mathrm{holdout}}$ | 完全封存 | 最终泛化评估，只打开一次 |

为什么要这么折腾？因为如果提案器能看到选择集的具体轨迹，它就可能写出"恰好修掉这几道题"的投机补丁。让接受决策基于它没见过的任务，才能逼出真正通用的修复。这个思路和机器学习里 train/val/test 的划分一脉相承，但用在 harness 演化上，说实话我之前没见到做得这么较真的。

### 每一代的流程

每一轮迭代走这么几步：提案器读记忆账本（frontier 得分、逐任务结果、已接受的假设、被丢弃的尝试都在里面，跨迭代持久化）和搜索集 traces，产出**一个**有边界的 git diff 补丁。然后过几道门：

1. **验证门**：检查 scope、有没有泄漏、import 是否合法、单任务冒烟测试，任一失败直接回滚；
2. **Test flip 门**：提案器得自己指定一个任务，候选补丁必须能翻转这个任务的结果，否则连昂贵的全量评估都省了，直接拒；
3. 通过后在**隐藏选择集**上评估；
4. **确定性接受规则**：选择集平均分严格提升，或者持平且验证器指标提升，才提交为新 frontier；否则回滚并记账。

防护栏也写得很死：禁止按任务 ID 分支、禁止硬编码答案、禁止在 prompt 里写验证器内容、禁止访问真值。补丁 scope 限定在基准可编辑目录和共享 agent 框架内。

搜索策略有两种：爬山（单 frontier，每轮一个补丁，只进不退）和树搜索（维护候选节点集合，提案器可以 explore / draft / debug / merge / improve）。树搜索只在 EnterpriseOps-Gym 上用了，而且是"先树搜索探索、后爬山利用"的顺序设计——作者自己也承认这不是因果性的头对头对比，只是说明两种模式互补。

---

## 🧪 实验：三个环境，21 个补丁

三个环境覆盖了企业 agent 的三种典型形态：

| 基准 | 规模 | 任务 | 评分 |
|---|---|---|---|
| ITBench SRE | 40 场景 | Kubernetes 根因分析，查告警、事件、traces、指标、拓扑 | 结构化诊断评分 |
| EnterpriseOps-Gym ITSM | 103 任务 | 事件/问题/变更/知识/用户五类 ITSM 工作流，ServiceNow MCP 后端 | SQL 验证器查数据库终态 |
| AutomationBench Finance | 100 任务 | AP/AR、费用、报表、记账，横跨 47 个模拟 SaaS 应用 | 程序化断言，违规得 0 分 |

演化用 GPT-5.4（medium reasoning）当被测智能体，GPT-5.4 当提案器。三个环境分别只接受了 4、12、5 个补丁——加起来 21 个，就换来了下面这些数据。

### 主结果

![ITBench SRE 对比](https://arxiv.org/html/2608.24804v1/figures/harness_comparison_itbench.png)

*图 2：ITBench SRE（40 任务）上各 harness 的对比。默认 Stirrup 40.0%，Pi 42.1%，Codex 46.7%，在 Pi 上跑 GEPA prompt 优化到 61.2%，StarHarness 演化后的 Stirrup 直接到 75.0%。*

![EnterpriseOps-Gym 对比](https://arxiv.org/html/2608.24804v1/figures/harness_comparison_eops.png)

*图 3：EnterpriseOps-Gym ITSM（103 任务）成功率。默认 Stirrup 只有 23.3%，GEPA 甚至降到 21.4%（比 baseline 还差，说明纯 prompt 优化在这个环境水土不服），Codex 41.7%，StarHarness 43.7%。*

![AutomationBench 对比](https://arxiv.org/html/2608.24804v1/figures/harness_comparison_automationbench.png)

*图 4：AutomationBench Finance（100 任务）领域目标完成比例。各 baseline 在 56%-66% 之间挤成一团，StarHarness 拉到 83.2%。*

几个值得咂摸的点。其一，相比 GEPA（Pi）这种专门的 prompt 优化方法，StarHarness 在三个环境分别多涨 **13.8、22.3、17.6 个百分点**——差距来自演化对象的不同：GEPA 只优化 prompt，StarHarness 连工具 schema、skills、agent 循环一起改。其二，EnterpriseOps-Gym 上 GEPA 反而负优化，说明"只调 prompt"在有状态、多步骤依赖的环境里是有天花板的。其三，Codex 在 EnterpriseOps-Gym 上拿到 41.7% 相当接近 StarHarness 的 43.7%，作者分析 Codex 靠的是默认 MCP 预处理压缩 schema、减少无效调用——这反过来印证了接口层确实是性能瓶颈所在。

不过要泼一盆冷水：论文自己声明，和 GEPA/Codex 的对比是**描述性**的，两者在 prompt、工具、执行策略上全都不同，没法隔离单一变量的因果贡献。所以"碾压 GEPA"这种话别说太满。

### 泛化与迁移：这篇论文最值钱的证据

涨分不稀奇，稀奇的是涨的分能带走。

![EnterpriseOps-Gym 跨模型迁移](https://arxiv.org/html/2608.24804v1/figures/eops_benchmark_comparison.png)

*图 5：EnterpriseOps-Gym 上的跨模型迁移。灰色是 baseline，蓝色是套用演化后 harness（不重演化）。Qwen3.6-27B 从 18.2% 到 38.8%，GPT-5.4 从 23.3% 到 43.7%，GPT-5.5 从 37.8% 到 48.5%——超过了 Claude Fable 5 的参考分 48.1%（橙色，Artificial Analysis 外部参考，非受控对比）。*

完整的冻结迁移表更夸张：

| 基准 | 模型 | Baseline | StarHarness | 提升 |
|---|---|---|---|---|
| ITBench | Qwen3.5-27B | 25.6% | 70.0% | **+44.4 pp** |
| ITBench | GPT-5.4-mini | 33.1% | 79.4% | **+46.3 pp** |
| ITBench | GPT-5.4 | 40.0% | 75.0% | **+35.0 pp** |
| ITBench | GPT-5.5 (medium) | 50.8% | 78.7% | **+27.9 pp** |
| EnterpriseOps | Qwen3.6-27B | 18.2% | 38.8% | **+20.6 pp** |
| EnterpriseOps | GPT-5.4-mini | 13.6% | 31.1% | **+17.5 pp** |
| EnterpriseOps | GPT-5.4 | 23.3% | 43.7% | **+20.4 pp** |
| EnterpriseOps | GPT-5.5 (high) | 37.8% | 48.5% | **+10.7 pp** |
| AutomationBench | Qwen3.6-27B | 48.2% | 75.5% | **+27.3 pp** |
| AutomationBench | GPT-5.4-mini | 29.6% | 70.0% | **+40.4 pp** |
| AutomationBench | GPT-5.4 | 57.1% | 83.2% | **+26.1 pp** |
| AutomationBench | GPT-5.5 (medium) | 59.6% | 84.9% | **+25.3 pp** |

演化时用的是 GPT-5.4，结果 Qwen 这种完全不同的模型家族套上去增益反而更大——ITBench 上 Qwen3.5-27B 涨了 **44.4 个点**，GPT-5.4-mini 涨 **46.3 个点**，比演化时用的模型本身（+35.0）还多。我的解读是：演化学到的主要是**环境侧的接口与惯例知识**，而不是"怎么哄好某个特定模型"，所以它对弱模型的补偿作用更强。这个发现如果成立，工程含义很大——harness 演化出来的东西是环境资产，不是模型配件。

留出集泛化也稳：ITBench 演化集涨 45.0 pp、留出集涨 **31.7 pp**；EnterpriseOps 分别 +22.0 / **+15.1**；AutomationBench 甚至留出集（**+29.3**）比演化集（+23.0）还高。三层隔离的防作弊机制确实起到了作用。

### 它到底修了什么

轨迹分析把 21 个补丁学到的知识归成三类，这个分类本身挺有洞察：

1. **接口修复**：EnterpriseOps-Gym 修复了 MCP 参数处理，保留复合 schema、剪掉误导字段、在调用前剥离 null/空占位参数；AutomationBench 用结构化行操作替代了脆弱的原始电子表格编辑。
2. **环境惯例显式化**：把"priority 与 impact/urgency 联动更新""变更前先做 triage""相对日期锚定到沙箱时钟"这类从不写在文档里的规则，写进了 prompt 和 skills。
3. **操作知识与搜索压缩**：ITBench 获得了一个根据观测证据对候选上游原因排序的 forensics overview；AutomationBench 拿到了日期与财务计算器。把可程序化的重复工作从开放式推理里挪出来。

第三类我特别想多说一句。"搜索压缩"这个词用得很准——很多 agent 失败不是推理能力差，而是每一步都在重新发明轮子。给 harness 塞一个计算器、一个排序先验，模型就不用每次从 token 里现算日期了。

效果在轨迹层面看得见。ITBench 上假阳性诊断从 0.79 降到 0.33，真阳性从 0.45 升到 0.78——不是瞎蒙得更准了，是误诊少了。EnterpriseOps-Gym 上每任务轮数从 18.12 砍到 9.87，工具调用从 29.53 降到 16.83，成本直接腰斩（**-53%**），验证器通过率从 34.5% 跳到 72.8%。AutomationBench 上有违规的任务从 20 个降到 4 个，零分任务从 24 个降到 6 个。三个环境的推理成本分别降 17%、53%、29%。分数涨、轨迹短、花钱少，三个方向同时改善，这比单看分数更有说服力。

---

## 💡 我的判断

**亮点**：

- 把 harness 当一等优化公民，且演化范围覆盖 prompt、工具 schema、skills、子智能体、循环配置全栈——这比只优化 prompt 的工作（GEPA、DSPy 那一路）覆盖面大得多，数据也证明覆盖面确实重要。
- 搜索集/选择集/留出集的三层隔离 + test flip 门 + 确定性接受规则，整套防投机机制设计得很较真。留出集增益坚挺、跨模型迁移增益甚至更大，说明确实没刷分。
- 成本下降是真金白银的工程价值。涨 20 个点同时推理成本腰斩，这种买卖在商业部署里太香了。

**问题**：

- 21 个补丁的因果贡献无法隔离，作者自己也承认。演化过程更像一个黑箱的"知识沉淀器"，哪些改动真正起作用、哪些是搭便车的，说不清。
- 演化要跑全量评估，成本不低。论文没给出演化本身的总开销，这是个不小的省略。
- AutomationBench 的 Finance-100 子集和 harness 设置与原基准论文不同，得分不可直接横比——作者倒是老实交代了，但读者容易忽略。
- ITBench 上仍有遗留失败：近因已经够格收工了，agent 还在往上层原因瞎搜。接口能修，推理习惯难修。

**跟同期工作比**：GEPA、DSPy 优化的是"说什么"，StarHarness 优化的是"整个交互层长什么样"。前者是后者的真子集。和 Claude Code、Codex 这类手工打磨的 harness 比，StarHarness 提供的是**自动化演化的方法论**——人写 harness 靠经验，它靠搜索。我不认为它会取代手工 harness 工程，但"手工搭初版 + 演化精修"可能会变成标准工作流。

---

## 🔧 工程启发

如果你在做企业 agent 落地，有几个可直接搬走的点：

- **先跑 baseline 收集失败模式分类，再针对性改 harness**，别拍脑袋调 prompt。论文的 wrong_tool / context_loss / missing_evidence / premature_conclusion 四分法可以直接抄。
- **改 harness 时把"环境隐式惯例"显式化**——联动字段、日期基准、前置检查步骤，这些知识放 prompt/skills 里比指望模型自己悟靠谱得多。
- **给重复性计算做工具**（计算器、排序器），把模型从"每一步现推"里解放出来，轨迹会明显变短。
- **评估 harness 改动时用隐藏任务做接受决策**，防止优化过程过拟合到你盯着看的那几个 case。

论文还留了个有意思的口子：未来用 RL 联合演化 harness 和模型权重，让脚手架和策略共同专精。如果 harness 演化单独就能涨 20-35 个点，联合优化的上限在哪，我挺好奇的。

---

觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我
