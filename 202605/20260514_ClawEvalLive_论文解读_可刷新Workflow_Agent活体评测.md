# Claw-Eval-Live：连 Opus 都过不了 70%，工作流 Agent 离"自动化"还差一个数量级

## 核心摘要

Agent 评测圈最近一个憋了很久的问题——**benchmark 一旦冻结就开始过时**。WebArena 卷了两年，SWE-bench 刷到 80%，但你打开任何一个 enterprise agent 项目，会发现现实的工作流（HR 审批、CRM 跨系统、e-commerce 对账）和 benchmark 上的"修个 bug"、"订张票"长得根本不像。

Claw-Eval-Live 想解决两个事：（1）**benchmark 自身要能"持续刷新"**——从公开的工作流需求信号（ClawHub Top-500 skills）里聚类、加权、生成任务，每季度滚动一次；（2）**评分不能只看最终输出**——要看 trace、audit log、service state、workspace artifact，能确定性判断的就别让 LLM judge。当前 release 是 105 个任务、13 个前沿模型，结果让人有点喘不过气：**最强的 Claude Opus 4.6 也只过了 66.7%，没有一个模型上 70%**。HR、management、跨系统业务流程是普遍黑洞，而 dev/terminal 这类传统 SWE 任务反而接近天花板。

一句话评价：这不是又一个 leaderboard，是一个**自带过期机制的 benchmark 设计范式**，比起结果本身，方法论更值得看。

---

## 论文信息

- **标题**：Claw-Eval-Live: A Live Agent Benchmark for Evolving Real-World Workflows
- **作者**：Chenxin Li, Zhengyang Tang, Mingxin Huang, Yunlong Lin, Shijue Huang, Shengyuan Liu, Bowen Ye, Rang Li, Lei Li, Benyou Wang, Yixuan Yuan
- **日期**：2026/04/30
- **arXiv**：https://arxiv.org/abs/2604.28139
- **项目页**：https://claw-eval-live.github.io

![图1：Claw-Eval-Live 整体设计——从需求信号到时间戳快照、再到执行证据驱动的评分](https://arxiv.org/html/2604.28139v2/figures/fig1.png)

*图1：核心 pipeline——左边是 demand-side public signals 的刷新源头，中间是聚类→加权→任务扩展→候选筛选→优化选择的发布流程，右边是受控环境里的执行+证据级评分。每季度重跑一遍，让任务集跟着真实需求漂移。*

---

## 问题动机：benchmark 在"慢慢死"

说实话，我刚开始看 agent benchmark 的时候，对"静态发布"这件事没有特别警觉。直到一年前帮一个企业客户做工作流自动化的时候，发现一个尴尬的现实——

**他们关心的工作流，在公开 benchmark 里基本不存在。**

不是"我们 case 太小众"，而是 enterprise workflow mix 本身就在快速漂移：去年 OKR 周报是大头，今年变成跨系统对账；这季度 HR 审批被自动化平台接管了一半，下季度又被合规要求逆转。**任务结构两年大变，benchmark 还停在两年前的 mix。**

这就是 Claw-Eval-Live 想破的第一个局：把"任务集"从一个 frozen artifact 变成一个可重跑的 pipeline。每次 release 都是一个 time-stamped snapshot，但 pipeline 本身可以根据最新的工作流信号（ClawHub Top-500 skills）重新跑出新一版 release。

第二个局更狠：**别只看最终 response**。

> A polished memo is not evidence that the agent queried the correct records, compared the right entities, updated the right object, or repaired the broken artifact that the task actually depended on.

这话我特别共鸣。模型生成一份漂亮的对账报告，但悄悄漏掉了一笔状态变更——final text 看起来完美，业务实际崩了。在生产环境，"sounds right" 和 "did the work" 之间的鸿沟，是 OPS 事故和正常运行的差距。

---

## Signal-to-Snapshot：benchmark 怎么从"需求信号"里长出来

![图2：Signal-to-snapshot 构建流程的细节](https://arxiv.org/html/2604.28139v2/figures/fig2.png)

*图2：从公开 workflow 信号开始，聚类成稳定的 workflow patterns，转成 family weights，扩展成候选任务，再用 family-coverage 约束做最优子集选择。最终从 157 个 runnable candidate 里选出 105 个公开 release 任务。*

整个 pipeline 我觉得最有意思的两点：

1. **信号驱动的 family weighting**：不是作者拍脑袋说"我们要测 HR"，而是从公开信号里聚类出 family，再按出现频率/重要性加权。这把 benchmark 设计者的主观偏好降到了最低。
2. **Discrimination-aware 选择**：候选任务最终入选不仅看 family 覆盖，还看是否能在模型之间产生区分度。如果某个候选任务所有模型都 100% 通过或 100% 失败，就没有评测价值，会被剔除。这是一个非常工程化的想法——benchmark 的价值在于"能区分"，不在于"够全"。

不过这里我也想皱一下眉头——**signal-to-snapshot 的可重现性**到底有多强？公开信号源本身在漂移，聚类参数会有随机性，"family weights"在不同 release 之间能不能保持稳定？论文里说"each release is time-stamped"，但**跨 release 的可比性**这块其实是个未解决的问题。如果你在 v1 上得了 60%、v2 上得了 55%，到底是模型退步了还是 benchmark 变难了？

---

## 评测协议：action-grounded hybrid grading

每个任务在受控环境里执行，记录四样东西：

- **execution traces**：完整 trajectory
- **audit logs**：服务侧的调用记录
- **service state**：执行前后的状态 diff
- **workspace artifacts**：跑完后产出的文件

评分时**优先用确定性 check**（比如数据库 record 是不是改对了、git diff 是不是匹配预期），只在涉及"语义维度"（生成的报告是否合规、表达是否得当）时才上 structured LLM judging。

这点设计我特别认同。LLM-as-judge 在 agent 评测里被滥用得有点过分了——很多 benchmark 整篇都是 GPT-4 给打分，最后变成"用 GPT 评 GPT"的循环。Claw-Eval-Live 这种"能确定就别用 LLM"的姿态，是个值得抄的工程纪律。

公开 pass 阈值是 τ=0.80。每个任务有个 0-1 的 score，超过 0.8 算 pass。两个 leaderboard 指标：

$$\text{PassRate}(m)=\frac{1}{|\mathcal{T}|}\sum_{t\in\mathcal{T}}\mathbf{1}[s_{t,m}\geq\tau], \quad \text{Overall}(m)=\frac{100}{|\mathcal{T}|}\sum_{t\in\mathcal{T}}s_{t,m}$$

Pass Rate 体现"能不能完成"，Overall Completion 保留"做到几分"的信息。排名先看 Pass Rate，平手再看 Overall——这个 tie-breaking 设计让 leaderboard 同时回答两类问题。

---

## 主榜：13 个模型，没人过 70%

![图3：Leaderboard 散点图——pass rate × overall completion](https://arxiv.org/html/2604.28139v2/x1.png)

*图3：每个点是一个模型，横轴 Pass Rate，纵轴 Overall Completion。虚线是 70% 的 Pass Rate 标线——没有一个模型越过去。同时可以看到几个模型在 Pass Rate 相同时，Overall 差距挺明显，说明这两个指标确实是互补的。*

完整 leaderboard：

| 排名 | 模型 | 机构 | Pass Rate | Pass Count | Overall |
|------|------|------|-----------|------------|---------|
| 1 | Claude Opus 4.6 | Anthropic | **66.7%** | 70 | 83.6 |
| 2 | GPT-5.4 | OpenAI | 63.8% | 67 | 81.7 |
| 3 | Claude Sonnet 4.6 | Anthropic | 61.9% | 65 | 79.9 |
| 4 | GLM-5 | Zhipu AI | 61.9% | 65 | 78.1 |
| 5 | MiniMax M2.7 | MiniMax | 54.3% | 57 | 77.5 |
| 6 | MiMo V2 Pro | Xiaomi | 53.3% | 56 | 76.9 |
| 7 | Kimi K2.5 | Moonshot AI | 53.3% | 56 | 76.2 |
| 8 | Gemini 3.1 Pro | Google | 53.3% | 56 | 74.0 |
| 9 | DeepSeek V3.2 | DeepSeek | 51.4% | 54 | 69.3 |
| 10 | Qwen 3.6 Plus | Alibaba | 50.5% | 53 | 71.4 |
| 11 | MiniMax M2.5 | MiniMax | 50.5% | 53 | 70.9 |
| 12 | Qwen 3.5 397B | Alibaba | 49.5% | 52 | 72.7 |
| 13 | Doubao Seed 2.0 | ByteDance | 43.8% | 46 | 70.4 |

几个我觉得最有意思的观察：

**1. Opus 第一这件事不意外，但拉开的距离没那么夸张**——66.7 vs 63.8 vs 61.9，第一和第三只差 4.8 个点。说明在 workflow 这个 task family 上，头部模型基本是"半斤八两"，没有谁一骑绝尘。

**2. 国产开源模型表现亮眼**：GLM-5 跟 Sonnet 4.6 打平，MiniMax M2.7 排第五，DeepSeek 在 9 名但 Overall 拉胯——说明 DeepSeek 容易在"差一点点"的任务上完全 fail，而不是"过不了线但能拿 70 分"。这种模式跟我们之前用 R1 的体感一致：要么彻底搞定要么彻底偏题。

**3. 最差的 Doubao Seed 2.0 还有 43.8%**——头尾差 22.9 个点。这个 spread 比很多老 benchmark 都健康，说明 benchmark 的区分度是有的。

---

## 任务族级别的"黑洞"

![图4：13 个模型在 7 个分析组上的 Pass Rate 热力图](https://arxiv.org/html/2604.28139v2/figures/feishu_raw/feishu_family_heatmap.png)

*图4：颜色越深越绿表示 Pass Rate 越高。Development/Terminal 那一行几乎全绿——最强模型 100%，最弱也 72%+。但看 HR/People 那一行，整列接近全红，最强模型也只能勉强过 22%，几个模型直接 0%。*

把 105 个任务按 family 切开看：

| 任务族 | 任务数 | 表现 |
|--------|--------|------|
| Development / Terminal | 18 | **近天花板**（最强 100%，最弱 72%） |
| Productivity | 25 | **spread 最大**（88% vs 48%） |
| HR / People | 9 | **几乎全军覆没**（最强 22%，多个 0%） |
| Sales / CRM | 12 | 中等难度 |
| Finance | 8 | 中等难度 |
| Research / Doc | 16 | 中等偏易 |
| Management / Ops | 17 | 全 fail under public pass rule |

这张表让我愣了一下：

- **Dev/Terminal 已经接近 saturate**——这跟 SWE-bench 卷到 80%+ 的现象一致，模型在"代码类"工作上已经很能打。
- **HR/People、Management、跨系统业务工作流**才是真正的硬骨头。论文里举了几个 discriminator 任务：meeting-prep action（CTB_COMM_24）、multi-document merge（CTB_D01）、e-commerce monthly reconcile——这些任务要求精确的状态变更和跨系统一致性，**对工具调用的精确度、错误恢复能力、长任务规划能力都是降维打击**。

这个对比给了我一个很强的体感：**SWE-bench 类的进步不代表工作流自动化的进步**。代码任务有清晰的 spec、明确的 unit test、可重试的局部反馈，这些都是 HR 审批和跨系统对账没有的奢侈品。

---

## 任务区分度的"中间带"

![图5：在 0.8 阈值下，任务呈现两极分化，区分度集中在中间带](https://arxiv.org/html/2604.28139v2/x2.png)

*图5：横轴是任务，纵轴是各模型的 pass 比例。在 public pass rule（τ=0.8）下，任务分成两簇——全 fail 和全 pass——真正能区分模型的"中间带"任务相对集中。*

这个观察对 benchmark 设计很有启发：**大部分任务要么所有模型都搞不定（HR），要么所有模型都搞得定（Dev/Terminal 部分），真正能区分模型的就那么一拨**。Claw-Eval-Live 的 discrimination-aware 选择某种程度上就是为了把这些"中间带"任务保留下来——这是 benchmark 设计的核心 challenge。

---

## 我的判断

**亮点**：

- **方法论上有突破**：把 benchmark 从"frozen artifact"重新定义成"refreshable pipeline + time-stamped snapshot"。这套思路可以扩展到任何快速演变的领域。
- **评测协议很务实**：deterministic check 优先 + LLM judge 兜底，比纯 LLM-as-judge 干净得多。
- **覆盖面是真"工作流"**：18 个受控服务 + sandboxed workspace，service-backed workflow 和 workspace repair 在同一个 release 里。这是大多数前 benchmark 没做到的。
- **结果让人冷静**：66.7% 这个数对工程界是个清醒剂——我们离"放心交给 agent 跑工作流"还差得远。

**问题**：

- **跨 release 可比性**：每季度刷新一次的话，模型能力提升 vs benchmark 难度变化怎么解耦？论文没给出清晰的方法。
- **数据来源透明度**：ClawHub Top-500 signals 的具体筛选规则、family weighting 算法在主文里讲得不够细，复现门槛不低。
- **HR/Management 这类任务的 0% 是真的难，还是评测设定本身有 bug**？没看到失败案例的深度分析，我会担心是不是 grader 太严苛或 task fixture 本身有问题。
- **task family 的命名（Claw、ClawHub、OpenClaw、PinchBench）**让外行读起来有点像内部代号——可能是双盲审稿期的匿名化，但对外发布建议给个清晰的对应表。

---

## 工程启发

如果你在做：

- **企业 Agent 平台**：别只看公开 benchmark 的分数。Claw-Eval-Live 的 family 切分给了一个很好的索引——重点关注你业务里 HR/Management/Multi-system 这类高难度场景的实测能力，不要被 Dev/Terminal 类任务的高分骗了。
- **自己设计 benchmark**：discrimination-aware 选择 + action-grounded grading 这两个原则非常值得抄。绝大多数 benchmark 都被"评测过拟合"和"LLM-as-judge 噪声"两件事拖累了。
- **agent harness 选型**：从 leaderboard 看，Claude Opus 4.6 和 GPT-5.4 在 workflow 上仍是第一梯队，但国产模型（GLM-5、MiniMax M2.7、DeepSeek V3.2）在性价比和稳定性上已经追得很近。

最后一个反向论证——你可能觉得"agent 这两年进步那么大，66.7% 还会是问题吗？"。但你想想看，这个 benchmark 设计就是为了**让自己永远过期不了**——signal 在变、任务在刷、模型在追，永远不可能"saturate"。

某种意义上这才是 benchmark 该有的样子。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我*
