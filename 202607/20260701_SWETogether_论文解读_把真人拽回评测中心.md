# 把真人拽回评测中心：SWE-Together 重新定义编码 agent 的多轮考核

> 论文：SWE-Together: Evaluating Coding Agents in Interactive User Sessions
> 链接：https://arxiv.org/abs/2606.29957
> 作者：Yifan Wu, Zhuokai Zhao, Songlin Li, Ho Hin Lee, Jiacheng Zhu, Shirley Wu, Tianhe Yu, Serena Li, Lizhu Zhang, Xiangjun Fan, Shengzhi Li
> 单位：Meta
> 投稿：2026 年 6 月 29 日
> 主题：cs.SE / cs.AI

---

## 核心摘要

做代码辅助的真实场景里，编码 agent 跟用户的交互从来不是一次性把任务讲完。用户在第一轮可能只丢一句"这个 API 太慢"、中途补一句"要兼容 v2 协议"、最后再来一轮"我刚发现少处理了空数组"。但目前几乎所有主流编码评测（你叫得上名字的：SWE-bench、SWE-bench Pro、Terminal-Bench）都把这条多轮链路压扁成一次性指令，最后只看 patch 通过几个测试。

SWE-Together 这次想把"真人用户"重新请回评测正中央。它从 11,260 段真实录制的 user-agent 编码会话里，按可复现、可验证的硬条件挑出 109 个 repo 级任务，配上一个 trajectory-conditioned 的 LLM 用户模拟器，让它像真人甲方一样在被测 agent 跑偏时插入纠正、补充需求、回答追问。最后用两套互相独立的指标同时打分：最终仓库正确性（rubric judge）+ 用户纠错次数（User Correction）。

结果非常直白：能力更强的模型在正确性更高的同时，需要的纠正反而更少，pass@1 与 User Correction 之间的 Pearson 相关系数是负的 0.92。你想想看，"不用催"本身就是能力。

> 论文 ID 2606.29957，v1 投稿于 2026-06-29。

---

## 先把背景说清楚：SWE-bench 那套为啥快撑不住了

你大概已经在各种模型发布稿里反复看到一个数字：Claude Opus 4.5 在 SWE-bench Verified 上拿到了 80.9%。年初 OpenAI 自己悄悄停报 Verified 分数，转头推 SWE-bench Pro；同一批模型在 Pro 上的分数直接掉到 46%-58%，差 30 多个点。

这不是单一事件。SWE-bench Verified 上前 6 名的差距压缩到 1.3 个百分点，已经没有区分度；OpenAI 的污染审计还发现，前沿模型直接靠任务 ID 就能复现 gold patch——这种"背诵式解题"在 SWE-bench Pro 上立刻露馅，GPT-5 High 从 ~55% 掉到 23.3%。

更要命的是另一个维度：评测协议跟真实使用脱节。

你让一个 agent 修 bug，任务描述里通常要把"症状 + 期望行为 + 边界条件"都讲清楚。但现实里，开发者第一句话可能就是"这个太慢了帮我看看"，至于哪一段慢、期望多快、要不要动 schema，全靠后面几轮来回。

所以现在的认知越来越收敛：单轮 patch-through-test 那一套，确实快到极限了。Meta 这篇 SWE-Together 想解决的，正是"接下来该怎么评"的问题——把评测从"修好没修好"升级到"修的过程里，用户得操多少心"。

---

## SWE-Together 在评什么：一个直觉化的对照

先放一张官方给的对比图，把评测范式的差别一眼看明白。

![Figure 1：从 SWE-Bench 风格到 SWE-Together 风格](https://arxiv.org/html/2606.29957v1/x1.png)

*图 1：上半部分是传统 SWE-Bench 风格评测——一段完整任务描述丢给 agent，跑 terminal 拿到 patch，最后只看任务成不成；下半部分是 SWE-Together 风格——多一个 user interaction 循环，模拟器在 agent 跑偏时插入 no-op/question/redirect/new requirement 四种动作，评测时同时看 task success 和 user behavior。左下角是七个模型在 SWE-Together 上的 pass@1、pass² 和 User Correction 叠加。*

Figure 1 里有几个细节特别值得说：

- 左上角 SWE-Bench Pro 上的 pass rate 跟左下角 SWE-Together 的 pass@1 不在一个量级（Pro 上 70%、Together 上 60% 左右），这其实反映了 Together 的题更难、用户更挑剔，不是模型退步了。
- 左下角同一根柱子上叠了两段：实心是 pass@1，虚线是 pass²（两次复跑都过才算），后者能更稳定地反映可靠性。
- 折线是 User Correction，能直接看出"能自己干完"和"要催几轮"是两件事。

这个范式上的转变其实非常重要：评测目标从"修好了没"扩展到"修得顺不顺"。

---

## 任务是怎么造出来的：把 11,260 段对话挑到 109 道

SWE-Together 的数据来自 4 个开源上游：DataClaw、Pi-staging、Hyperswitch、SWE-chat。原始会话规模如下：

| 上游源 | 描述 | 原始会话 | 最终入选项 |
|---|---|---:|---:|
| DataClaw | 32 个社区贡献数据集 | 2,228 | 29 |
| Pi-staging | 29 个来自 Pi staging 流水线的子集 | 2,397 | 23 |
| Hyperswitch | 一个生产支付代码库的真实轨迹 | 784 | 9 |
| SWE-chat | 跨多个 agent harness 的会话 | 5,851 | 48 |
| **合计** | | **11,260** | **109** |

11,260 挑到 109，转化率 0.97%。这个数字第一眼看上去有点反直觉，但整套流水线是按"宁可丢，不可错"设计的。

### 三步流水线

![Figure 2：Session-to-Task 构建流水线](https://arxiv.org/html/2606.29957v1/assets/session_collection_pipeline_v2.png)

*图 2：第一步是确定性规则过滤，第二步 LLM 评估可行性，第三步在沙箱里产出可执行任务包。每一阶段都把不符合"可复现"硬条件的会话直接砍掉。*

**Step 1：确定性 Eligibility Filtering**。这一阶段完全用规则筛：会话里得有真实多轮用户消息、得有 agent 的代码修改、得能识别工作仓库。私人仓库、没交互的、agent 只看不动的，全砍。这一步不要 LLM 调用，便宜而且稳定。

**Step 2：Viability Screening**。LLM 看着压缩过的 session 摘要（仓库元信息、消息/工具/编辑计数、用户消息样例、改过的文件路径、截断的 shell 命令），判断这个会话的"主要交付物"能不能在本地沙箱里复现。PR 管理、issue triage、部署操作、依赖私钥、依赖线上服务的，全部出局。

**Step 3：Task Construction**。Host 起一个隔离沙箱，把 normalized session + 生成 prompt 喂给一个 task-generation agent；它在沙箱里重新做一次更严格的仓库级筛选，clone 仓库到 pinned commit，识别本地 setup/test 命令，写出 task artifacts。最终每个任务包里有：原始 session 记录、用户首轮 instruction、pinned 执行环境、确定性验证 artifacts、task-specific 的 user-simulation prompt。

这个分层设计有个细节很关键：host 跟 sandbox 严格隔离，task construction 不依赖 host 机器的本地路径、已装工具链或者已经 apply 过的 fix，避免"评测里能过、换个机器就崩"。

---

## 关键设计：用户模拟器不是剧本，是有锚点的裁判

任务能复现只是第一关，更难的是怎么"重放"用户。在 SWE-Together 里这事儿被拆成了两条原则：

**原则一：trajectory-conditioned，不是 schedule**。模拟器每个 checkpoint 都看 agent 最新的活动摘要（agent activity、agent output、turn diff、timing）+ 自己的 memory（之前 turn 摘要、自己的历史决策和消息），而不是按固定步数触发。否则 agent 走快走慢都会失灵。

**原则二：anchored to the original session**。每个 task-specific 模拟器都从原始 session 里重建一份"用户画像"：这个人想要什么、有什么约束、什么时候会出手。这个 anchor 决定"现在该不该说话"。

看 Figure 3 就能看明白它的工作循环：

![Figure 4：用户模拟器在每个 checkpoint 看的上下文](https://arxiv.org/html/2606.29957v1/assets/user_sim_runtime_loop.png)

*图 3：模拟器每个决策点都看三类东西——固定 session anchors（persona、session analysis、original intents、message cap）、当前 turn summary（agent activity、agent output、turn diff、timing）、自己之前的 memory。最终从 5 个结构化动作里挑一个：no-op / question / redirect / new requirement / check external。*

它每次只做一个决策：

- no-op（默认）：让 agent 继续，不消耗原始 follow-up 配额
- question：问澄清
- redirect：把跑偏的轨迹拽回来
- new-requirement：补一个后续要求
- check-external：让 agent 去查外部产物

说实话这个设计比"读完所有原始 follow-up 按顺序念"好在哪？因为它解决了两种经典失败：

1. 固定回放：被测 agent 路径跟原 session 不一致时，原文照念会出现"该说的话在不该说的时刻出现"，评测结果就不可比。
2. 自由模拟：纯靠 LLM 即兴发挥，会逐渐 drift，session 之间的用户意图变得不可控。

SWE-Together 的做法相当于"用原始 session 当剧本大纲 + 用 agent 当前轨迹决定播到哪一幕"，这个折中是有讲究的。

---

## 评测协议：把"正确性"和"交互"拆成两套互相独立的指标

SWE-Together 的评测框架分为两大块——"任务对不对"和"模拟器行为如何"，刻意不合并。Figure 4 把这个分层画得很清楚。

![Figure 5：正确性 + 交互诊断的拆分](https://arxiv.org/html/2606.29957v1/assets/interaction_diagnostics_panel.png)

*图 4：最终 correctness 是 outcome metric，单独打分；下面分两个诊断面板：Intent Coverage 衡量模拟器是不是忠实保留了原始用户意图（simulator-fidelity diagnostic），User Correction 衡量 agent 引发的纠正性引导（agent-behavior diagnostic）。两个 diagnostic 不并入 score，只作为对比信号。*

### 正确性怎么打：两阶段 rubric judge

为什么不直接用 test pass/fail？SWE-Together 给的理由跟 OpenAI 弃用 SWE-bench Verified 是同一类：固定的 executable checks 容易出现两类错——要么太窄、卡死合理实现，要么太宽、把不要求的行为也算完成；而且单测看不全交互里后段才出现的需求。

他们的做法是 rubric judge，分两阶段：

- Phase 1 跑一次，从原始 reference patch 里推导出 task rubric——一组带权重的行为目标（目标 weight 归一化到 1）。这套 rubric 跟具体 agent 实现无关，行为化的。
- Phase 2 用同一套 rubric 对每个 agent 提交的仓库状态打分，每个目标二值 met / not met + 证据。

最终 score 就是一个加权和再 round 到两位小数：

$$\text{score} = \mathrm{round}\!\left(\sum_{g} w_g \cdot \mathbb{I}[g\ \text{met}],\ 2\right)$$

而且 host 端还有一个 validator 检查 rubric 覆盖、权重归一化、决策与 score 一致性——这套护栏是为了防止 judge 本身飘。

### 交互怎么量化：Intent Coverage 和 User Correction

**Intent Coverage** 衡量模拟器对原 session 用户意图的忠实度：

- 先对每道题把原始 session 拆成 atomic original intents（每个 intent 代表原用户的某条独立请求）
- 再对每次 replay，把模拟器发的消息和 intents 做匹配，分别算 recall 和 precision

$$\text{IntentCoverage} = \mathrm{round}\!\left(0.70 \cdot I_{\text{recall}} + 0.30 \cdot I_{\text{precision}},\ 2\right)$$

recall 权重大，因为漏掉一条原始 intent 会直接改变 agent 看到的题面。

**User Correction** 是这个 benchmark 最直白的"agent 能力指标"：对模拟器每条消息打多标签，统计两类纠正性行为：

- correction：直接说"你这是错的"——权重 1
- nudge：含蓄地"再想想"——权重 0.2

$$\text{UserCorrection} = N_{\text{correction}} + 0.2 \cdot N_{\text{nudge}}$$

剩下的 request、question、verification、workflow、approval、context 都不计入。也就是说，nudge 是个轻量级信号，只在你明确指出 agent 错在哪时权重才拉满。User Correction 先在 task 内对 replicate 取平均，再在 task 间取平均——保证每道题权重相等，不会被某些题里反复触发 nudge 的人为刷高。

---

## 实验设置与主结果

SWE-Together 在 109 个任务、7 个前沿模型、统一的 opencode harness 上跑，k=2 复跑。正确性用 judge score（[0,1]）做主要信号，阈值 τ=0.85。

四个 task-weighted 正确性指标：

| 指标 | 含义 | 严苛度 |
|---|---|---|
| pass@1 | 单次跑通的边际概率 | 较松 |
| SSR（Stable Solve Rate） | 同 task 先平均再判阈 | 中等 |
| pass² | 两次复跑都过才算 | 最严 |
| MeanJudge | 平均连续 judge score | 连续 |

主表（Table 2）如下：

| 排名 | 模型 | pass@1 | SSR | pass² | Mean judge | U-Corr | Tok./task | Min./task |
|:---:|---|---:|---:|---:|---:|---:|---:|---:|
| ★ | Reference (Oracle) | ~78% | ~78% | ~78% | 0.90 | — | — | — |
| 1 | **Claude Opus 4.8** | **63**% | **59**% | **52**% | **0.801** | **1.38** | 74.0k | 23.3 |
| 2 | GPT-5.5 | 58% | 55% | 48% | 0.763 | 1.59 | **29.9k** | **10.7** |
| 3 | Claude Opus 4.6 | 58% | 58% | 46% | 0.755 | 1.59 | 42.0k | 23.2 |
| 4 | GLM-5.2 | 55% | 48% | 42% | 0.735 | 1.53 | 41.7k | 24.5 |
| 5 | GLM-5.1 | 52% | 49% | 35% | 0.729 | 1.54 | 41.6k | 38.8 |
| 6 | DeepSeek-V4-Pro | 48% | 38% | 29% | 0.679 | 1.76 | 49.8k | 21.0 |
| 7 | MiniMax-2.7 | 40% | 34% | 26% | 0.630 | 2.17 | 43.4k | 36.2 |

几个一眼看出来的结论：

**第一名 Claude Opus 4.8 在四项正确性指标上全拿第一，并且 U-Corr 最低（1.38）**。它的 pass@1 仍然比 reference patch 整体低 15 个点左右——这条天花板后面单独说。

**GPT-5.5 在能力上排第二，但成本压得最低**：output tokens 29.9k，wall-clock 10.7 分钟。它的 SSR 比 Opus 4.6 略低，但 pass² 更高——这意味着它单次可能略弱，但跨次一致性更好。

**GLM-5.2 和 GLM-5.1 几乎打平**，SSR 都在 48%-49% 区间，但 5.2 的 pass@1（55% vs 52%）和 pass²（42% vs 35%）都更高，是这一档里"靠得住"的那个。

**DeepSeek-V4-Pro 和 MiniMax-2.7 垫底**，MiniMax-2.7 在四个正确性指标上全是最低，U-Corr 2.17 也是全场最高。

### 最值得拎出来说的一条结论

**更强的模型需要更少的纠正。** 论文里这张散点图（Figure 6）直接把这个相关摆出来：

![Figure 6：能力与 User Correction 的负相关](https://arxiv.org/html/2606.29957v1/assets/correction_subsets_v2.png)

*图 5：横轴是 User Correction（越往右越多），纵轴分别是 pass@1（左）和 SSR（右）。三套 marker：ALL=109 全部任务，ACTIVE=收到至少 1 次纠正的子集，HARD=mean judge < 0.85 的子集。三个子集下七模型都几乎落在一根直线上。*

数字上的 Pearson 相关系数：pass@1 是 -0.92，SSR 是 -0.84，mean judge 是 -0.93。你想想看，"不需要催"这件事跟"能解出来"几乎是一回事，能力越强、correction 越少这条规律在不同难度子集上都成立。

### 效率 vs 能力：两个轴几乎独立

![Figure 7：Stable solve rate vs 成本](https://arxiv.org/html/2606.29957v1/assets/stable_time_tokens.png)

*图 6：左图是 stable solve rate vs 每任务 wall-clock 分钟数，右图是 vs output+reasoning tokens。向左上为更优。*

- GPT-5.5 同时是能力第二 + 成本最低，是帕累托最优的那个
- Opus 4.8 能力强但 token 烧得最多（74.0k / 任务），wall-clock 倒还好（23.3 分钟）
- GLM-5.1 / MiniMax-2.7 是最慢的两个（38.8 / 36.2 分钟），但 capability 又不在前段——典型的"慢且不灵"
- DeepSeek-V4-Pro 能力垫底段但 wall-clock 还好，输出 token 偏多

这条挺值得琢磨的：成本和能力的相关性弱，意味着不同模型在不同维度的取舍空间很大，单一指标排榜单容易把这条信息压扁。

### Reference 怎么也没法满分

93 个有可提取 reference patch 的任务上，reference patch 自己的 mean judge 是 0.90，pass 率 78%。剩下那 22% 不是因为 reference 不对，而是因为：

1. **过程性需求**（约 35% 失败目标）：rubric 继承了原 session 的过程要求——先定位 root cause、回答后续追问、给用户解释改动——一个静态 patch 表达不了，所以 reference 自己也会被扣分
2. **提取噪声**：少数 reference patch 没完整覆盖原 session 跨多个 commit 的工作
3. **原 session 本身不完美**：比如修了几个 code path 但漏了 headline bug

也就是说，SW-Together 自己也意识到 rubric judge 不可能满分，这部分失分是从 session-construction 流程里继承来的。

### Intent Coverage 稳不稳：模拟器自己靠不靠谱

七个模型里，6 个的 Intent Coverage 都在 0.70-0.72 区间，recall 0.72-0.74，precision 0.66-0.72。GPT-5.5 略低（0.68）。这种窄波动说明模拟器在 agent 不同表现下都能保持稳定，差异更多来自"agent 主动解决掉了某些 intent、模拟器没机会重复表达"。

更狠的是 Turing-style 验证：让 4 个标注者在 paired trajectory 里盲猜哪条是真人用户、哪条是模拟器，52 道题 × 2 套共 156 对轨迹、312 次判断，模拟器的"图灵通过率"是 46%，95% CI 是 [40.5%, 51.6%]——落在 50% 区间内，标注者区分不出模拟和真人。

> 这个数字挺好。说明模拟器没"明显机器味"，给不同 agent 的模拟条件是一致的，差异主要来自 agent 自己。

---

## 与同类 benchmark 的定位

| Benchmark | Repo 级 | Agent-env 多轮 | 交互回放 | 真实任务源 | 真实用户会话 |
|---|:---:|:---:|:---:|:---:|:---:|
| SWE-bench family | ✓ | ✓ | ✗ | ✓ | ✗ |
| Terminal-Bench | ✗ | ✓ | ✗ | ✗ | ✗ |
| MINT / ConvCodeWorld | ✗ | ▲ | ✓ | ✗ | ✗ |
| CodeAssistBench | ✓ | ✓ | ✓ | ✓ | ✗ |
| RECODE-H / FronTalk | ▲ | ▲ | ✓ | ▲ | ✗ |
| BigCodeArena / CodeChat | ◆ | ◆ | ▲ | ✓ | ✓ |
| SWE-chat | ✓ | ✓ | ✗ | ✓ | ✓ |
| **SWE-Together** | **✓** | **✓** | **✓** | **✓** | **✓** |

SWE-Together 是这一列里唯一五个维度全打勾的：repo 级、agent-env 多轮、交互回放、真实任务源、真实用户会话都齐了。

值得提一嘴的是，同期 ColBench/SWEET-RL 也是 Meta FAIR 做的多轮编码评测，用 1 万+ 训练任务和 1 千+ 测试用例，但它的"用户"是从规则策略合成的，不是来自真实录制 session。SWE-Together 在这个方向上其实是往前推了一步——把合成 user 换成 recorded user，再把 user 的干预条件和 anchor 一起搬过来。

---

## 我的判断

这篇论文值得做严肃的工程化尝试。说几个具体的感受：

**1）"能力"和"用户负担"是同义词。** Pearson -0.92 不是一个普通数字——它意味着你只要看 User Correction 就能大致反推出 agent 的 capability。这意味着评估 agent 不一定非要堆更多新题，盯紧这条已经够用。

**2）评测的"分母"决定了它的天花板。** 109 题 / 0.97% 转化率，看出来是按"每题必须可复现"硬筛的，量上不去是这套范式的根本代价。但比起堆 500 题刷分，109 道高保真任务的可解释性更值钱。后续要扩，关键在 Step 2/3 的 LLM-as-judge 和 sandbox 编排能不能继续保证质量。

**3）rubric judge 的设计细节比看起来重要。** 两阶段（Phase 1 推 rubric、Phase 2 应用）+ 权重归一 + host-side validator——这套护栏是必要的，否则 judge 本身就会随着 prompt 漂移。OpenAI 在 SWE-bench Pro 上发现"verifier 自己不可靠"是同样的问题。

**4）用户模拟器的"anchored"设计是个工程亮点。** trajectory-conditioned + session-anchored 双原则，加上 no-op 占默认 action，让模拟器不会在 agent 已经做对的情况下强行刷存在感。这个设计如果做接口开放出来，对所有需要"模拟真实用户"的评测都值得复用。

**5）有几点要警惕：**

- 109 任务里接近一半（48）来自 SWE-chat 同一个上游，分布偏斜是否会让 simulator 在 SWE-chat 风格上更"顺"？
- 任务数少 + 复跑 k=2，pass² 这种严格指标的置信区间很宽。从 0.42 到 0.46 这种差距可能噪声就比信号大。
- Reference 自己 78% 这条天花板有点尴尬。如果不消掉"过程性需求"这种源于 session 构造的失分，agent 端的 progress 在 paper reader 那里会被低估。
- 论文没展开说怎么对付"用户在第一轮就把所有要求都讲完"那种 session——这种 session 模拟器基本没事可干，User Correction 会被人为压低。

**6）它到底处在什么位置？** 工程整合创新，不是底层突破。整套评测的核心方法（rubric judge、LLM-as-user-simulator、anchored simulation）在 ColBench、MINT、SWEET-RL 上都已经有原型，SWE-Together 真正的贡献是把"真实录制 session 作为 simulator 锚点"这一条做实，让多轮评测的可信度上了一个台阶。如果你在做代码辅助产品或者 agent 评测工具，这套 pipeline 的设计思想（特别是 simulator 那块）非常值得抄过来——比起另起炉灶造 benchmark，先把"用户在场"这件事做扎实，区分度立刻上来。

---

## 收尾

如果让我只记一件事，就是这条：**在静态榜单已经挤不动的当下，编码 agent 的下一个分水岭不在"能不能修好"，而在"修的过程里用户得操多少心"。**

从我自己看 agent benchmark 演进的经验，2024 年大家比"哪个 agent 能解更多题"，2025 年开始比"哪个 agent 在更难、更污染的题上扛得住"，到 2026 年这波（SWE-bench Pro、FrontierCode、SWE-Together、Claw-SWE-Bench）大家同时在比"评测协议本身靠不靠谱"——这是一个范式成熟的信号，说明单点榜单已经没意义，得看协议和题面的设计。

如果你的 agent 团队现在还在拿 SWE-bench Verified 当主战场，建议赶紧把 SWE-Together 这套指标体系接进自家 CI——尤其是 User Correction 这一条，几乎是免费的、可解释的、和能力高度共线的 sanity check。剩下的 109 题能跑多少就先跑多少，先拿到 baseline 再说。

---

觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。
