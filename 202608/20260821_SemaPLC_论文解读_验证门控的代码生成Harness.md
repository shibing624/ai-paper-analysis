# SemaPLC：把代码生成关进「验证门」

你有没有碰到过这种情况——模型拍着胸脯说"这段 PLC 代码已经搞定了"，你信了，结果部署到车间，流量低于阈值时设定值该变 500 不变，反而卡在了默认值 2500？

这不是段子。这正是 SemaPLC 论文里 Figure 2 的真实 case study：三个 baseline（LLM4PLC、AutoPLC、Agents4PLC）全部"通过"了编译，但放到真实 PLC runtime 上跑，行为与参考实现完全不符。论文核心想讲的就是一件事——**生成的代码只有在嵌入现有项目并通过传统工具链验证后才算数**。说到底，模型自我验收那一套，在工业代码生成里真不够用。

---

## 核心摘要

**痛点**：PLC 控制系统跑着工厂、电站、水处理厂，但目前的 LLM-PLC 系统普遍在"独立 POU"颗粒度上自评，代码能不能在真实项目里编进去、跑起来，没人系统测过。

**方案**：SemaPLC 是一个**验证门控**的 agent harness。它把生成—编辑—终止这个循环的最终判决权交给三个外部证据：specification audit、compilation、live runtime validation。模型可以说"我修好了"，但只有日志能证明它真修好了，才算交付。

**效果**：在 117 个独立 POU 任务上，七个模型严格验证通过率 72.6% mean，比最强 baseline（Agents4PLC 63.9%）高 8.8 个点。在 65 个项目级任务上，动态行为分数 52.2 vs 基线最高 31.4，几乎翻倍。

**判断**：这不是模型层的突破，是 harness 层的成熟。价值不在"调出了更好的 prompt"，而在于**把验收纪律做到工具日志那一层**——这一思想完全可以迁出 PLC 域，迁移到任何"模型自我验收不可信"的代码生成场景。

---

## 论文信息

- **标题**：SemaPLC: A Project-Grounded, Verification-Gated Agent Harness for PLC Code Generation
- **arXiv**：2608.18565（v1，2026-08-19 提交）
- **机构**：Midea AIRC、KUKA、SJTU、ZJU
- **作者**：Yanlun Tu、Huacan Wang、Ziyue Zhou、Jie Zhou、Ningyan Zhu、Ge Chen、Wangyi Chen、Tengfei Zhou、Yifan Zhou、Dasheng Yang、Xiaofeng Mou、Hui Zhang、Yi Xu（13 人）
- **代码**：<https://github.com/midea-ai/SemaPLC>

---

## 一、问题：能跑不等于能用

先把背景拉齐。PLC（Programmable Logic Controller）是工厂自动化的大脑，IEC 61131-3 是它的编程语言族，其中 Structured Text（ST）是文本化的那一个。LLM 生成 ST 代码不是新故事——LLM4PLC、AutoPLC、Agents4PLC 已经在这个赛道跑了一阵。**问题在于，他们评估生成质量的方式偏了**。

论文作者们做了三件挺狠的事：

1. 把 Agents4PLC 公布的 117 个独立 POU 任务逐个审计，发现 43 个任务的 oracle（"标准答案"）本身就有缺陷——错的常量、自相矛盾的性质、凭空发明的阈值、复制粘贴的重复。
2. 在 IEC 61131-3 的工业级 Spec2Control 语料里挑了 10 个工厂、65 个 section 任务，构建了"项目上下文"评估轨道。模型不是从零写一个 POU，而是要**改写**一个已经有模块库、变量声明、入口 harness、build 配置的真实项目。
3. 评分拆成三层：integrated compilation（能不能编进项目里）、static behavior（文本断言覆盖率）、dynamic behavior（部署到真实 runtime 后行为与参考实现比）。

第二层尤其关键。先看 Function track（117 任务）：

| 方法 | MiniMax-M2.7 | MiniMax-M3 | Qwen3.5-Plus | DS-V4-Flash | DS-V4-Pro | GLM-5.2 | GPT-5.5 | Mean |
|------|---------|---------|----------|----------|----------|---------|---------|---------|
| LLM4PLC | 22.2 | 15.4 | 13.7 | 41.0 | 43.6 | 30.8 | 44.4 | 30.2 |
| AutoPLC | 49.6 | 65.0 | 67.5 | 54.7 | 61.5 | 59.0 | 79.5 | 62.4 |
| Agents4PLC | 53.8 | 55.6 | 67.5 | 54.7 | 62.4 | 74.4 | 78.6 | 63.9 |
| SemaPLC bare | 39.3 | 60.7 | 62.4 | 34.2 | 55.6 | 63.2 | 71.8 | 55.3 |
| **SemaPLC** | **69.2** | **69.2** | **75.2** | **67.5** | **69.2** | **76.1** | **82.1** | **72.6** |

*表 1：Function track 严格验证通过率（%，分母 117；inconclusive/empty 算失败）*

**72.6% mean，所有七个模型上 SemaPLC 都是第一**，比 Agents4PLC 高 8.8 个点。表 1 还能看出一个有意思的细节：cross-model 离散度从 25~31 个点（baseline）缩到 14.6 个点（SemaPLC）。意思是 harness 把"模型能力差距"熨平了——差的模型装上 harness 也能逼近好模型裸跑的水平。这一点对工程落地很关键。

但 Function track 还不是故事的高潮。**真正判生死的，是项目级的 dynamic behavior**。

| 指标 | LLM4PLC | AutoPLC | Agents4PLC | **SemaPLC** |
|------|---------|---------|------------|---------|
| Integrated compilation | 58.7 | 81.5 | 71.2 | **89.4** |
| Static behavior | 75.7 | 74.0 | 71.7 | **81.6** |
| Dynamic behavior | 22.4 | 31.4 | 30.3 | **52.2** |

*表 2：Project-context track（65 任务，0-100 标度，mean 跨 7 个模型）*

**static behavior 大家都在 71~76 之间，相差不到 5 个点**；**dynamic behavior 拉开了——22.4 到 52.2，差 30 个点**。这正是 SemaPLC 想喊的那句话：**执行才是检验工业代码生成的唯一标准**。文本断言覆盖率高，只能说明模型写的代码"长得像"，部署到 runtime 上行为对不对，根本看不出来。

我在读到这一段的时候愣了一下。**PLC 这种连续运行的、有时序约束的、强安全相关的系统，模型说"看起来对"根本不算数**。传统 PLC 工程里"必须接上模拟器跑 case"这件事，AI 时代本来就不该被绕过。

---

## 二、方案：harness 不是 prompt，是纪律

SemaPLC 的核心是一个**验证门控**的 agent harness。它的设计哲学浓缩成三句话——

1. **Project-grounded generation**：生成的逻辑必须嵌入现有项目，复用变量、FB、接口、build 约定，不能从零造一个。
2. **Multi-source verification**：三种外部证据一视同仁——specification audit、compilation、live runtime validation。
3. **Verification-gated iteration**：agent 不准以"自我判断"终止；外部检查失败、缺失、含糊，都必须修或报告失败。

为了把这三句话落到工程细节，作者们搭了这么个结构（看图 1）：

![SemaPLC 整体架构图：左侧项目/任务 grounding，中部 agent core 的 Plan→Generate→Edit→Interpret→Repair 内环，右侧三层验证（Spec/Build/Run）通过 verification gate 决定是否接受交付](https://www.mulanai.com/fs/files/0821_5de0c97c_fig1.png)

*图 1：SemaPLC harness 整体架构。三种颜色箭头含义不同——蓝色是 artifact/控制流，绿色是 logged evidence/results，橙色是 repair/invalidation。*

我把图 1 拆开讲：

- **左侧**：控制需求 + 任务上下文（local POU 接口或已有 PLC 项目）→ Project/Task Grounding → Grounded Context Γ。这一步检索项目结构，定位相关模块，复用变量和 FB，编辑范围被限定住。
- **中上**：PLC Skill Library，文档化的领域知识，包括 IEC 61131-3 实践、FB 模式、timers/counters、安全/互锁、架构模式。这是"项目级"领域知识，不含任何 benchmark 答案。
- **中间**：Agent Core，跑在通用 event-driven tool-use 核心上（参考 Yao 2023 ReAct 那一脉），内部循环是 Plan→Generate→Edit→Interpret→Repair。注意它不含 PLC 特定逻辑，所有 PLC 能力通过外部组件引入。
- **右侧**：三层验证输出 logged results——SPEC 检查（信号、边界、优先级、互锁）、BUILD 检查（语法、类型、符号、接口）、RUN 检查（PLC Rack + Injected Inputs + Observed Traces）。
- **底部维修回路**：诊断反馈 → Repair → Edit 提交 → **Invalidate V**（旧判定全部作废）→ 重新跑三检 → Verification Gate 决定 Accept 还是 Report Failure。

### Algorithm 1：把验收纪律写成代码

最关键的不是图，是那段 14 行的伪代码。验证门控的循环长这样：

```text
Input:  requirement R; context X; required checks K with completion
        criteria; per-check retry limit r; interaction budget B
Output: implementation L with logged verification results V,
        or failure

 1: Γ ← Ground(X);  L ← Generate(R, Γ);  V ← ∅
 2: while budget B remains do
 3:   for all c ∈ K without a valid verdict in V do
 4:     (v, e_c) ← RunCheck(c, L)        {spec audit, compilation, runtime}
 5:     V[c] ← v  if log entry e_c confirms v, else unchecked  {earned claims}
 6:   end for
 7:   if V satisfies the completion criteria then return (L, V)  {accept}
 8:   F ← { c ∈ K | V[c] failed, retries(c) < r }
 9:   if F = ∅ then return failure with V  {no repairable check}
10:   L' ← Repair(L, { e_c | c ∈ F })
11:   if L' ≠ L then V ← ∅  {edit invalidation}
12:   L ← L'
13: end while
14: return failure with V  {budget exhausted}
```

*算法 1：SemaPLC 的验证门控生成循环*

**这段伪代码里有三件事值得抄下来**——任何严肃的 coding agent 都该照这个套路做：

**第一，edit invalidation（第 11 行）**。代码一动，所有先前的验证结果作废，下次循环全部重跑。翻译成大白话就是"verdict 绑定到 exact bytes"。这一条堵死了大量"先编出一个版本拿绿勾，再偷偷改"的投机行为。

**第二，earned claims（第 5 行）**。每一个"通过"必须对应一条 tool-call log。没有日志的自我声明，全部降级为 unchecked。这等于把模型从"评审席"踢下去，让工具日志当评审。

**第三，bounded retries + budget（第 2、8、9、14 行）**。每个 check 最多 r=2 轮重试，全局有交互预算 B，预算耗光或者没有可修项就报失败。**失败是可接受的交付物**，这是工程上很关键的态度——不是所有任务都能完成，但要诚实告诉调用方"完不成"。

### 三层验证到底在检什么

**Spec audit**（spec-review skill）：不是简单的关键字匹配。它按 defect class 走清单——contract extraction、coverage（每个被点名的设备/信号都有 FB 实例吗？每个 published variable 都有 driver 吗？）、boundary discipline（strict vs non-strict 比较显式选择了吗？）、global invariants、behavioral fidelity、scan-cycle semantics。收尾处还钉了六条领域公理：执行器在报警/互锁激活时不能留在危险态；互斥指令不能同时为真；故障传感器不能让控制器继续调节；故障/复位后 fail-safe 方向必须保持；工程值不能出校准范围；普通逻辑不能绕过 permissive。

**Compilation**：用 RuSTy 编译，结果包括 source line、error category、diagnostics。但有个细节——**只返回第一条诊断**，避免连锁重写让修复"牵一发动全身"。

**Live runtime validation**（benchmark-verify skill）：编译部署、初始化 runtime、注入场景输入、采样外部变量、跟参考 trace 对比。**评分场景是从隐藏参考实现里独立生成的，不是从任务描述自动派生**——这一条是防作弊的关键，否则模型可以"对题作答"。运行时采样的是 persistent assertions，不是每扫一次都录，这样既能覆盖瞬时事件又不会数据爆炸。

故障定位也有讲究：flat trace 指向 wiring，changing-but-wrong 指向块逻辑，late transition 指向 timer 或 edge detector。**正确的判定只来自外部可观测行为**，内部状态只能用来指导修复。

### MCP 工具层

整套工具用 Model Context Protocol（Anthropic 的那个）暴露，同时有 CLI 给程序化调用。我数了一下，附录 C 列了 16 个 MCP 工具（`plc_check`、`plc_compile`、`plc_detectIO`、`plc_upload`、`plc_buildAndRun`、`plc_start`、`plc_stop`、`plc_status`、`plc_getLogs`、`plc_readVariables`、`plc_forceVariables`、`plc_trace`、`plc_record`、`plc_waitFor`、`plc_verifyBehavior`、`plc_buildSimulation`），9 个 CLI 入口（`verify`、`compile`、`buildAndRun`、`detectIO`、`genModbusConfig`、`readVariables`、`force`、`trace`、`waitFor`、`genScene`、`status`、`serve`）。这些工具不是新发明，但**把"现场跑"和"对话式 agent"连起来**，才是 harness 的工程含量。

---

## 三、实验：动态行为才是真判官

### RQ1 函数级：所有模型都赢

表 1 已经看过，但有个**bare vs full** 的对比值得专门聊。bare 是 SemaPLC 剥掉 skills 和 tools 后的纯生成循环（同一个 backbone、同一套 prompt），full 是完整 harness。每个模型从 bare 到 full 的提升：

- MiniMax-M2.7：39.3 → 69.2（**+29.9**）
- DeepSeek-V4-Flash：34.2 → 67.5（**+33.3**）
- 强模型（GPT-5.5）：71.8 → 82.1（+10.3）
- cross-model 离散度：37.6 → 14.6

**弱模型获益最大**。原因是这些检查是外部的，模型自身知识对结果的影响被稀释。**harness 不是 prompt 调优，是模型无关的可靠性层**。这一点对工程选型意义重大——你不用纠结"用 GPT-5.5 还是 Qwen3.5"，装上 SemaPLC，差距会显著缩小。

### RQ2 项目级：动态分把大家拉开

回到表 2 那个关键观察——**static behavior 三个 baseline 都在 71~76 之间，dynamic behavior 从 22.4 一路拉到 52.2**。

论文里 Figure 2 的 case study 把这个观察具象化了。需求是：低流量（FT-701 < 50 kg/hr）→ SlaveSP = 500；变送器故障 → SlaveSP = 2500。

- **LLM4PLC**：编译失败（E007 缺分号）。
- **AutoPLC**：编译失败（E048 引用未定义 PT_701_High_Alarm）。
- **Agents4PLC**：编译通过，但行为错了——它的低流量 IF 在故障态下也会执行，**把 2500 覆盖成 500**。注意这是 Agents4PLC 自己的源代码逻辑能推出来的，但没人去推，因为基线只看编译过不过。
- **SemaPLC**：中间候选也错了（一个 ManOut 服务两个 case），但**runtime 验证把 30 kg/hr 注入后，期望 500 实际 2500，差异被记录**，触发定向 repair（按 cause 选 ManOut），再验证通过。

这段 case 让我**真正接受**了"执行验证"这件事的价值。文本断言就是看不出这种 case——故障态和正常态是互斥的分支，断言无法同时 hold 两个互斥条件；只有把代码真的跑起来，看时序行为，才能发现谁覆盖了谁。

### RQ3 层间消融：runtime 给的增益最大

论文里最耐看的是表 3——DeepSeek-V4-Flash 上的层间消融：

| 配置 | Compile | Static | Dynamic | Tokens | Reqs |
|------|---------|--------|---------|--------|------|
| None（仅生成） | 64.6 | 71.5 | **23.1** | 34k | 8.9 |
| + Spec | 70.8 | 74.0 | 30.3 | 60k | 14.6 |
| + Compile | 83.1 | 77.8 | 43.7 | 74k | 25.5 |
| + Runtime (full) | 84.6 | 78.0 | **54.1** | 129k | 47.8 |

*表 3：项目级验证层消融，DeepSeek-V4-Flash，65 任务*

**每一层都单调提升 dynamic score，从 23.1 到 54.1**。其中 compilation 层单次贡献最大（+13.4），因为编不过的程序在 runtime 直接 0 分；runtime 层次之（+10.4），但代价最贵（token 从 74k 涨到 129k，请求从 25.5 涨到 47.8）。**static score 从 71.5 到 78.0，几乎不动**——再次印证"文本断言看不到运行时问题"。

表 4 把 3590 个 scenario-port 检查拆成 outcome 分布：

- not built/port missing：33.4% → **3.3%**（compile 层 + runtime 层联手干掉）
- correct：23.1% → 54.1%
- wrong value (breach scenario)：1.2% → 17.3%（**注意这个值在涨**）

wrong value 涨这件事乍看像"变差了"，但作者解释得清楚——**这是一个覆盖度扩展的副产品**。一个检查要"能被判定为错"，必须先被运行到；只有 runtime 验证把场景跑起来后，那些"平时没被触发的边界 case"才暴露出来。**未暴露的 wrong 比暴露的 wrong 更危险**。这个观点我特别认同，工程上"我不知道我哪里错了"是最可怕的。

### Formal coverage 是真的不行

表 5 是对函数级形式化验证覆盖率的统计——用 PLCverif + nuXmv 这套工业标准流水线，看它在 1293 个交付程序上能给出多少"conclusive"判定（satisfied 或 violated）：

- 无 REAL/timer：75.7% conclusive
- 有 REAL：87.0% conclusive
- 有 timer（TON）：**0.0% conclusive**，174 个 property 全部 inconclusive

**timer 跨扫描周期，是形式化验证的传统盲区**。这说明光靠 PLCverif 这种成熟工具，是拿不到 timer 行为结论的。**所以 runtime validation 不是冗余，是补盲**。这一段对所有做工业形式化验证的团队都是个提醒——别迷信工具覆盖率，stateful timing 该跑还得跑。

### RQ4 交互成本：动态收益有代价

表 6 把成本摆出来：

| 轨道 | 方法 | Requests/task | Time/task (s) |
|------|------|---------------|---------------|
| Function | Agents4PLC | 6.3 (4.4–7.2) | 454 (241–688) |
| Function | SemaPLC | 6.5 (5.5–7.6) | **71** (41–156) |
| Project | Agents4PLC | 6.9 (6.8–7.0) | 344 (47–917) |
| Project | SemaPLC | **34.1** (16.4–60.4) | 347 (25–1380) |

*表 6：交互成本对比*

**Function track 上 SemaPLC 反而更便宜**——71s vs 454s，请求数几乎一样。原因是 Agents4PLC 每轮迭代都跑 PLCverif + nuXmv，模型检查本身耗时间；SemaPLC 把验证包成 tool call，编译 + runtime 是并行化友好的。

**Project track 上请求数从 6.9 涨到 34.1，跨模型离散度从 0.2 涨到 44**。这是架构差异——Agents4PLC 是固定多 agent 流水线，迭代次数封顶；SemaPLC 是开放式循环，每个 tool call 都由模型决定，verification gate 不让它停就一直转。强模型平均 16.4 次，弱模型能涨到 60.4 次。**动态收益是用模型交互买的**。

这一段对工程选型很关键——你愿意为 dynamic 50+ 的稳定度付出 5 倍的 token 吗？工业场景大概率愿意，但要在 SLA 里写清楚"交互预算耗光 = 报失败，不静默交付"。

---

## 四、批判性审视：这是不是过度包装的工程整合？

我读完第一遍的时候，**最大的疑问是：SemaPLC 到底有什么是新东西？**

工具——MCP、PLCverif、RuSTy、PLC runtime——全是现成的。
方法——RAG-grounding、agent loop、tool use——全是已知范式。
评估——benchmark 改造、oracle 审计——以前也有人做。

但仔细想，**新东西在纪律层**：

- "Agent 不准自评"——这条以前没人系统化强制。
- "Edit invalidation"——verdict 绑 bytes，不是"大方向对就行"。
- "Earned claims"——所有声明必须有 tool log 交叉验证。

这三件事单独看都不复杂，**合在一起构成了 SemaPLC 真正的贡献**——一套可审计的、可证伪的、可迁移的"模型生成—外部验证"契约。这是从"模型自我验收"到"外部证据驱动"的一次范式跃迁。

**但有几点需要泼冷水**：

**1. 任务规模还是小**。117 + 65 = 182 个任务，覆盖 10 个工厂。工业 PLC 项目动辄上千 POU、上万 IO，65 个 section 任务能代表的复杂度有限。论文自己也承认——"动态评分跑的是参考实现派生出来的限定场景集，未见运行条件下的行为未测量"。

**2. 评测泄漏的疑虑虽然被排除，但没排除干净**。Function track 上 oracle 是审计修复过的；Project track 上评分场景和参考 trace 隐藏；但 spec-review skill 在两个轨道上**都用 task spec 作为输入**。如果未来有同源团队的 spec 写法泄露，spec audit 实际上就接近于"对题作答"了。附录 E 的隔离设计做了大量工作，但工业语料里"语义模仿"很难穷尽排除。

**3. 强模型上优势收窄**。论文自己写了——GPT-5.5 上 SemaPLC 比 Agents4PLC 在 dynamic 上只领先 1.8 分，static 上还掉到 88.8 以下。**这意味着 harness 的边际价值随模型能力下降**。当模型本身足够强时，harness 是兜底；当模型弱时，harness 是救命稻草。**这是一个非常工程化的 trade-off，不是普适的"模型无关"**。

**4. 工业垂直域的局限**。PLC 域的强项是**有可形式化的语义、有 runtime 可以真跑、有项目结构可参考**。这三个条件同时满足的代码生成场景，在通用软件工程里其实不多。前端代码、Web 后端、嵌入式 C++——你想"runtime 验证"都不一定有 runtime。这不是说 harness 不能迁移，而是说迁移时**外部验证的"硬"程度**会显著下降，需要重新设计。

**5. 真正未解的问题是 prompt 级自适应**。SemaPLC 的 harness 是固定的，skill library 是人工编写的。如果能让 harness 自己根据任务类型选择验证层组合（比如低风险任务跳过 runtime），可以进一步降本。这一步论文没做。

---

## 五、对 Coding Agent 的迁移启发

虽然 SemaPLC 是在 PLC 域做的，但**"外部验证门控 vs 模型自我验收"这一原则对所有 coding agent 都有价值**。我把它抽象成三条规则，给设计 harness 的人参考：

**规则 1：每个 verdict 绑 bytes，不绑 intent**。代码一改，验证全部重跑。任何"看起来还是对的所以保留旧判定"的捷径都要拒绝。SemaPLC 第 11 行的 `if L' ≠ L then V ← ∅` 是这一条的最简表达。

**规则 2：声明必带证据，证据必带日志**。模型说"我修好了"不算，必须有 tool log 能交叉验证。这是 SemaPLC 第 5 行的 `earned claims`。落地到通用 coding agent 上，就是每个"pass"必须对应一个具体的 test run / build success / type check result，没有就降级为 unchecked。

**规则 3：失败是可交付物**。SemaPLC 第 9、14 行明明白白把"报失败"作为正常 return path。**这比"无限重试到超时"安全得多**。一个能诚实说"我修不好" 的 agent，比一个为了给绿勾而走偏的 agent 可信一万倍。工业场景尤其如此。

**规则 4：评估要分 layer，别只看一个分数**。SemaPLC 把 compile / static / dynamic 拆开评。**如果只看 static score，你会以为所有方法差不多**；dynamic 一拉开，立刻看出谁在"打榜"，谁在"真做工程"。这条对所有 benchmark 设计都是当头棒喝——**没有执行的 benchmark 都不算 benchmark**。

把这四条对照一下 Darwin Godel Machine、AutoDesign、Agentic ESOpt 这类**"AI 改造 AI 自身代码"的 agent 范式**，你会发现它们大部分还在"self-verify"层打转——模型自己读自己代码、自己给自己打分。**如果有人能把这套 verification gate 迁到那类系统上，让 LLM 在修改自己的代码时必须过编译 + 测试 + 形式化 property**——那才是 harness 层的下一步。

---

## 六、收尾

SemaPLC 不是"AI 取代工程师"的宣言，它做的是**让 AI 的产出在工程师的工具链里能站住脚**。它把工程纪律写进 harness，把验收权交给工具日志，把"能跑"和"能用"切开打分。

最让我在意的不是 72.6% 或 52.2 这些数字，而是 Figure 2 那个 500 vs 2500 的差异——**模型能写对代码，但分不清"低流量"和"故障"哪个优先级高**。这种语义错配，文本断言看不出来，编译器看不出来，只有把代码灌进 runtime，让真实时序跑起来，才知道答案。

这才是"verification-gated"真正的含义——**门不是为了难住模型，是为了把"运行时不行为"这件事从系统里彻底挤出去**。

工业代码生成的下一个台阶，可能就长这样。

---

## 参考文献

- Liu et al. 2026. Agents4PLC: Automating closed-loop PLC code generation and verification in industrial control systems using LLM-based agents. IEEE TSE 52(5).
- Yang et al. 2025. AutoPLC: Generating vendor-aware structured text for PLCs. ASE 2025.
- Fakih et al. 2024. LLM4PLC: Harnessing large language models for verifiable programming of PLCs. ICSE-SEIP 2024.
- Koziolek et al. 2026. Spec2Control: Automating PLC/DCS control-logic engineering from natural language requirements with LLMs. ICSE-SEIP 2026.
- Darvas et al. 2015. PLCverif: A tool to verify PLC programs based on model checking techniques. ICALEPCS 2015.
- Yao et al. 2023. ReAct: Synergizing reasoning and acting in language models. ICLR 2023.
- Chen et al. 2024. Teaching large language models to self-debug. ICLR 2024.
- Anthropic. 2026. Introducing the Model Context Protocol.

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。*
