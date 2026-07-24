# 少改几行提示词，就能让 Claude Opus 的代码任务胜率反超 ultracode

你有没有过这种体验：在 Codex / Claude Code 上调教自家 Agent，琢磨了一晚上 prompt，把"先列计划再写代码"改成"先拆子任务再汇总"，结果跑出来的任务完成质量忽上忽下，成本还翻了一倍？

更扎心的是——你花了几小时调的 prompt，可能还不如官方直接开 `xhigh` 或者 `ultracode` 推理档位来得稳定。

Sakana AI 和 UC Berkeley 联合发表的 [Recursive Harness Self-Improvement](https://arxiv.org/abs/2607.15524)（arXiv 2607.15524，下文简称 **RHI**）就是冲着这个问题去的。它的核心命题很直接：

> 对于一个固定的 coding agent，**只改 prompt 里的 harness 描述，几轮迭代之后就能让它的胜率反超同款模型加 `xhigh` / `ultracode` / `max` 这些高推理档位，而且总成本还能砍掉最多 60%**。

听起来像在卖焦虑+许愿，对吧？我第一反应也是："不可能吧，提示词优化收益一般 5% 顶天了。" 但看了论文 30 个跨域任务的实验和组件级消融之后，我承认它提出的"轨迹局部自比较"思路确实打中了一个被忽略的优化维度——**信息路由（information flow）**，而不是生成更长的推理。

先抛个一句话的判断：这篇论文做的是**第一半**的回路（harness 怎么改），不是**第二半**（改完 harness 产生的高质量 trace 怎么回流训练下一代模型）。这意味着它在系统层面是个"效率优化器"，不是"自我进化算法"。定位要拎清楚，再看细节。

---

## 🎯 一句话看完整篇

RHI 把"agent 的 harness"当成一个**纯文本对象**来编辑：用当前 harness 的输出和上一版输出做一次 LLM-as-a-judge pairwise 比较，把偏好塞进历史，再用 LLM 改写下一次 harness——每轮 1 次执行 + 1 次评估，开销恒定为 $O(1)$，比 Meta-Harness / Self-Harness 这类需要维护候选集（$\Theta(m^2)$）的方案便宜几个数量级。3 轮以内，`opus-4.8-high` 加 RHI harness 就能在 30 个跨量化金融 / 机器人 / 药物 ML 任务上击败自家 `opus-4.8-ultracode`，且推理成本降 60%、cache 读写降 64%。

---

## 📖 论文信息

| 项 | 内容 |
|---|---|
| 标题 | Recursive Harness Self-Improvement |
| 作者 | Hyunin Lee, Jinglue Xu, Jeffrey Seely, Donghyun Lee, Matei Zaharia, Yujin Tang |
| 机构 | Sakana AI, UC Berkeley |
| arXiv | [2607.15524](https://arxiv.org/abs/2607.15524) |
| 提交时间 | 2026-07-17 |
| 一作背景 | Hyunin Lee 是 UC Berkeley 博士生，实习在 Sakana AI（脚注里说明） |

注：Matei Zaharia 是 Databricks 创始人，框架作者。Yujin Tang 是 Sakana AI 的研究人员。这个组合既懂学术也懂系统。

---

## 🧠 问题动机：模型-系统正在"共同进化"，但没人认真优化过系统这一侧

最近一年 agentic coding 的进展越来越多地来自**模型（model）和系统（harness）的共演化**，而不是单纯把模型训大。一个直观的观察是：

- 模型升级 → 用户能写出更强的 harness
- 更强的 harness → 生成更结构化、信息密度更高的执行 trace
- 更高质量的 trace → 可以当作下一代模型的 post-training data

这就形成了一个**递归反馈环**。在 Sakana AI 的视角里，**trace 质量才是这个环的真正瓶颈**——你 trace 里多堆点无关上下文，模型就要多烧 token 在过滤噪声上；trace 缺了关键证据，下游训练出来的模型就缺一条信息通路。

但优化 harness 这件事，在工业界有两条路都不好走：

1. **改 provider-built harness**（OpenAI / Anthropic 自家的 agent loop）：要兼顾所有用户、任务、场景，更新一次是几个团队几个月的工作量。你想给自己私有任务调个 harness，等官方发版是不现实的。
2. **暴力搜索 user-constructed harness**（Meta-Harness、Self-Harness、TTHE、AlphaEvolve 这些）：每多一个候选，就要多跑一次完整的 black-box agent 执行 + 评估，开销是 $\Theta(m^2)$。当 $m$ 是个 30 任务的 batch 时，单位任务优化预算根本撑不住。

RHI 切中的是第三条路：**轨迹局部自比较（trajectory-local self-comparison）**——只看自己和上一版差多少，不看整个候选池。在 30 个任务上，它声称只要迭代 1-2 轮就能把 harness 改到接近 $\Theta(m^2)$ 暴力搜索的水平。

这个设定我之前没见有人把"成对比较"做到如此极致——`Self-Harness` 至少还要维护一条回归测试，`Meta-Harness` 要维护 Pareto 前沿。RHI 把所有这些都砍掉了，只留一个"上次 vs 这次"的偏好对，再让 LLM 当优化器。

坦率说，我看到这里的时候心里嘀咕了一句："这跟 sequential refinement + LLM judge 有啥本质区别？" 答案是——本质区别不在"对比"这个动作本身，而在**对比信号用在哪里**。RHI 的 pairwise feedback 不是用来选 best-of-N 的，而是用来**改 harness 这个 meta-object** 的。这是一个"meta-level learning"的姿势。下一节展开说。

---

## 🏗️ 方法核心：把 harness 当成"被编辑的 prompt"

### 3.1 算法一眼看明白

下图就是 RHI 的全部精髓，5 步走一圈：

![RHI 算法流程图](https://www.mulanai.com/fs/files/0721_2aef1e64_RHI_algo.png)

*图1：RHI 的 5 步循环。① 把 task + harness 拼成 prompt 喂给 coding agent；② agent 产出一个 repository（output[i]）；③ LLM 评估器把 output[i] 和 output[i-1] 做 pairwise 比较得到 preference feedback；④ 把这个 feedback 存进 self-comparison history；⑤ harness optimizer 用历史改写下一次 harness。*

关键的几点：

- **每轮开销恒定**：1 次 agent 执行 + 1 次 LLM pairwise 评估，$O(1)$，跟候选集大小 $m$ 无关。论文里的对比表（Table 1）把这个账算得清清楚楚：

| Objective | N_trace | N_pair | Total Cost |
|---|---|---|---|
| Ideal 目标（vs 全体参考分布） | $M$ | $\binom{M}{2}$ | $\Theta(M^2)$ |
| 有限候选搜索（Meta-Harness/TTHE 类） | $m$ | $\binom{m}{2}$ | $\Theta(m^2)$ |
| **RHI 轨迹局部目标** | **1** | **1** | $\Theta(1)$ |

- **不存 loss、没梯度**：harness 空间是离散的（文本），所以 $D_x^{(i)}$ 不是真的梯度，而是"语义上的动量信号"。论文里原话是 *momentum-semantic signal*——我第一次看到时也皱眉，但仔细想想，跟 GEPA 的 Pareto frontier 维护、TextGrad 的"自然语言梯度"在精神上是一脉相承的，区别是 RHI 的信号最瘦。

- **评估器和优化器都没有"看到"评估标准**：评估器 $\mathcal{L}_{\text{eval}}$ 知道 $x_{\text{eval}}$（要评哪些维度），但 harness optimizer $\mathcal{L}_{\text{harness}}$ 拿到的只有当前 harness 和偏好历史。这意味着 $x_{\text{eval}}$ 是**通过偏好信号间接传递的**，这是一个相当聪明的设计——避免了"优化器直接对着评估器走捷径"的潜在问题。

### 3.2 Harness 本身的结构：roles、instructions、contracts、hops 四件套

要把 harness 当成 prompt 来改，首先得知道它长啥样。论文把 harness 拆成两个一级组件、四个二级组件：

![Harness 结构分解](https://www.mulanai.com/fs/files/0721_fa1395e1_RHSI_har.png)

*图2：Harness 分解。Agent Design = role + instruction（每个 agent 的身份和行为描述）；Agent Workflow = contract + hop（子 agent 和 orchestrator 之间传什么、按什么顺序交互）。RHI 的优化器在系统 prompt 里被显式要求"重点打磨 contract 和 hop"，不重点改 role。*

为啥重点打磨 contract 和 hop？论文给了一个非常优雅的直觉——**contract 决定子 agent 之间传什么信息，hop 决定这些信息按什么顺序流**。如果把"全量同步历史"比作 dense attention（每个 token 看见所有 token），那 task-specific contract 就相当于"学了一个 task-dependent 的 attention sparsity mask"。这跟 Longformer / BigBird 的 sparse attention 是一个精神，但发生在 prompt 层面。

更有意思的是，作者声称当基础模型足够强时，agent 往往不真的"开多个子 agent"，而是在**单 agent 内部做多 persona 推理**。也就是 harness 上写的多 agent 拓扑，最后落地时是一个 model 内化了多 persona 的协作——这是更现实的工程场景，system 层的 multi-agent 描述更多起到"信号引导"作用。

这个观察让我对 harness 的认识更深一层：**它不是系统拓扑的源码，而是一种"对模型行为的软提示"**。把它当 prompt 改，跟传统 DSPy 那种优化"单个 prompt 模板"是一回事，只不过优化的对象规模更大、结构化更强。

### 3.3 为什么"几次迭代"就够？

论文里有个我差点没注意的细节：**$s_i < \epsilon$ 就 break**（Algorithm 1 第 260 行）。也就是 RHI 会监控"这一轮 harness 相比上一轮的胜率增量"，如果低于阈值就停。

这意味着 RHI 的"few-iteration"是**自适应的**——不是硬编码迭代 3 次就完事，而是看边际收益。论文里 sonnet-4.6 跑到了 5 次、opus-4.8 跑到了 3 次就饱和。

这个设计的好处是：**它把"够不够好"这件事外包给了 pairwise judge**。你不告诉它"几轮收敛"，让偏好信号自己决定何时停。这跟 gradient descent 的 early stopping 在精神上类似。

---

## 🧪 实验结果：3 轮打败 ultracode，成本砍六成

### 4.1 主实验：所有 base model 上 RHI 都赢

先看全家福。这张是论文首页图，6 个 panel 把 sonnet-4.6 / opus-4.7 / opus-4.8 三个 base model vs 各自最强 TTS（test-time scaling）档位的对比全画在一张上：

![RHI vs TTS 全家福](https://www.mulanai.com/fs/files/0721_fe400800_frontpag.png)

*图3：RHI 在三个 base model 上对不同 TTS baseline 的成对胜率（绿色条）。柱子高度=30 个任务中赢的个数（max 30），灰色背景=对手赢+平局。橙色线=归一化成本。皇冠标记的是赢>19.5 的"显著胜利"。上排 sonnet-4.6 vs max、opus-4.7 vs xhigh / vs max；下排 opus-4.8 vs xhigh / vs max / vs ultracode。*

**关键观察**（左下角 sonnet-4.6 vs max）：

- `H[0]`（也就是初始 harness）只能赢约 11/30，明显弱于 `sonnet-4.6-max`（约 16 赢 + 14 输）
- `H[2]`（2 轮 RHI 之后）反超，赢 20/30 输 10/30
- 之后 `H[3]`、`H[4]` 保持优势，但胜率不再单调上升

**最戏剧化的是右下角 opus-4.8 vs ultracode**：

- `H[2]` 赢 23/30 输 7/30
- 同时成本只有 ultracode 的 40%（橙色线下沉）
- 论文里那个 "↓60%" 标签就是这里

这意味着：**对于最强 base model（opus-4.8），加 RHI 反而比开内置的 ultracode 多 agent 编排更有效，而且便宜一大截**。这个结论我反复读了几遍确认不是误读——它不是"和 ultracode 打平"，是"显著超过"。

### 4.2 单模型细节：opus-4.8 跑完整段

为防止主图太密看不清细节，论文里给了 opus-4.8 的完整 0/1/2 轮迭代 vs 三个 baseline：

![opus-4.8 + RHI H_2 的对决明细](https://www.mulanai.com/fs/files/0721_de774f69_opus48hi.png)

*图4：opus-4.8-high + RHI 改进的 H[2] vs opus-4.8 的 xhigh/ultracode/max。左图是 gpt-5.5-max 当 judge，右图是 opus-4.8-xhigh 当 judge。绿色=win，红色=loss，灰=平局。两种 judge 下都是绿色大幅领先。*

这张图说明了 RHI 的结论**对 judge 不敏感**——gpt-5.5-max（不同模型家族）和 opus-4.8-xhigh（同家族高推理）这两个 judge 给了基本一致的偏好。这点很重要，因为"用 GPT 评 Claude"经常被怀疑有家族偏见，RHI 至少在这两个 judge 上没翻车。

### 4.3 成本明细：cache 读写降一半以上

性能好不是白来的——RHI 在资源维度也站得住：

![opus-4.8 归一化资源消耗](https://www.mulanai.com/fs/files/0721_a78f17b2_normaliz.png)

*图5：opus-4.8 各配置的归一化 cost / output / cache 读写（以 opus-4.8-high = 1 为基准）。三组柱状分别对应 xhigh、max、ultracode 三个 baseline，以及 H[0]、H[1]、H[2] 三轮 RHI 改进。绿色虚线是 ultracode 高度，橙色虚线是 max 高度。*

**几个硬数据**：

- H[2] 归一化 cost = 1.69，ultracode = 4.15 → **省 60%**
- H[2] 归一化 output token = 1.65，ultracode = 3.57 → 省 54%
- H[2] 归一化 cache 读写 = 1.69，ultracode = 4.74 → **省 64%**

注意一个反直觉的点：**output token 没有增长**。常规"加 multi-agent"会带来协调 overhead（多了子 agent 各自的解释、评判、投票），output token 会涨。但 RHI 的 H[2] output token 跟初始 H[0] 几乎一样（1.42→1.81）——这恰好说明 RHI 不是靠"多生成长推理"赢的。

### 4.4 sonnet-4.6 那条线：弱模型也能拉

为了证明结论不挑 base model，看一下 sonnet-4.6 的结果：

![sonnet-4.6 性能](https://www.mulanai.com/fs/files/0721_81d7ee9e_sonnet46.png)

*图6：sonnet-4.6-high + H[i]（i=0..4）vs sonnet-4.6-max 在 30 个任务上的成对胜率。两种 judge（gpt-5.5-max / opus-4.7-xhigh）一致显示 H[2] 之后稳定胜出。*

![sonnet-4.6 资源](https://www.mulanai.com/fs/files/0721_91b58cc3_sonnet46.png)

*图7：sonnet-4.6 归一化资源。H[2] cost=2.38，max cost=2.56，省 7%；H[2] cache 读写=3.31，max=4.91，省 33%。*

sonnet-4.6 的节省幅度比 opus-4.8 小（cost 只省 7%），但**胜率提升在 2-3 轮之后就稳定**。这暗示 RHI 的性价比对弱模型反而更敏感——你用 sonnet-4.6 加 RHI 拿到的 7% 成本节省 × 60% cache 节省，仍然是实打实的优化。

### 4.5 关键 claim 的数字总结

把三个 base model 拉成表：

| Base model | TTS baseline | RHI 轮次 | 胜率（win/30） | cost 变化 | cache 变化 |
|---|---|---|---|---|---|
| sonnet-4.6 | max | H[2] | 20 / 30 | -7% | -33% |
| opus-4.7 | xhigh | H[1] | 19-22 / 30 | -18% | -37% |
| opus-4.8 | ultracode | H[2] | 23 / 30 | **-60%** | **-64%** |
| opus-4.8 | xhigh | H[2] | 23 / 30 | ≈ 同 high | 持平 |
| opus-4.8 | max | H[2] | 23 / 30 | -23% | -32% |

数据来源：Figure 1 + Figure 5（opus-4.8）+ Figure 7（sonnet-4.6）+ Section 5.2/5.3 文字描述（opus-4.7）。

---

## 🔬 消融：为什么有效？——信息路由，而非长推理

主实验回答了"行不行"，消融回答"为什么行"。这部分才是我觉得 RHI 真正有洞察的地方。

### 5.1 假设 1：是不是"模型被换强了"？

RHI 作者做了一个诚实的检查：用 sonnet-4.6 + RHI（H[i]）去比 opus-4.7 本身（不加 RHI）。

![sonnet-4.6 + RHI vs opus-4.7](https://www.mulanai.com/fs/files/0721_d9de53c4_sonnet46.png)

*图8：sonnet-4.6-high + H[i]（i=0..4）vs opus-4.7-high 和 opus-4.7-xhigh。曲线显示 RHI 在 sonnet-4.6 上的胜率随迭代提升，但始终没追平 opus-4.7。*

**结论**：RHI 是 **train-time scaling 的补充而非替代**。它能拉高同款 base model 的天花板，但没法帮你跳过 base model 的能力鸿沟。

这是必须坦白的一刀切版本：想用 sonnet-4.6 + RHI 干翻 opus-4.8？做不到。

### 5.2 假设 2：Harness 到底改了什么？

这是消融最有意思的部分。论文用 t-SNE / UMAP 把整个 harness 文本投影到 2D，看 H[0] → H[1] → H[2] → H[3] → H[4] 之间的语义漂移：

![整 harness 的语义漂移](https://www.mulanai.com/fs/files/0721_2b36c8d0_wholehar.png)

*图9：sonnet-4.6-high 30 个任务在 5 轮 RHI 下的全 harness 嵌入投影（UMAP / t-SNE × 两种 embedding 模型）。颜色=任务域，黑色描边=初始 H[0]，灰白=后续 H[1..4]。H[0] 跟后续版本在所有 embedding × 投影组合里都明显分簇。*

**第一个观察**：H[0] 跟所有 H[i≥1] 在所有 embedding 模型（OpenAI text-embedding-3-large / all-mpnet-base-v2）和投影方法（UMAP / t-SNE）下都明显分离。这说明 RHI 真的在改 harness，不是在做形式微调。

**第二个观察**：H[0]→H[1] 的余弦相似度 0.82 是最低的，之后 H[1]→H[2]→H[3]→H[4] 分别是 0.97/0.98/0.99——**第一轮改动最大，后续都是微调**。这跟主实验的胜率曲线一致：H[0]→H[1] 性能涨得最多。

### 5.3 组件级别：谁在变？

把 harness 拆成 4 个组件，分别看每一类在迭代中怎么变：

![四个组件的 UMAP/t-SNE 投影](https://www.mulanai.com/fs/files/0721_ed000dd1_fig4_4_1.jpg)

*图10：sonnet-4.6-high 在量化金融 30 个任务上的 4 组件嵌入（role / instruction / contract / hop × 2 种 embedding × 2 种投影）。颜色=任务，点形=RHI 迭代轮次。注意 contract 和 hop 的同任务点连成线（同一任务不同轮的演化），role 和 instruction 没有连线。*

**关键观察**：

- **contract 聚类最清楚**——同一任务在不同 RHI 轮次下的 contract embedding 形成清晰的"任务专属簇"
- **hop 也有类似但稍弱的模式**
- **role 几乎没有任务聚类**——所有任务的 role 混在一起
- **instruction 中等程度聚类**

**这跟论文的核心论点直接对应**：RHI 优化器在系统 prompt 里被要求"重点打磨 contract 和 hop"，所以这两类组件进化得最显著。

接下来用余弦相似度定量看：

![组件之间的余弦相似度演化](https://www.mulanai.com/fs/files/0721_e375a69b_fig26_a.png)

*图11：4 组件在 RHI 迭代之间的 cross_cos 余弦相似度（sonnet-4.6-high）。**contract 最先达到 0.6+ 高相似度**（从 0.48 起步，第二轮就到 0.66），其他组件的稳定化慢得多。role 全程在 0.3 附近。*

**数值上的硬证据**：

- contract 跨轮相似度 0.48 → 0.66 → 0.69 → 0.72（快速稳定）
- hop 0.48 → 0.52 → 0.52 → 0.50（中等稳定）
- instruction 0.42 → 0.66 → 0.66 → 0.66（迭代 2 之后稳定）
- role 0.28 → 0.32 → 0.32 → 0.32（全程变化小）

**含义**：RHI 学到的是**任务专属的 contract**（子 agent 之间传什么信息），这跟"长推理"完全是两码事。增长的不是输出长度，而是**信息流的结构**。

### 5.4 隐式目标函数：信息论的解释

消融的最后一段是最理论化也最容易被跳过的——论文提出了一个**信息论隐式目标**作为对 RHI 行为的解释：

![RHI 隐式目标示意图](https://www.mulanai.com/fs/files/0721_f75a9687_RHIobjec.png)

*图12：RHI 隐式目标。L_harness 优化器同时收到两类信号：f_ext（外部可控，由 harness prompt 决定，倾向于提高 contract/hop 的任务信息量）和 f_int（内部模型依赖，作为"功能特化引导"，抑制跨组件冗余）。*

公式：

$$J(g_i) = \underbrace{\sum_{\texttt{hc}\in\mathcal{C}_{\text{ext}}} \frac{1}{K_{Xi}^{\texttt{hc}}} \sum_k I(z_{Xk}^{\texttt{hc},(i)};X)}_{f_{\text{ext}}} - \beta \underbrace{\text{TC}\left(\{z_{Xk}^{\texttt{hc},(i)}\} \mid X\right)}_{f_{\text{int}}}$$

翻译成人话：

- **$f_{\text{ext}}$**：让被 prompt 强调的组件（contract、hop）跟任务 $X$ 的互信息尽量大
- **$f_{\text{int}}$**：让所有组件在"已知任务"之后的总相关（total correlation）尽量小——也就是说**别让四个组件都说同一件事**

论文里把 $f_{\text{int}}$ 类比成"classifier-free guidance"那种"从无条件和有条件生成中合成引导信号"的做法——这个类比有点飘，但它在数值上确实对应一个观察：role/instruction/contract/hop 在条件于任务之后的冗余度（task-conditional TC）从 4.84 nats 单调下降到 3.63 nats（text-embedding-3-large），从 3.51 降到 2.62 nats（mpnet）。

**这个目标函数并不是 RHI 真的在优化的东西**——论文明确说 RHI 是 black-box search，optimizer LLM 没看 $x_{\text{eval}}$。作者是说：**RHI 观测到的轨迹恰好跟这个信息论目标一致**。这是一个 **a-posteriori 的解释**，不是 a-priori 的设计。这一点在 Section 6.3 的开头就明说了。

这让我对论文的诚实度加了分——很多人写"我们提出了一个 X 目标函数"其实是把事后归纳当成了事前设计，RHI 没这么干。

### 5.5 跟同行的差异

为了防止你怀疑"这不就是 GEPA / Self-Harness / Meta-Harness 的另一个名字"，论文在 Related Work 里正面区分了：

| 方法 | 优化对象 | 信号来源 | 候选集 |
|---|---|---|---|
| Meta-Harness | 可执行 harness 代码 | source code + 分数 + trace | Pareto 前沿 |
| Self-Harness | harness edits | 失败挖掘 + 回归测试 | 多个候选分支 |
| TTHE | 批量级 harness | 执行反馈（健康度/通过率） | 群体演化 |
| GEPA | 单一 prompt | 执行轨迹 + 反思 | Pareto frontier |
| **RHI** | **harness 文本** | **自身前后 pairwise 偏好** | **$m=1$** |

RHI 的核心差异是 **每轮只跟自己比**，不开候选池。代价是单步信号噪声大，作者用"累积偏好历史"（$\mathcal{D}_x^{(i)}$）来补偿——本质是"用 sequential 多次 noisy 比较替代 one-shot 干净比较"。这是 RHI 的核心 trade-off：**用时间换覆盖度**。

---

## 🤔 我的判断：定位是"系统层效率优化器"，不是"自我进化"

读完整篇后我的判断分三块：

### 6.1 这篇论文最值钱的地方

不是"60% cost reduction"这个数字，而是 **"把 harness 当 prompt 改"这个视角**。

我之前在 agent 调优上的认知是：想提升 agent 能力，要么换强模型，要么加 inference 算力，要么维护一组 workflow 候选。RHI 给我打开了一个新的角度——**你写给模型的那段 harness 描述，本身就是一个可学习对象**，而且改它的开销可以做到 $O(1)$。

这跟 LangChain/LlamaIndex 那种"workflow as code"的思路完全不同。RHI 是 **workflow as text**——不编译成代码，纯粹靠 prompt 引导模型做对的事。这意味着：

- 任何支持 prompt 编辑的 black-box agent 都能用
- 不需要 framework 支持
- 失败可观察（harness 文本就是 debugging 入口）

### 6.2 我对实验的几个保留

**第一个保留：benchmark 是 LLM 生成的"任务"。** 30 个任务是用 LLM 把 Citadel、Amazon、Genentech 的招聘 JD 改写来的"研究风格任务"。这是个聪明的工程做法，但你要问自己：**这些任务的"难度"分布跟用户实际场景匹配吗？** 我自己平时让 Claude Code 干的事远没有这么"结构化"（没有 standardized deliverable，没有完整 metrics.json）。RHI 在这种场景下能不能复现，论文没给数据。

**第二个保留：LLM-as-a-judge 的偏好信号可能有偏差。** 论文承认用了 gpt-5.5-max 和 opus-4.7/4.8-xhigh 当 judge，并且在不同 judge 上都看到了类似的 RHI 优势，这点我比较服。但 LLM judge 本身对"格式完整、报告漂亮"这种 deliverable 表面特征是有偏好的——RHI 是不是只是"训出了一个更会写 report 的 harness"？论文没拆开来研究。

**第三个保留：sonnet-4.6 vs opus-4.7 那一刀切得非常诚实，但反过来也说明 RHI 不是万能。** 弱模型 + RHI 不能跨越能力鸿沟。这跟 "RHI 取代 train-time scaling" 听起来差很远——它只是同一款 base model 上的优化器。

**第四个保留：作者自己也说了"trace 是第一半"**。下一半是"这些 trace 怎么回流到训练"——这是更难也更值钱的回路。RHI 只做了一半的承诺，剩下要等后续工作。

### 6.3 跟当下 agentic system 趋势的关系

RHI 的位置在 agent 系统的优化栈里大概是：

```
┌─────────────────────────────────────┐
│ 1. 换更强 base model (train-time)  │  ← 提升能力上限，但贵
├─────────────────────────────────────┤
│ 2. 改 provider-built harness        │  ← 一次改动覆盖全用户，慢
├─────────────────────────────────────┤
│ 3. RHI: 改 user-constructed harness │  ← 本文位置：$O(1)$/迭代
│    （prompt 层）                     │
├─────────────────────────────────────┤
│ 4. 改 test-time scaling (TTS)        │  ← 推理档位高，cost 涨
│    (high → xhigh → max → ultracode) │
├─────────────────────────────────────┤
│ 5. 改 trace → post-training          │  ← RHI 的未来一半
└─────────────────────────────────────┘
```

**RHI 跟 TTS 不是替代关系，是互补**——主实验里 RHI 改的是 high 档位，但 RHI 之后的 H[2] 比 max 还强且更便宜。这暗示：**TTS 那一层（推理档位）其实是个粗糙的"加推理"旋钮，而 RHI 是更细粒度的"改信息路由"旋钮**。工业界未来很可能把 RHI 思路嵌进 provider-built harness，让"xhigh"档位内部就用 RHI 进一步特化。

### 6.4 如果你也在调 agent，能从 RHI 学到什么

不一定要复现 RHI 算法本身，但有几个**可立即借鉴**的设计原则：

1. **把 harness 拆成结构化组件（role / instruction / contract / hop）**。每次修改时分别打分，能定位"是哪一个组件在拖后腿"。RHI 的核心 insight 就是 contract 和 hop 最重要。
2. **监控"输出了多长"，不要把它当成性能代名词**。RHI 的强项在于它不靠增长 output token 就能赢——如果你在优化时只盯胜率不看 token，可能不知不觉在拼"谁生成得更长"。
3. **多次 noisy pairwise 比一次 clean 评分更划算**。如果你的评估是 LLM-based 的 pairwise，把预算摊到"多次迭代 × 跟上一版比"通常比"一次 vs 黄金 baseline"信息密度更高。
4. **优化器不要看到评估标准**。RHI 的 L_harness 不读 x_eval，只看偏好历史——这个解耦很关键，避免"优化器 hack 评估器"。

---

## 📚 写在最后

RHI 的故事讲得直白：**在固定的 base model 上，靠修改 harness 文本就能把 agent 在开放任务上的胜率从 11/30 拉到 23/30，cost 砍 60%**。这听起来像是"魔法 prompt engineering"，但论文用组件级消融证明它不是——RHI 学到的是**任务专属的 contract 模板**，是信息路由的结构性优化，不是 prompt 玄学。

如果让我给这篇论文打一个工程师视角的实用分：

- **学术价值**：高。提出了一个干净的 $O(1)$ 优化视角，并且在 30 个任务上证明可行。
- **工程价值**：中高。算法本身可以几小时复现（用 LangChain 的 ReAct + LLM judge 就能搭个 demo），但要复现到论文里的胜率需要认真的 prompt 工程。
- **直接可用度**：中。Sakana AI 没开源代码（截止我读这篇时），需要自己实现。但 "harness as text + LLM pairwise judge" 这个范式任何人都能马上试。

最有意思的副产物其实是那个**信息论隐式目标**——它给"多 agent 系统的优化"提供了一个新的形式语言：最大化任务互信息 + 最小化跨组件冗余。如果这个框架被后续工作验证或者推广，RHI 就会从"一个很酷的方法"变成"一个可推广的范式"。

论文在最后一段说：*"Future work will complete the second half of the loop by investigating how the resulting execution traces can be effectively internalized into future foundation models."* ——这是更重要的那一半。如果 RHI 产生的 trace 真的能提升下一代模型，那就是真正意义上的 self-improvement loop，而不只是一个 inference-time 的工具。

Sakana AI 的研究品味一直很"系统流"——之前的论文很多是从系统角度找突破口，而不是单纯刷 benchmark。这篇 RHI 延续了这个风格，给 2026 年 agentic system 的"训练-部署"缝隙提供了一个**轻量级的实用桥梁**。

---

## 🔗 链接

- 论文 arXiv：https://arxiv.org/abs/2607.15524
- Sakana AI：https://sakana.ai

---

觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。
