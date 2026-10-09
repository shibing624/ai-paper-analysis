# 群聊记忆到底难在哪？SpeakerMem-R1：给每个人建一份档案，再用 GRPO 训一个 3B 小写手

你有没有试过让 AI 帮你回忆一个五人微信群上周的讨论结论？它会告诉你"大家决定走轻松路线"，但你要是追问"Ben 当时同意了吗"，它大概率开始编。

这不是模型变笨了。多人对话的记忆问题，比单人对话难一个量级：一句话是谁说的、说的是谁、是个人私事还是群体共识、这个结论后来被推翻过没有——这些信息在传统的"压缩+检索"管线里，几乎全部丢失。

## 核心摘要

浙大的这篇 **SpeakerMem-R1**（arXiv: 2609.26780）就干一件事：给多人对话建一套"说话人中心"的双轨记忆。一轨原封不动存带说话人标签的原始消息（System 1），另一轨让 Writer 模型把消息改写成结构化的个人档案和群组状态（System 2），查询时按"人、事件、时间"三维把两轨证据拼起来。更狠的是，他们用 speaker-conditioned GRPO 把一个 Qwen2.5-3B 小模型训成了本地可部署的 Writer——在 305 题的对照实验里，RL 把 SFT 写手的准确率从 57.38% 拉到 **68.20 个百分点**，达到大模型写手参考线（71.48%）的 95.4%。在 EverMemBench 公开榜单上拿下 **62.33 个百分点**，是目前已报告的最好成绩。我的判断：这不是底层突破，但工程整合做得相当扎实，"source/owner 分离"这个设计值得所有做群聊 Agent 的人抄作业。

## 论文信息

- **标题**：SpeakerMem-R1: Speaker-Centered Dual-Track Memory for Multi-Party Dialogue
- **作者**：Haobo Zheng, Tan Tang（通讯）, Yan Chen, Weijie Wang, Yingcai Wu
- **机构**：State Key Lab of CAD&CG, Zhejiang University
- **链接**：https://arxiv.org/abs/2609.26780 （2026 年 9 月 22 日提交）
- **代码**：https://github.com/2022hpsk/SpeakerMemR1

---

## 🎯 问题动机：多人对话的记忆，不是"检索相关文本"那么简单

单人对话的记忆系统这两年已经卷得很厉害了——Mem0、A-MEM、MemGPT、Zep，随便拎一个出来在 LoCoMo 上都能跑。但这些系统一进多人群聊就集体翻车。

论文用三个最近的多人基准（GroupMemBench、SocialMemBench、EverMemBench）把这个问题钉死了：通用记忆系统在群聊场景下性能大幅退化，有些配置下甚至打不过朴素的 BM25。

为什么？作者把失败归因到两个耦合的瓶颈：

- **消息归因**（message attribution）：谁说的、说谁的、是个人观点还是群体共识。"Cara 说 Mom 的膝盖还疼"这句话，source 是 Cara，owner 是 Mom——传统系统把这俩搅在一起。
- **状态重建**（state reconstruction）：群聊历史是交错的、多话题并行的、不断修订的。Session 1 大家讨论 Mom 的需求，Session 3 出现分歧，Session 5 才达成最终决定。你要回答"最终决定是什么"，得从三条时间线里把状态链重建出来。

![图1：多人对话不是扁平消息流](https://arxiv.org/html/2609.26780v2/figure1.png)

*图1：一个家庭群聊的案例。左边是三个 session 的交错对话；中间展示了三类现有记忆方案的翻车方式——Flat Memory 漏掉 Cara 的发言（MISSED FACT·TOPIC·MEMBER），Topic Memory 把成员和话题混淆（CONFLATED MEMBER·SCOPE），Graph Memory 的边丢了 source/observe 关系；右边是 SpeakerMem-R1 的方向：Profiles + Group + Query Organization，对应 Anchor–Separate–Resolve–Compose 四步。*

说实话，看到 Figure 1 中间那一列的时候我挺有共鸣的。之前做对话系统的时候我们也试过图谱方案，边一多，"谁观察到谁"这种关系根本挂不住，最后退化成一个昂贵的噪声生成器。

---

## 🏗️ 方法核心：双轨记忆 + 四步查询组织

整个系统的核心 idea 一句话讲清：**原文一轨保真，结构化一轨建状态，查询时按人×事件×时间拼装证据**。

![图2：SpeakerMem-R1 总体架构](https://arxiv.org/html/2609.26780v2/figure2.png)

*图2：上层是 Extraction——chat stream 按说话人拆分，Writer 结合 roster 和 S2 当前头部状态输出 ADD/UPDATE/NOOP 动作，RL 只训 Writer；中层是 Storage——S1 是 append-only 的逐字消息（speaker·time·turn），S2 分四层：Person Core、Person Profile、Group Interaction、Group Insight，每条派生记录通过 from_ids 链回原始消息；下层是 Retrieval——S1 召回后判断 Sufficiency（不够就 ASK 改写问题再搜），S2 按 owner 行式检索，空行回退到该人的原文；最终 Raw + Derived 证据一起交给冻结的 Answerer。*

### 五层存储：一轨原文 + 四层派生

System 1 是唯一逐字层，每条消息带着文本、说话人、时间、频道存进去，不用语言模型，纯追加。这层保证"原话永远在"。

System 2 是四层派生结构，按 scope 劈成两半：

| 层级 | Scope | 存什么 |
|------|-------|--------|
| Core | Person | 稳定身份、事实、立场、习惯行为 |
| Profile | Person | 对某人的观察、跨人认知（"Cara 觉得 Mom 需要帮助"） |
| Interaction | Group | 跨说话人的事件、关系、决策 |
| Insight | Group | 群体规范、共识、例外 |

关键设计是每条派生记录的 schema：

$$r = (x, \mathrm{src}, \mathrm{own}, \mathrm{scope}, \mathrm{event}, \mathrm{time}, \mathrm{state}, \mathrm{ref})$$

**src 和 own 分离**是我觉得全文最值钱的一个字段设计。自己汇报的情况 src = own；但"Alice 认为 Bob 同意了"这种，src = Alice、own = Bob。谁提供信息、信息关乎谁，一刀切开。多人对话里大量的归因错误，根子上就是这两个角色被合并了。

另外 UPDATE 是**非破坏性**的：引用已有 entry_id，追加新节点，通过 links/superseded_by 把状态串成链，历史版本不覆盖。这保证了"当前状态"和"历史状态"两种查询都能回答。

### 查询时的四步：Anchor–Separate–Resolve–Compose

查询不是简单向量搜。先把查询编译成约束：

$$Q_q = (\mathrm{rows}(q), \mathrm{issue}(q), \mathrm{mode}(q), \mathrm{scope}(q))$$

rows 是要覆盖的 PERSON/GROUP 行（"每个人"就展开全部 roster），issue 是事件约束，mode ∈ {head, full} 控制要当前状态还是完整历史，scope 是 source–owner 约束（"Alice 对 Bob 的看法"就固定 src=Alice, own=Bob）。

然后两条检索路径独立预算、各走各的：S1 找原话和局部上下文，支持 Expand 邻居消息和一次 ASK 补充搜索；S2 按行展开后逐行按 issue/事件/时间选记录。某行派生记录为空？回退到这个人的原文消息。还没有？显式保留空行——不让模型脑补。

这个"空行保底"的细节挺戳我的。记忆系统最忌讳的不是找不到，而是找不到的时候装作找到了。

---

## 🔧 Writer-R1：用 GRPO 训一个 3B 小写手

系统层面的 Writer 是模型无关的，你可以直接拿 GPT 当 Writer。但论文单独研究了一个问题：能不能用 RL 把本地小模型训到接近大模型写手的水平？答案是可以。

![图3：Speaker-Conditioned LoGo-GRPO 训练循环](https://arxiv.org/html/2609.26780v2/figure3.png)

*图3：A 部分是 on-policy rollout——同一段对话采样 G=8 条独立写作轨迹，每条轨迹是一串 ADD/UPDATE/NOOP 决策；B 部分是双重信用分配——局部用 SpeakerLevenshtein 在 owner 桶内做一对一匈牙利匹配打分，全局在终态比较"S1+S2"与"仅 S1"的 QA 差值，再按 γ=0.95 折扣回填到每个写作位置；C 部分是 position-wise GRPO——只在相同写作位置比较组内优势，clip 系数 0.2、KL 系数 0.1，只更新 Writer。*

### SpeakerLevenshtein：按 owner 分桶的结构相似度

传统的编辑距离在这里不 work——你得防止"张三的记录写得好"掩盖"李四的记录全丢了"。这个奖励函数把 token-F1 和归一化序列匹配率组合起来，在 owner 桶内做坐标一致的一对一匹配：

$$\Phi_{\mathrm{SL}}(M, M^\star) = w_1 \frac{1}{|P|} \sum_{p \in P} F_p + w_2 \min_{p \in P} F_p$$

宏观平均项管整体质量，最差 owner 项防止高频人物掩盖低频人物和 GROUP 记录。权重 $w_1=0.80$、$w_2=0.20$。这个 worst-owner 项的设计动机很直白：群聊里说话最少的那个人，往往才是问题问的那个人。

### 局部到全局的回报

全局信号有个很聪明的设计——**终端 QA 增益**：

$$R_g^{\mathrm{QA}} = \mathrm{QA}(\text{System 1 + System 2}; M_{g,T}) - \mathrm{QA}(\text{System 1}; M_{g,T})$$

同样的 S1 原文证据，加上你写的 S2 之后 QA 涨了多少，才是 Writer 的功劳。光靠原文就能答对的题，不算你的。这个差分设计直接堵死了 reward hacking 的一条路。

每个写作位置的回报是四项加权：结构合法性（0.20）、记忆质量（0.45）、结构惩罚、折扣后的终端 QA 增益（0.35），折扣因子 γ=0.95。组内相对优势只在**相同写作位置**之间计算，没有组内方差的位置直接跳过——这就是论文标题里 speaker-conditioned GRPO 的出处，思路继承自 Memory-R2 的 LoGo-GRPO，但改成了按位置对齐比较。

---

## 📊 实验结果

### 主实验：三个多人基准全面领先

作者在 GroupMemBench（745 题）、SocialMemBench（1,031 题）、EverMemBench（2,400 题）上对比了 BM25、dense retrieval、Mem0、A-MEM、HippoRAG 和 Full context，并且用 DeepSeek-V4-Flash 和 GPT-5.6-luna 两套模型配置验证跨模型稳健性：

| 方法 | GroupMem | SocialMem | EverMem-All |
|------|----------|-----------|-------------|
| BM25 | 44.6 | 28.6 | 52.5 |
| Embed | 34.5 | 38.1 | 45.9 |
| Mem0 | 21.6 | 13.7 | 19.9 |
| A-MEM | 27.1 | 56.8 | 28.5 |
| HippoRAG | 27.0 | 55.9 | 48.0 |
| Full context | — | 69.4 | — |
| **SpeakerMem-R1** | **47.9** | **69.2**（ASK 关闭时） | **61.9** |

（上表为 DeepSeek-V4-Flash 配置的 Acc.，完整系统 SocialMem 为 64.9。）

几个值得咂摸的点：

- GroupMem 上只比 BM25 高 3.3 个点——说实话这个差距不算悬殊，BM25 的全局词匹配在这类题上意外地能打。
- SocialMem 的 69.2 离 Full context 的 69.4 只差 0.2 个点。也就是说在能塞下全文的小基准上，这套系统已经摸到了"无损记忆"的天花板。
- EverMem 开放问题上 40.0% vs BM25 的 27.0%，涨了 13 个点，这才是差距拉得最开的地方——越需要跨人整合线索，结构化记忆越值钱。
- 有个反直觉的细节：完整系统的 SocialMem 准确率（64.9）反而低于关闭 ASK 的版本（69.2），但 token-F1 更高（32.7 vs 27.4）。作者的解释是补充检索会带进多余的人或错误的 scope，把本来能重叠的答案判成错。所以他们同时报两个指标。这个坦诚我喜欢。

### EverMemBench 公开榜单：62.33% 登顶

在 EverMind-AI 公开报告的 EverMemBench 榜单上（GPT-4.1-mini 答题 + Gemini-3-Flash 评判，2,400 题全量）：

| 方法 | Single | Multi | Temp | Update | 加权总分 |
|------|--------|-------|------|--------|----------|
| MemoBase | 60.09 | 12.85 | 18.00 | 30.60 | 36.21 |
| Mem0 | 55.40 | 11.24 | 6.33 | 51.87 | 39.92 |
| Zep | 73.71 | 8.03 | 13.00 | 43.66 | 41.67 |
| MemOS | 71.36 | 18.88 | 15.67 | 45.15 | 44.63 |
| RippleMem | 92.02 | 22.09 | 21.33 | 58.96 | 54.75 |
| EverOS | 94.37 | 28.11 | 20.33 | 84.70 | 60.08 |
| **SpeakerMem-R1** | 93.43 | 24.10 | **35.00** | 80.22 | **62.33** |

2.25 个点压过 EverOS。但看分项就有意思了：Single（93.43）和 Update（80.22）其实都不如 EverOS，赢在 Temp（35.00 vs 20.33，涨了快 15 个点）和 Proact（75.64 vs 68.62）。Temporal 这一项的大幅领先，恰好验证了非破坏性 UPDATE 链的设计价值——别家把历史覆盖了，它没覆盖。

Multi 只有 24.10，Skill 和 Role 也都在 40 出头，作者自己承认跨证据推理、偏好和角色归因仍是瓶颈。没有把弱点藏起来，这点比很多论文强。

### Writer RL 对照实验：3B 追到大模型的 95.4%

10 个 held-out SocialMem 网络、305 题、查询和答题模块全部冻结，三次运行取均值：

| 写手 | 模型 | Acc. (%) | 与 LLM 写手差距 |
|------|------|----------|------------------|
| SFT (epoch 10) | Qwen2.5-3B | 57.38 ± 0.33 | −14.10 pp |
| **Writer-R1 (30 步)** | Qwen2.5-3B | **68.20 ± 0.66** | −3.28 pp |
| LLM writer | DeepSeek-V4-Flash | 71.48 ± 0.66 | – |

RL 只跑了 30 步就多答对 33 题，从 SFT 的 57.38 拉到 68.20。考虑到这是个可以本地部署的 3B 模型，这个性价比相当可以。不过作者也很克制，明确说这不能证明 RL 有跨领域泛化能力。

### 消融与检索预算

消融（Figure 4）显示：去掉 S2 全部派生层、去掉 S1 原文轨、只去掉 GROUP 层或只去掉 PERSON 层，三个基准上全部掉点——双轨和双层视图是互补的，少了谁都不行。

检索预算的敏感性也做了网格扫描：

![图4：检索预算敏感性网格](https://arxiv.org/html/2609.26780v2/fig_topk_sensitivity_grid.png)

*图4：S1 top-k ∈ {5,10,20} × S2 k ∈ {2,4,6} 的九宫格，上半是 Acc、下半是 token-F1。S1=20、S2=4 时 Acc 最高 73.11 / token-F1 26.53；继续加大 S2 到 6 反而降到 71.15——更多的结构化证据并不总是更好，存在噪声拐点。*

### LoCoMo 边界测试：诚实但不好看

作为双人对话的边界测试，SpeakerMem-R1 在 LoCoMo 全部 1,986 题上拿 70.85%（1,407 题正确）。但分类目看就有点尴尬了：

| 方法 | Single-hop | Multi-hop | Temporal | Open-domain | ALL(Non-AD) |
|------|-----------|-----------|----------|-------------|-------------|
| Mem0 | 67.13 | 51.15 | 55.51 | 72.93 | 66.88 |
| MemOS | 81.09 | 67.49 | 75.18 | 55.90 | 75.80 |
| LightRAG | **86.68** | **84.04** | 60.75 | 71.88 | **79.87** |
| SpeakerMem-R1 | 77.88 | 41.13 | 70.72 | 40.62 | 67.34 |

Multi-hop 41.13、Open-domain 40.62，被 LightRAG 甩开一大截。作者把这个定位为"边界测试"而非主战场——系统是为人多、关系杂的群聊设计的，双人长对话里那些花活（个人/群组视图、owner 分桶）用不上，反而成了开销。这个解释说得通，但也说明这套系统不是万金油。

---

## 🤔 我的判断

这篇论文的定位很清晰：**针对多人对话这个具体痛点做扎实的系统设计**，不是什么底层范式突破。

亮点有三个。一，src/owner 分离加非破坏性更新链，直接打在多人记忆失败模式的七寸上，EverMemBench 的 Temp 分项涨了 15 个点就是证据。二，RL 只训 Writer、其余全冻结的切割方式非常工程友好——检索和答题的稳定性不受影响，训练成本也可控，Memory-R1 那套是整链路训，这里明显更实用。三，评估做得老实：双指标、跨两套模型配置、消融齐全、弱点分项直接亮出来，连"ASK 有时候帮倒忙"这种不利结果都写了。

问题也得说。GroupMem 上只赢 BM25 3.3 个点，说明在话题相对集中的群聊里，结构化记忆的溢价有限；Multi-hop 类问题全面偏弱，说明这套"按人分行"的组织方式对需要跨人串证据的推理帮助不大——而这恰恰是多人对话里最难的那部分。另外整个系统假设 roster 可靠、source/owner 可识别，真实群聊里的别名、成员变更、潜水围观群众，全都是坑。

工程启发很明确：如果你在做的 Agent 需要进群聊场景，别再只用"向量库 + 摘要"了。至少把说话人标签钉死在每条记忆上，把"谁说的"和"说谁的"分开存，UPDATE 别覆盖历史。这三件事不需要 RL 也能做，做了就能避开一大半归因错误。至于要不要上 GRPO 训小 Writer，取决于你对本地部署和成本的敏感度——30 步训练追到大模型 95% 的水平，这笔账还是划算的。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
