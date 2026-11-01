# 给智能体装记忆，居然跑不赢裸 RAG？Memory Tax 一文把记忆系统的账算明白了

你有没有过这种直觉：现在做 Agent，不上个记忆系统都不好意思跟人打招呼。Mem0、A-MEM、各种图记忆、RL 训练的记忆策略……架构一个比一个花哨，论文里的对比表一个比一个好看。

但你有没有想过一个问题——这些对比，对手到底是谁？

大多是"记忆系统 A vs 记忆系统 B"。真正该问的那句"比起直接把原始历史丢给检索器（naive RAG），你这套记忆到底多赚了几个点"，反而很少有人认真回答。UIUC 这篇 Memory Tax（arXiv:2610.34073）就把这笔账翻出来算了一遍，结论挺扎心的：**在三个主流记忆 benchmark 上，绝大多数记忆系统跑不过 naive RAG**。花了成倍的 token 和 LLM 调用，换来的净收益是负的。

这不是一篇"我们提出了更强记忆系统"的论文，而是一篇拆账的论文。我觉得这是它最值钱的地方。

## 核心摘要

这篇论文把"记忆系统相对 naive RAG 的净值"拆成两半：记忆系统答对而 RAG 答错的查询，叫 **Memory Benefit**（记忆收益）；RAG 本来答对、被记忆系统搞丢的查询，叫 **Memory Tax**（记忆税）。在 LongMemEval-s、PersonaMem-v2、MINT-Eval 三个 benchmark、11 个记忆系统、3 个模型 backbone 的大规模评测里，大多数系统的 tax 大于 benefit——净值为负。最夸张的案例里，一个系统在 LongMemEval-s 上比 RAG 低了 40 个点。作者进一步从忠实度、可访问性、稳健性三个系统侧维度，加上 benchmark headroom 这个评测侧维度，把记忆税的来源一层层剥开，并把概念推广到经验记忆和世界模型记忆。值得每个在做 Agent 记忆的人认真读一遍。

## 论文信息

- **标题**：Memory Tax: Dissecting the Net Value of Agent Memory
- **作者**：Xuan He, Sining Zhoubian, Ruozhen Yang, Yuhao Wen, Zhineng Wang, Yuyang Hu, Ke Yang, Yuyu Zhang, Ping Nie, Jiawei Han, ChengXiang Zhai
- **机构**：UIUC 等
- **链接**：https://arxiv.org/abs/2610.34073 （2026 年 10 月 25 日提交）
- **项目页**：https://hexuan21.github.io/MemoryTax/

---

## 🎯 问题动机：记忆系统的"净值"一直没人算

先交代一下背景。长程 Agent 离不开记忆——用户偏好、历史对话、过往经验，总不能每次都把几十万 token 的原始历史塞进上下文。于是大家发明了各种记忆机制：把对话压缩成记忆条目、建知识图谱、做生命周期管理、甚至用 RL 学一个记忆写入策略。

这些工作的评测有个共同的习惯：跟别的记忆系统比。偶尔带上 RAG 基线，设置也常常被弱化——top-k 调小、不加重排、或者干脆不提细节。我的第一反应是，这有点像高考只跟同班同学比名次，从不看全省线。

这篇论文的切入角度很朴素：**记忆处理的本质是对原始历史的一种偏离**（deviation）。任何非平凡的记忆操作都会改变信息的组织、保存或访问方式。这种偏离是双刃剑——设计初衷是好的（重组分散的事实、蒸馏用户偏好），但有些偏离会把证据弄丢或者埋得更深。收益那半边大家天天在宣传，亏损那半边没人算。

于是公式就一句话：

$$\\text{Net Value} = \\text{Memory Benefit} - \\text{Memory Tax}$$

两者都是相对 naive RAG 的 per-query 对比量。Tax 大于 Benefit 时，这套记忆系统就是在赔钱做生意。

## 🏗️ 实验设置：这次的账算得很公道

在拆结果之前，先看看这个实验的"会计准绳"定得怎么样。说实话，这部分做得比我预期的细致。

**Benchmark 三个**：

| Benchmark | 语料数 | 问题数 | 每语料 token 中位数 | 答案形式 |
|---|---|---|---|---|
| LongMemEval-s | 500 | 500 | 约 115k | 简答 |
| PersonaMem-v2（20% 子集） | 40 | 934 | 约 123k | 四选一 |
| MINT-Eval（10% 子集） | 60 | 1557 | 约 139k | 简答 |

子集都按题型分层采样。LongMemEval 偏检索友好，后两个涉及聚合和多干扰推理，更难。

**记忆系统 11 个**，按组织方式分四组：

- 平铺存储：A-Mem、SimpleMem、MemOS
- 层级/生命周期：MemoryOS、LightMem、OpenViking
- 图结构：HippoRAG2、PlugMem、MAGMA
- 训练策略（RL 学记忆写入）：Mem-T、Memory-R2

**Backbone 三个**：Qwen3.5-4B（Q4B）、Qwen3.6-27B（Q27B）、GPT-5.6 Luna（low，API）。统一用 Qwen3-Embedding-0.6B 做嵌入、Qwen3-Reranker-0.6B 做重排，评测用 LLM-as-Judge（与人类标注一致率 95.1% 到 100%）。

**RAG 基线是认真搭的**：按对话轮次切 chunk（1024 字符左右贪心合并），BM25 和稠密检索各取 top-k 后去重合并，再过 reranker 砍半。有单轮和双轮两个变体。更关键的是公平性措施——作者提出 Retrieval Scope（RS）指标，**把各记忆系统的 top-k 调到 RS 不低于 RAG 单轮版**，也就是说多数情况下是给记忆系统放大了检索预算。这等于说：我已经尽量让你赢了，你再输就真没借口了。

## 📊 主结果：一片赤字

直接上数据。LongMemEval-s 上的判分准确率：

| 方法 | Q4B | Q27B | 5.6-Luna |
|---|---|---|---|
| No Context | 9.0 | 9.0 | 17.6 |
| Full Context | 58.6 | 68.4 | 84.0 |
| Gold Context（oracle） | 73.4 | 84.6 | 92.8 |
| RAG 单轮 | 70.2 | 77.6 | 82.0 |
| **RAG 双轮** | **72.2** | **81.4** | **90.0** |
| A-Mem | 70.4（−1.8） | 79.4（−2.0） | 84.0（−6.0） |
| MemOS | 62.0（−10.2） | 76.6（−4.8） | 78.0（−12.0） |
| SimpleMem | 60.2（−12.0） | 73.6（−7.8） | 84.0（−6.0） |
| OpenViking | 31.8（−40.4） | 58.8（−22.6） | 63.0（−27.0） |
| HippoRAG2 | 68.8（−3.4） | 78.2（−3.2） | 81.0（−9.0） |
| MAGMA | 60.8（−11.4） | 68.2（−13.2） | 76.0（−14.0） |
| Mem-T | 71.0（−1.2） | 75.0（−6.4） | 78.0（−12.0） |

括号里是相对 RAG 双轮版的差距。**没有一行是正数**。OpenViking 在 Q4B 上落后 RAG 整整 40.4 个点——这个数字看到的时候我愣了一下，31.8 的准确率意味着这套系统把检索能力毁得比"不检索直接猜"好不了多少。

PersonaMem-v2 和 MINT-Eval 上稍好一点，但也就是"少数系统勉强打平"的水平：PersonaMem 上 RAG 双轮是 36.2 / 48.7 / 48.4，只有 MemOS（+1.0）、MemoryOS（+0.9）、LightMem（+0.2）等少数例外略微为正；MINT-Eval 上 RAG 双轮 26.1 / 33.1 / 34.5，几乎全负，只有 Mem-T 在两个 backbone 上为正（+0.9 和 +3.9）。

还有个有意思的细节：GPT-5.6 Luna 在 PersonaMem 上反而经常输给 Q27B。作者的解释是它偏好通用答案、回避刻板印象和隐私信息——安全对齐在这里变成了绊脚石。

光看准确率还不够震撼，配上成本看才是绝杀：

![Figure 1 左：LongMemEval-s 上准确率与 token 成本的 Pareto 图（Q27B）](https://arxiv.org/html/2610.34073v1/Figures/4_lme_27b_score-token.png)

*Figure 1（左半）：横轴是每个语料的 token 成本（对数刻度），纵轴是准确率，气泡大小是记忆库存储量。RAG 单轮和双轮孤零零地待在左上角——成本低一个数量级以上，准确率还最高。右边那一坨彩色气泡是各个记忆系统，花了几十上百倍的 token，挤在更低的准确率区间里。*

![Figure 1 右：MINT-Eval 上的同款 Pareto 图（Q4B）](https://arxiv.org/html/2610.34073v1/Figures/4_mint_4b_score-token.png)

*Figure 1（右半）：MINT-Eval 上的格局类似。Pareto 前沿被 RAG 占着，记忆系统集体在前沿右下方"交税"。*

这张图的视觉冲击力强过任何文字。你花大价钱建了记忆大厦，结果发现人家搭个帐篷住得比你舒服。

## 🔬 拆账：记忆税到底从哪来的

光说"净值为负"只是现象，论文真正的功夫在第 4 节——把税的来源一层层剥开。

### Venn 图：收益和税各有多少

作者对每个系统和 RAG 做逐查询的对错重叠分析：

![Figure 2：记忆系统与 RAG 单轮版的逐查询正确性 Venn 图（LongMemEval-s, Q27B）](https://arxiv.org/html/2610.34073v1/Figures/4_venn_lme_27b.png)

*Figure 2：蓝色是"记忆对、RAG 错"（benefit），红色是"记忆错、RAG 对"（tax），黄色是都对。A-Mem 是少见的正净值（+9：40 个 benefit 对 31 个 tax）；OpenViking 是灾难现场（−94：只抢回 46 个，却弄丢了 140 个）。*

注意到没有，即使是净值为正的系统，benefit 那一栏也从来不小——记忆系统确实能解决一些 RAG 搞不定的查询。问题从来不是"记忆没用"，而是**弄丢的比抢回的多**。

### 忠实度：信息在构建阶段就丢了

第一个税源是记忆构建阶段的信息损失。作者做了个 oracle 实验：别管检索了，直接把 gold memory units（包含正确答案的记忆条目）喂给模型，看能考多少分：

![Figure 3：原始性能 vs 喂入 gold memory units 后的性能，右侧框内是 memory tax 在 oracle 前后的变化](https://arxiv.org/html/2610.34073v1/Figures/4_oracle_compare.png)

*Figure 3：灰色段是喂入 gold memory 后的增益，红框数字是 tax 从原始到 oracle 的变化。如果 gold memory 都救不回来，说明信息在构建时就丢了。MAGMA 的 tax 从 15.0 降到 6.4，说明它的信息保存得不错、问题主要在检索端；而 OpenViking 从 28.0 只降到 24.2——oracle 都救不动，构建阶段就把证据弄没了。*

微观的人工标注（500 条采样事实）更直白：OpenViking 只有 39.4% 的事实无损失，38.0% 的事实压根没有对应的记忆单元；MAGMA 保存最好（93.6% 无损失），MemOS 85.8%，SimpleMem 65.6% 且 27.6% 部分损失。

### 可访问性：信息还在，但埋得更深了

这是我觉得最反直觉的一个发现。有些系统信息明明保存得好好的，检索时就是找不到。

作者做了个"迭代重放"实验：反复检索，统计把 gold 证据找齐需要几轮：

![Figure 4：找齐证据所需的检索轮次分布（LongMemEval-s, Q27B）](https://arxiv.org/html/2610.34073v1/Figures/4_access_rounds_lme_27b.png)

*Figure 4：RAG 有 82.7% 的查询第一轮就能把证据找齐；记忆系统全面掉队——LightMem 61.6%、OpenViking 53.7%、MAGMA 68.3%、PlugMem 60.1%、Memory-R2 74.9%。尾部还带着一块"找不齐"的阴影（斜线部分），OpenViking 和 PlugMem 有接近两成的查询重放多轮都捞不回证据。*

为什么会这样？相似度漂移图给了微观解释：

![Figure 5：query 与 gold raw turn / gold memory unit 的相似度分布漂移（PersonaMem-v2, Q27B）](https://arxiv.org/html/2610.34073v1/Figures/4_access_drift_psn_27b.png)

*Figure 5：左图是整体分布，灰色是"query vs 原始轮次"的相似度，彩色是"query vs 记忆单元"的相似度——OpenViking 的中位数从 0.351 掉到 0.239，整个分布左移。右图是逐点差异，相当比例的查询相似度为负漂移，还有一部分（no mem）根本没有对应记忆单元。*

你想想看，记忆条目是把原始对话压缩、改写、聚合过的产物。压缩过程把证据碎片化成多个条目、掺进无关信息稀释掉、或者干脆换了个说法。query 和证据之间的语义距离就这么被拉远了。MAGMA、LightMem 的 gold-mem 分数接近 gold-raw、却远高于实际检索分数，坐实了这一点：**不是没记住，是记住了找不着**。

### 稳健性与评测侧：税还有两个隐藏来源

论文还指出两个容易被忽视的税源。一个是**跨 backbone 的稳健性**：固定构建模型、换 QA 模型时，记忆系统的性能波动比 naive RAG 大——记忆机制引入了额外的脆弱性。另一个是 **benchmark headroom**：作者在更多记忆 benchmark 上跑了几个极简基线，发现 IDF 关键词匹配就能上 50 分，RAG-Hybrid 能到 70 分左右。现有评测环境对检索太友好了，留给重型记忆系统发挥的空间本来就有限——税在这个背景下显得更刺眼。

这部分我有个小质疑：headroom 这个论点反过来也可以为记忆系统辩护——现在的 benchmark 测不出记忆的真正价值，不代表真实场景没有。作者自己也承认评测都是静态、离线、英文的。这个局限写得还算诚实。

## 🌐 泛化：记忆税不止存在于对话记忆

第 5 节把记忆税的概念推广到两种完全不同的记忆形态，这是论文视野上的加分项。

**经验记忆**（experiential memory）：在 ALFWorld、WebShop、Mind2Web、BigCodeBench、AppWorld、τ²-Bench 上评测 Memp、ReasoningBank、G-Memory、SkillPro，对比无记忆 ReAct 和 naive RAG（按任务描述检索最相似轨迹）：

![Figure 8：经验记忆在六个 agent 任务 benchmark 上的表现，绿色为超过 RAG，红色为低于 RAG](https://arxiv.org/html/2610.34073v1/Figures/5_evo_scorebar_twin.png)

*Figure 8：网页类任务（WebShop、Mind2Web）上把长轨迹蒸馏成工作流能超过 RAG——benefit 大于 tax；但工具使用类任务（AppWorld、τ²-Bench）上多个记忆方法落后于 RAG，税再次现身。*

**世界模型记忆**：在 Context-as-Memory 和 MemoBench 上测长程一致性，结果更冷——直接检索回注朴素帧（naive frames）的 I3DM 拿到最强一致性（SSIM 0.560 / 1−LPIPS 0.704），各种 learned KV、visual latents、memory tokens 的花式记忆表示一个都没打过。记忆税在世界模型里同样存在。

## 💡 设计启示：怎么少交点税

论文最后给了三条方向性建议，我觉得都有实操价值。

一，**可恢复的偏差**：检索时同时从原始历史检索、和记忆单元联合重排，给系统留一条回到原始证据的轻量路径。多个系统加上这一招后一致提升。这其实承认了记忆处理的偏离不可逆，那就别硬扛，留条退路。

二，**递归自我改进式记忆**：把历史查询、命中的证据、检索路径本身也作为记忆材料——"记住它是怎么记忆的"，实现跨查询的 test-time scaling。这个思路挺漂亮，记忆的元认知。

三，**AoT 与 JiT 的折中**：把记忆看成隐式推荐系统，用用户兴趣、访问模式当先验，选择性地分配构建算力，而不是对所有历史一视同仁地"逢过必记"。

## 🤔 我的判断

这篇论文不是提出新方法的工作，是一面镜子。

它真正的贡献有三层：把"记忆净值"这个概念形式化成 benefit 减 tax 的可计算框架；用一套对记忆系统足够宽容（调大 top-k、给足检索预算）的实验，得出让领域不太舒服的结论；再把税的来源定位到忠实度、可访问性、稳健性三个可操作的维度上。这套分析框架本身比任何单一数字都更有长期价值。

但也要泼点冷水。三个主 benchmark 都是检索友好型的，作者的 headroom 分析其实自己点破了这一点——当答案就在某一轮对话里、检索就能捞到时，任何重组织都是纯开销。记忆真正的价值场景（跨会话的偏好演化、需要多源聚合的推理、超长周期的经验复用）在现有评测里占比太小。所以"记忆系统跑不赢 RAG"这个结论的正确读法不是"记忆没用"，而是：**在检索友好的任务上，重组织是负资产；记忆系统该去赚 RAG 赚不到的钱**。

工程上的启发很直接：如果你正在给 Agent 设计记忆，先搭一个认真的 naive RAG 基线（混合检索 + reranker），把它当成你的"税负起征点"。你的记忆系统每加一层处理，都该回答一个问题——这一层是增税还是减税？以及，无论如何留一条绕过记忆、直达原始历史的检索通路，这是论文验证过的最便宜有效的减税手段。

记忆税这个概念，大概率会成为之后记忆系统论文绕不开的汇报项。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
