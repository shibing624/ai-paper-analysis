# 35B 撬动万亿：用「时程」换「参数」的 Agent 缩放新路径

你有没有这种感觉——现在最强的 agent 模型，动辄 1.6T、1T 参数，部署成本高到让人头皮发麻。小团队想复现同等能力？数据、算力、训练 pipeline 缺一不可，基本等于做梦。

那有没有可能换个思路：不用堆参数，而是把 agent 一次任务里能"走多远"这件事给堆起来？上海 AI 实验室（InternAI）联合多机构在 arXiv:2606.30616 上挂出的 **Agents-A1** 就是干这个的——一个 **35B 的 MoE 模型**（激活参数 3B），在长时程 agent 基准上把 Kimi-K2.6、DeepSeek-V4-Pro 这些 1T+ 级对手按在地上摩擦。

> 注：作者列表超过 50 人，机构横跨上海 AI 实验室、复旦、上海交大等。这是典型的"大团队系统工程"产物。

---

## 🎯 核心摘要

**痛点**：长时程 agent 任务（科学发现、复杂工程、深度研究）需要模型在长时间跨度内持续规划、调工具、验证证据、修正策略。现有两条 scaling 路线各有瓶颈——堆参数路线复现成本极高，堆时程路线又缺统一的基础设施和能力整合方案。

**方案**：Agents-A1 提出三件事：
1. **Long-Horizon Knowledge-Action Infrastructure**——把外部知识、动作、观察、验证器统一成一个 KAG（Knowledge-Action Graph），用 proposer-solver-verifier 自博弈扩展出 6 个领域子图
2. **三阶段训练 Recipe**——全领域 SFT → 领域教师模型（Search/Science/Inst./Tools 四路）→ 多教师 On-Policy Distillation
3. **Salient Vocabulary Alignment (SVA)**——在 teacher 选出的 top-k 高概率 token 上做截断 reverse KL，避免单 token 蒸馏的近邻替代不受约束

**效果**：35B 激活 3B 的模型，在 SEAL-0 (56.4)、GAIA (96.0)、HiPhO (46.4)、FS-O (79.0)、FS-R (40.0)、IFBench (80.6) 六个基准上反超 1T 级对手；在 BrowseComp、SciCode、HLE 等上与 1T 模型高度可比。

**我的判断**：这是一篇**工程整合型突破**，不是底层算法创新。价值在于证明了"scaling the horizon"这条非主流路径的可行性，并给出了一套可复现的 recipe。但别被"35B 打 1T"唬住——它靠的是 100K 条 45K tokens 的长轨迹数据 + 精细的多教师蒸馏，本质是用**训练成本**换**推理参数**。对资源受限的团队来说是福音，但别指望拿个 35B base 就能复现。

---

## 📖 论文信息

| 项 | 内容 |
|---|---|
| **标题** | Scaling the Horizon, Not the Parameters: Reaching Trillion-Parameter Performance with a 35B Agent |
| **arXiv** | [2606.30616](https://arxiv.org/abs/2606.30616) |
| **提交日期** | 2026 年 6 月 29 日 |
| **机构** | Shanghai AI Laboratory 等（论文未列出统一通讯单位） |
| **模型权重** | [HuggingFace: InternScience/agents-a1](https://huggingface.co/collections/InternScience/agents-a1) |
| **代码仓库** | [GitHub: InternScience/Agents-A1](https://github.com/InternScience/Agents-A1) |

---

## 🤔 为什么需要这篇论文？

先说清楚"长时程 agent"到底难在哪。真实任务里——比如让 agent 跑 12 小时自主调一个 Kaggle 比赛，或者让 agent 自主重建一个热带气旋的历史轨迹——模型需要：

- **长时间跨度的规划**：不是"搜索→回答"这种两轮对话，是"分解问题→调工具→看结果→重规划→再调工具→验证"的几十上百轮循环
- **跨域能力整合**：同一个 agent 既要会写代码、又要会搜资料、还要会遵循复杂指令、做工程优化
- **早期错误的容错**：第 3 步调错工具，不能直接崩盘，得能识别、回溯、换路径

传统上有两条路：

**路线 A：堆参数**。Kimi K2.6（1T 总量/32B 激活）、DeepSeek V4-Pro（1.6T 总量/49B 激活）就是这条路。效果确实好，但部署成本和复现门槛都极高。K2.6 在 OpenRouter 上单日调用 2970 亿 tokens，V4-Pro 适配华为昇腾芯片——这些都不是小团队玩得起的。

**路线 B：堆时程（scaling the horizon）**。把"思考过程"显式展开，让 agent 的中间决策、知识获取、验证信号都变成可训练的监督。这条路理论上更亲民，但有两个一直没解决的瓶颈：

1. **缺基础设施**——长轨迹训练需要统一的"知识-工具-观察-验证"环境，没有这个，agent 学不到真正的长程能力
2. **缺能力整合方案**——不同领域（搜索、科学、工程、指令遵循）的推理模式冲突，堆在一起训往往互相干扰

Agents-A1 的全部贡献，就是把这两个瓶颈给填了。

---

## 🏗️ 方法核心：三条腿走路

整篇论文的思路可以浓缩成一张图：

![Figure 2: Agents-A1 三阶段训练流程](https://www.mulanai.com/fs/files/0701_023dbaf5_train_fr.png)

*图 1：Agents-A1 的三阶段训练 pipeline——从多领域数据出发，先做全领域 SFT，再训练 4 个领域教师模型（Search/Science/Inst./Tools），最后通过 Salient Vocabulary Alignment 统一蒸馏成一个可部署的 35B 学生模型*

简单说，就是"先吃百家饭、再拜四师父、最后融会贯通"。

### 第一条腿：Long-Horizon Knowledge-Action Infrastructure（KAG）

这是整篇论文最底层、也最容易被忽略的设计。作者把每个领域的训练语料都组织成一个 **Knowledge-Action Graph（KAG）**：

$$\mathcal{G}_d = (\mathcal{C}_d, \mathcal{A}_d, \mathcal{O}_d, \mathcal{V}_d)$$

- $\mathcal{C}_d$：证据语料（wiki、论文、代码、数据集）
- $\mathcal{A}_d$：动作空间（工具调用、检索、编辑、执行）
- $\mathcal{O}_d$：观察空间（工具返回、检索结果、执行状态）
- $\mathcal{V}_d$：验证器（正确性、证据支持、约束满足）

轨迹就是这些节点之间的边：$\tau = [(s_1, a_1, o_1, v_1), \ldots, (s_T, a_T, o_T, v_T), y]$

更聪明的是扩展机制——用 **proposer-solver-verifier 自博弈**：

- **Proposer** 采样图区域出题
- **Solver** 用检索+工具解题
- **Verifier** 检查答案、证据、执行结果、shortcut 风险

五项接受门槛（可验证、有效、过程信息性、证据覆盖、无 shortcut）保证生成的题目质量。

![Figure 3: 知识-行动基础设施](https://www.mulanai.com/fs/files/0701_ea15791d_KAG_infr.png)

*图 2：KAG 基础设施全貌——左侧是从 web、书籍、论文、代码、数据库等异构语料出发，通过 5 种原子能力（信息获取、工具调用、可执行迭代、证据验证、约束追踪）抽取，中间构建统一的 KAG，右侧通过自博弈扩展出 Coding/Agentic/Instruction/MLE/Scientific/Mid-train 6 个领域子图*

这套基础设施最终支撑起 6 个异构领域：Long-horizon Search、MLE、Scientific Reasoning、Instruction Following、Tool Calling、General Agentic Tasks。对应到 SFT 数据集的轨迹长度分布：

| 数据来源 | 平均 Token 长度 |
|---------|---------------|
| Deep research | 44K |
| Coding and engineering | 48K |
| Scientific reasoning | 37K |
| Instruction following | 3K |
| General agentic tasks | 39K |
| **Overall** | **45K** |

**45K tokens 的平均轨迹长度**——这就是"scaling the horizon"最直观的体现。常规 SFT 数据平均几百到几千 tokens，这篇直接拉到 45K。

### 第二条腿：三阶段训练 Recipe

**Stage 1: Full-domain SFT**

Base model 选了 **Qwen3.5-35B-A3B**（35B 总量/3B 激活的 MoE），在 ~100K 条长轨迹上做全领域 SFT。超参数中规中矩：

- Learning rate: $1 \times 10^{-5}$
- Cosine schedule + 5% warmup
- Batch size: 16
- Max sequence length: 131,072
- AdamW + weight decay 0.1

关键工程细节是 **sample packing**——把多个短样本拼到最大上下文，用 attention mask 防止交叉污染。没有花活，就是把长序列训练跑稳。

**Stage 2: 领域级教师模型**

四个领域教师，每个训练方式不同：

**Search Teacher**：SFT + GRPO RL。每个 prompt 采 8 个 rollout，奖励三件套——correctness reward（LLM judge 判答案对错）、search behavior penalty（效率惩罚 + 重复查询惩罚）、format calibration reward。训练配置：LR $1 \times 10^{-6}$，max tool calls per rollout 300，max sequence 131K。

**Science Teacher**：两阶段 SFT。先在"纯推理轨迹"（问题分解、物理假设、单位一致性、答案验证）上训，再在"工具增强轨迹"（search/visit/code/scholar 四种工具）上继续。强调"先会想，再会用工具"。

**Instruction Following Teacher**：两阶段 GRPO。Stage 1 在 Nemotron 数据上训多约束指令遵循（规则可验证 reward）；Stage 2 在长上下文学习数据上训（答案匹配 reward）。**动态采样**——只保留 reward 非均匀的 group，过滤全相同的。

**Tool-calling Teacher**：SFT + RL。奖励设计有讲究——outcome reward $r^{\text{out}} \in \{0,1\}$ 加 process score $r^{\text{proc}}$，用 **asymmetric advantage**：

$$A_i = A_i^{\text{out}} + \lambda_{\text{neg}} \cdot \mathbb{1}[r_i^{\text{out}}=0] \cdot A_i^{\text{proc}}, \quad \lambda_{\text{neg}} = 0.5$$

也就是说，**只在 outcome 失败时才用 process score**——成功了就不额外奖励中间步骤，避免 agent 学会"假装在思考"。

**Stage 3: Multi-teacher On-Policy Distillation（核心创新）**

这是整篇论文的技术核心。设计思路：

1. Student 从 Stage 1 的 SFT checkpoint 初始化
2. 每个 prompt 带 domain label，由对应领域 teacher 提供蒸馏信号
3. **Teacher 不生成参考轨迹**——而是对 student 自己的 on-policy rollout 做 token 级评分
4. Tool outputs、user turns、environment observations 从 loss 中 mask（这些不是 student 生成的）

Salient Vocabulary Alignment（SVA）解决了什么？传统 sampled-token OPD 只对已实现 token 评分，附近的"差一点就会选"的高概率替代 token 完全不受约束。SVA 的做法是：

**在 routed teacher 分布的 top-k valid tokens 上重新归一化两个分布**：

$$\bar{p}_{s'}(u) = \frac{p_{s'}(u)}{\sum_{v \in \mathcal{S}_{i,t}^{(k)}} p_{s'}(v)}, \quad \bar{p}_{t,i}(u) = \frac{p_{t,i}(u)}{\sum_{v \in \mathcal{S}_{i,t}^{(k)}} p_{t,i}(u)}$$

然后在这个"salient support"上算截断 reverse KL：

$$\ell_{\text{SVA}}^{(i)} = \frac{1}{|R_i|} \sum_{t \in R_i} \sum_{u \in \mathcal{S}_{i,t}^{(k)}} \bar{p}_{s'}(u) \log \frac{\bar{p}_{s'}(u)}{\bar{p}_{t,i}(u)}$$

同时还监控 student 在 salient support 上的覆盖率 $\rho(i,t) = \sum_{u \in \mathcal{S}_{i,t}^{(k)}} p_{s'}(u)$，覆盖率高表示接近全词汇对齐。

说实话，这套设计的工程意义在于：**让 distillation 信号更稳**。传统 OPD 的梯度方差大，是因为它只盯一个 token；SVA 把视野扩到 teacher 认为合理的 top-k 集合，相当于给 student 的"选择题"圈了一个范围，蒸馏信号更密集也更平滑。

---

## 🧪 实验结果：35B 真能打 1T？

先看主实验表——这是最硬的部分：

| Benchmark | **Agents-A1 (35B)** | Qwen3.5-35B-A3B | Nex-N2-mini | Kimi-K2.6 (1T+) | DSV4-Pro Max (1T+) | GPT-5.5 (1T+) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Long-horizon Search** | | | | | | |
| BrowseComp | **75.5** | 67.9 | 74.1 | 83.2 | 83.4 | 84.4 |
| XBench-DS-2510 | 86.0 | 71.0 | 82.0 | **90.0** | **90.0** | 84.0 |
| SEAL-0 | **56.4** | 38.7 | 49.6 | 50.5 | 55.0 | 42.3 |
| GAIA | **96.0** | 78.6 | 82.5 | 80.6 | 98.1 | 87.4 |
| **Engineering** | | | | | | |
| SciCode | 44.3 | 35.8 | 29.9 | 53.5 | 50.0 | **56.1** |
| MLE-Bench-Lite | 43.9 | 34.9 | 34.9 | 62.1 | 63.6 | **72.7** |
| **Scientific Research** | | | | | | |
| HLE w/ tools | 47.6 | 36.2 | 32.0 | **54.0** | 48.2 | 52.2 |
| HiPhO | **46.4** | 37.7 | 38.5 | 41.1 | 38.7 | 43.3 |
| FS-O | **79.0** | 60.3 | 52.0 | 73.0 | 76.0 | 78.0 |
| FS-R | **40.0** | 2.9 | 5.0 | 17.9 | 13.3 | 26.7 |
| **Instruction Following** | | | | | | |
| IFBench | **80.6** | 64.4 | 54.1 | 71.8 | 73.5 | 75.9 |
| LongBench V2 | 60.2 | 57.7 | 59.6 | 62.0 | **64.3** | — |
| **General Agentic** | | | | | | |
| τ²-Bench | 79.8 | 79.0 | 74.5 | 81.9 | **82.2** | 81.6 |
| VitaBench | 38.8 | 35.6 | 23.0 | 35.6 | **49.0** | 45.0 |
| **Scientific Agentic** | | | | | | |
| MatTools | 47.1 | 15.9 | 34.1 | **63.8** | 47.1 | 68.8 |
| MolBench-Bind | **56.8** | 48.7 | 51.4 | 21.6 | 37.8 | 62.2 |

**Agents-A1 拿第一的基准**（6 个）：SEAL-0 (56.4)、GAIA (96.0)、HiPhO (46.4)、FS-O (79.0)、FS-R (40.0)、IFBench (80.6)。

**这个表透露出几个有意思的信号**：

第一，**搜索和工具使用类任务（SEAL-0、IFBench）35B 完全可以反超 1T**。这暗示当任务结构是"多轮调用+验证"时，参数规模不是瓶颈，轨迹质量和训练信号更重要。

第二，**深度工程类任务（SciCode、MLE-Bench-Lite、MatTools）还是 1T 强**。这些任务需要深厚的代码理解和长链路推理，参数规模带来的"知识容量"优势还是显著的。Agents-A1 在 MLE-Bench-Lite 上 43.9 vs GPT-5.5 的 72.7，差了快 30 个点。

第三，**科学发现类（FS-R）出现极端分化**——Agents-A1 40.0，Qwen3.5 base 只有 2.9。这说明 SFT + 蒸馏的"长程能力"在这个基准上贡献巨大，但 1T 模型（GPT-5.5 26.7）反而没打过 35B，挺反直觉的。

**我的判断**：35B 能在长时程搜索/指令遵循上打 1T，但在"需要深知识的硬核工程"上仍然打不过。这是符合直觉的——参数规模的真正价值是"知识存储"，agent 的"长程行为"更多是训练数据和 RL 的功劳。

### 三阶段训练的贡献

把 SFT 和完整 Agents-A1 分开看，蒸馏阶段的价值非常清晰：

| Benchmark | Qwen3.5-35B-A3B | Agents-A1-SFT | Agents-A1 |
|---|:---:|:---:|:---:|
| BrowseComp | 61.0 | 74.6 | 75.5 |
| SEAL-0 | 41.4 | 52.3 | 56.4 |
| GAIA | 59.8 | 95.2 | 96.0 |
| HLE w/ tools | 47.4 | **41.6** | 47.6 |
| IFBench | 70.2 | **68.7** | 80.6 |
| FS-R | 2.5 | 31.7 | 40.0 |
| MatTools | 21.0 | 37.0 | 47.1 |
| MolBench-Bind | 46.0 | 46.0 | 56.8 |

**关键发现**：SFT 阶段在 **HLE** 和 **IFBench** 上反而出现了性能下降（HLE 47.4→41.6，IFBench 70.2→68.7）。论文解释是"长思考推理模式与多轮 agentic 模式的差异"——SFT 强行把两种模式揉在一起，互相打架。

**而 OPD 蒸馏阶段把这些下降全部修复了**（HLE 41.6→47.6，IFBench 68.7→80.6）。这恰好证明了 SVA 的价值：**当一个统一模型需要同时具备多种推理模式时，领域教师的 on-policy 指导是必不可少的，不是可选项**。

### 消融实验：每个教师都不可或缺

| Teacher | 关键提升 |
|---|---|
| **Search Teacher** (SFT+RL) | GAIA 59.8→95.1 (+25.3)，SEAL-0 41.4→54.1 (+12.7) |
| **Science Teacher** (SFT) | FS-R 2.5→54.3 (+51.8)，FS-O 64.5→82.0 (+17.5) |
| **Inst. Following Teacher** (RL) | IFBench 70.2→82.0 (+11.8) |
| **Tool-calling Teacher** (SFT+RL) | τ²-Bench Airline 16.0→72.0 (+56.0)，VitaBench Cross 11.5→30.0 (+18.5) |

每个教师都把对应领域的能力拉了一大截。Science Teacher 在 FS-R 上 +51.8 的提升尤其夸张——这个基准基本上是 Science Teacher 一手撑起来的。

---

## 🔬 两个让人印象深刻的 Case Study

论文还给了两个 12 小时长时程任务的实测案例，展示了 Agents-A1 在真实长链路任务上的能力。

### 案例 1：ICML 2013 Whale Challenge

任务：检测鲸鱼叫声。起点是 naive CNN baseline，12 小时自主迭代：

> 时间数据分析 → 音频增强 → 时间局部化训练 → Mel-spectrogram CNN 集成架构 → 大规模增强

best validation AUC 从 0.58 提升到 0.9935——金牌级别。

![Figure 4: ICML 2013 Whale Challenge 优化轨迹](https://www.mulanai.com/fs/files/0701_b9625069_appendix.png)

*图 3：Agents-A1 在 ICML 2013 Whale Challenge 上的 12 小时优化轨迹，best validation AUC 随 wall-clock 时间的增长曲线，标注了关键突破时刻*

### 案例 2：2008 年纳尔吉斯气旋重建

任务：自主重建 Severe Cyclonic Storm Nargis 的完整轨迹。从识别 IBTrACS 数据源开始，完成数据提取、清洗、派生指标计算、可视化、结果综合——一个完整的"规划→编码→执行→验证→科学分析→报告生成"闭环。

![Figure 5: 纳尔吉斯气旋分析](https://www.mulanai.com/fs/files/0701_8ab62d19_appendix.jpg)

*图 4：Agents-A1 重建的纳尔吉斯气旋（2008）综合特征——包括最佳路径图、最大风速时序、经纬度时序、移动速度、转向角度等，完整复现了气旋的形成、西北移动、转向东北、登陆缅甸南部的全过程*

这两个案例的价值在于：它们不是 benchmark 数字，是**真实的工程/科研链路**。如果 Agents-A1 真能稳定复现这种 12 小时自主迭代，那"scaling the horizon"这条路确实有戏。

---

## 💡 我的判断：值不值得花时间细读？

### 亮点

1. **路线价值**：在"参数规模内卷"的大背景下，提出了一条**不依赖堆参数**的路径。对学术界和资源受限团队意义重大。
2. **基础设施完整**：KAG + self-play graph search 的设计，把"长时程数据怎么造"这个问题给工程化解决了。这套基础设施可以复用到其他领域。
3. **蒸馏方法扎实**：SVA 解决了传统 OPD 的梯度方差大问题，**asymmetric advantage** 的设计（只在失败时用 process score）非常符合工程直觉。
4. **消融充分**：每个教师、每个阶段都有独立贡献验证，没有"包打天下"的模糊叙事。

### 问题

1. **复现成本不低**：100K 条 45K tokens 的 SFT 数据 + 4 个教师模型 + 多阶段 RL——这套 pipeline 的训练成本不一定比直接训 1T 模型便宜多少。论文没说用了多少 GPU。
2. **Base model 选择有偏向**：用 Qwen3.5-35B-A3B 做 base，但 ablation 里没看到用其他 base（如 Llama 系列）的对比。35B MoE base 本身就有 3B 激活的"甜头"，换 dense model 不一定 work。
3. **公平性存疑**：论文说"对比 1T 模型用了对方 technical report 的原数字"，但实际推理配置（temperature、reasoning effort、tool setup）未必完全一致。比如 GPT-5.5 用了 xhigh reasoning effort，其他模型用 temperature 0.7，这个对比是否完全公平？
4. **"scaling the horizon"是个新瓶旧酒**：把推理时程拉长这件事，OpenAI o1、DeepSeek R1 都在做。这篇论文的特殊贡献是把它系统化、做成 recipe，但底层 idea 不算新。
5. **Eval benchmark 选取可能有利**：FS-R 40.0 vs GPT-5.5 26.7 这种悬殊比分，是 agent 模型的强项（长链路推理）压倒了 1T 的强项（知识容量）。论文精心挑选的 benchmark set 是不是在展示 Agents-A1 最擅长的场景？

### 工程启发

如果你的团队在做长时程 agent，这篇论文有三个**直接可借鉴**的点：

- **别再手工攒 SFT 轨迹了**——用 proposer-solver-verifier 自博弈造数据，质量更高、规模更大
- **领域教师 + on-policy distillation 比 end-to-end SFT 稳得多**——尤其是当你的 base model 知识容量有限时
- **Salient vocabulary alignment 可以推广**——这个 trick 不只适用于 agent model，通用 LLM 蒸馏也能用

---

## 🔗 收尾

Agents-A1 的最大价值，不是"35B 打 1T"这个看起来很燃的 headline——而是它**给出了一套完整的、可落地的"非参数 scaling"路径**。

在 Kimi K2.6、DeepSeek V4-Pro 都在堆万亿参数的 2026 年，这篇论文提供了一个**反共识但有说服力**的替代方案：用更长、更密集的轨迹数据 + 精细的多教师蒸馏，让小模型也能跑长时程任务。

这条路不一定比堆参数更省钱（训练成本摆在那里），但它的**复现门槛更低、推理成本更低**——对学术界和中小团队来说，这比"又一个大模型 release"更有意义。

最后留一个开放问题：Agents-A1 的成功，到底是因为 45K tokens 的长轨迹数据，还是因为 SVA 蒸馏方法的巧妙？论文没有把这两个变量完全拆开。如果有人能在更小规模上复现并对比，会是很有价值的工作。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。*
