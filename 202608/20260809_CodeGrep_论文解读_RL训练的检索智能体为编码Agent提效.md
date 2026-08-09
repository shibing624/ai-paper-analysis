# CodeGrep：RL 训练的检索智能体，为 LLM 编码 Agent 提效

- **论文标题**：CodeGrep: An RL-Trained Retrieval Agent for LLM Coding Agents
- **arXiv 链接**：https://arxiv.org/abs/2608.05886（v1，2026-08-06 提交）
- **作者**：Wuya Chen（网易广州 AI Lab）、Yihao Yang、Yang Cao（独立研究员）、Yue Lin（网易广州 AI Lab）
- **领域**：软件工程（cs.SE）、人工智能（cs.AI）

---

## 一、核心问题与动机

现代 LLM 编码智能体（如 Claude Code、OpenHands）存在一个共同的低效问题：**大部分 token 预算花在"找到要修改的文件"上，而不是"写补丁"本身**。

关键数据：一个 30B 的 OpenHands 智能体在 SWE-Bench Verified 上，每个已解决 issue 平均消耗 **23 轮、631K tokens**，其中很大比例被 `grep`、`glob`、`view_file` 调用消耗——这些调用常陷入"模糊的探索循环"，经常在错误方向上探索 20+ 轮后才放弃。

论文提出两个此前未被联合检验的假设：

| 假设 | 内容 |
|---|---|
| H1（有效性） | 更好的文件级检索能引导智能体避开死胡同探索，提高 resolve rate |
| H2（效率） | 智能体解决大致相同的 issue 集合，但 rollout 更短 |

**核心发现：两个假设都不一致地成立**。检索质量通过**精度阈值（precision threshold）**而非线性收益传导到下游：

- BM25（precision 0.375）→ **损害**智能体；
- Jina（precision 0.445）→ **中性**；
- CodeGrep（precision 0.677）→ **越过阈值**，开始"买到"效率（-15% 轮次、-19% tokens、+1.2pp resolve rate）。

## 二、方法

### 2.1 系统总览（图 1）

系统分为两个解耦阶段：

- **训练阶段**：CodeGrep 在自定义 RL 环境中用 GRPO 优化，监督信号来自 CATM 挖掘的相关性标签；
- **推理阶段**：训练好的 CodeGrep 接收 issue 描述，输出一小份候选文件列表，注入**冻结的** OpenHands 下游智能体的 prompt 中。

关键设计：**只有检索器被训练，下游智能体完全不动**，因此任何下游测量结果都可以毫无歧义地归因于检索器对 prompt 的贡献。

### 2.2 CodeGrep 模型架构

| 项目 | 细节 |
|---|---|
| 基座模型 | Qwen3-14B-Instruct（14B 参数） |
| 工具原语 | 三个只读原语：`grep`（正则搜索）、`glob`（路径匹配）、`read`（文件内容）——模仿开发者浏览陌生代码库的方式 |
| 并行度 | 每轮最多发出 **8 个工具调用**，并发执行 |
| 轮次上限 | 最多 **4 轮**（3 轮探索 + 1 轮回答，24 次有效读取） |
| 终止条件 | 产出最终答案或达到轮次上限 |
| 输出格式 | 固定 JSON schema：`<answer>{"files": [...], "line_ranges": [...]}</answer>` |

注意：`line_ranges` 字段在下游是"死信息"（OpenHands 编辑工具只消费文件路径），因此在 v3 中将其从训练目标中移除。

### 2.3 RL 环境：轻量级 Git-worktree 沙箱（图 3）

**问题**：复用官方 SWE-Bench Docker 镜像在单机训练规模下不可行——每个镜像 1–3 GB，拉取需数分钟，数千个实例需要 TB 级磁盘。而 CodeGrep 训练只读代码，镜像内的 Python 环境纯属累赘。

**解决方案（三层架构）**：

1. **Layer 1**：从任意 SWE-Bench 风格数据集中提取唯一的 `(repo, commit)` 对；
2. **Layer 2**：每个 repo 只保留一个 `git clone --bare`，用 `git worktree add` 为每个 commit 实例化轻量工作树——磁盘占用从 N×repo_size 降至 repo_size + N×worktree_size，且无网络 I/O；
3. **Layer 3**：rollout 时 grep/glob/read 作为原生子进程在目标 worktree 内运行，带路径穿越保护、64 路并发限制（asyncio 信号量）和每工具超时（grep/glob 10s、read 5s）；输出截断至 4096 字符。

效果：每次 rollout 的环境交互从**分钟级降到毫秒级**，使单机 8×B200 节点上的多轮智能体 RL 可行。此外使用原生 `grep -rn --include=*.py` 而非纯 Python 实现，吞吐提升约一个数量级。

### 2.4 CATM：训练数据挖掘

**为什么不用 gold patch？** SWE-Bench gold patch 触及的文件作为"相关文件"标签是**不完整**的：它遗漏了那些必须被*阅读*以理解修复、但本身不被编辑的辅助文件。直接对 patch 信号优化会使检索器偏离期望行为。

**CATM（Code Agent Trajectory Mining）采用行为相关性定义**：如果某个过去的智能体在解决 issue 时打开了某文件并基于其内容产生了非平凡的推理，则该文件是相关的。

三阶段流水线（作用于来自 Nebius AI 的 **67,074 条开源 OpenHands 轨迹**）：

| 阶段 | 操作 |
|---|---|
| Stage 1（挖掘） | 提取每个文件读取工具调用（`str_replace_editor` 的 `view` 命令），规范化路径，丢弃目录、文档（README.md、issue.md）和无扩展名文件。对每个幸存文件 f，记录其后的 assistant 消息作为 post-reasoning，token 长度记为 l(f) |
| Stage 2（裁判过滤） | LLM 裁判（GLM-5.1-FP8，16 路并行，约 5 小时）将每个文件的 post-reasoning 分类为 RELEVANT 或 NOT_RELEVANT；采取保守偏置——任何非明确 NOT_RELEVANT 的输出都视为 RELEVANT |
| Stage 3（强度感知加权） | 沿用 LRAT，每个文件获得其推理长度的指数饱和分数 |

加权公式（β 是 l(f) 的中位数，μ_raw 是 w̃ 的均值，归一化使 E[w]≈1）：

```
w̃(f) = 1 - exp(-ln 2 · l(f)/β),   w(f) = w̃(f)/μ_raw
```

**真值集构造**：对每个 issue x，奖励的目标集为 **G(x) = G_patch(x) ∪ {f ∈ L(x) : w_f ≥ 0.15}**，其中 G_patch(x) 是 gold-patch 文件，L(x) 是 CATM 挖掘集。权重起噪声过滤作用；每个幸存文件都是 F_β 奖励中的硬正例，与 patch 文件地位平等。

**产出：31,977 个有效训练样本（47.7% 保留率）**；被丢弃的部分主要是误入歧途的读取和文档查找，证实了裁判过滤的必要性。

### 2.5 奖励设计与三次迭代（核心贡献）

**组件分数**：两个精度偏置的 F_β 统计量（β=0.5）：

- **文件级** F_β^file：预测文件集与目标文件集的标准 F_β（边界情形：两者都空得 1 分，恰好一个空得 0 分）。精度偏置反映下游约束：假阳性会污染智能体上下文，而漏检通常可由智能体自身工具调用弥补；
- **行范围级** F_β^lr：预测 p 匹配 gold g 当且仅当文件名相同且 p 覆盖 g 至少 50%；贪心一对一匹配。

**效率信号**：设 C_total 为 rollout 发起的工具调用总数，T 为轮次数，则每轮平均工具调用数 **c̄ = C_total/T**（0≤c̄≤8），将 c̄=4 视为自然饱和点。

**表 1：三次奖励迭代总结**

| 组件 | v1 | v2 | v3 |
|---|---|---|---|
| 基础任务分数 | ½(F_β^file + F_β^lr) | 同 v1 | **仅 F_β^file** |
| Reward 层缩放 | 1/max(1, c̄/4) | 1_{c̄>0} | 1_{c̄>0} |
| Advantage 层缩放 | — | s(c̄) | 同 v2 |

- **v1（reward 层效率缩放）**：R = ½(F_β^file + F_β^lr) · σ(c̄)，σ(c̄) = 1/max(1, c̄/4) ∈ (0,1]。**失败模式**：策略漂移升高（KL 达 ~0.31）+ 下游效率几乎为零。
- **v2（缩放移至 advantage 层）**：A_i = A_i · s(c̄_i)，s(c̄) = √min(c̄/4, 1)。平方根软化折扣：c̄=1 时 s=0.50（仍保留非平凡梯度信号），c̄≥4 时 s=1。两个动机：（1）原始奖励的组内排名被保留，GRPO 的组内比较始终锚定在任务表现上；（2）折扣只作用于特定 rollout 的梯度步，不影响组内其他成员。v2 解决了漂移，但产生新症状：**长度利用**（completion 长度冲到 ~2000 tokens，clipping ratio 接近 20%）。
- **v3（从基础奖励中移除 line-range）**：R = F_β^file · 1_{c̄>0}，沿用 v2 的 advantage 缩放。理由：（1）接口不匹配——下游编辑器没有 view_range 参数，预测的行范围在推理时是死信息；（2）系统性标签偏置——CATM-only 条目按构造没有 gold 行范围，导致大部分训练数据上 F_β^lr=0。

### 2.6 训练配方

| 项目 | 配置 |
|---|---|
| 框架 | GRPO，基于 ms-swift，单机 8×B200 |
| 参数高效微调 | LoRA（rank 32, α=64），应用于 14B 基座所有线性投影；基座权重冻结 |
| 显存方案 | 优化器与模型状态空闲时 offload 到 CPU；训练与 rollout 通过 vLLM colocated 模式共享 GPU（tensor-parallel 8，35% 显存给 vLLM，开启 prefix caching） |
| 优化器 | 峰值 LR 5×10⁻⁶，cosine 衰减，5% 线性 warmup；GRPO KL 系数 β=0.02 |
| 批次 | 每步 64 个 prompt（per-device batch 1，梯度累积 8，8 卡），每 prompt 采 8 个 rollout = 每步 512 个有效 rollout |
| 采样 | rollout 上限 4096 生成 token、4 轮工具调用；temperature 1.0，top-p 0.9，top-k 50 |
| 早停 | 第 897 步早停（计划 2000 步），单次运行端到端约 27 小时 |

**训练动态（图 2）**：

1. **奖励**：v3 收敛最高（0.60–0.65 vs v1/v2 的 0.45–0.48）；
2. **KL-to-reference**：v1 在第 900 步漂移到 ~0.31，而 v2/v3 保持在 0.09/0.15；
3. **平均工具使用轮次**：v3 降至 ~2.1 并保持稳定；v1/v2 在第 500 步后反弹至 ~2.6。

**训练稳定性诊断（图 4）**：(a) 平均 completion 长度——v2 剧烈膨胀（平滑峰值 ~1150 tokens，原始峰值 ~2000），v1 有温和隆起，v3 全程保持 ~300–400；(b) 梯度范数三者全程有界；(c) completion clipping ratio——v2 峰值 ~20%，v1 ~14%，v3 全程低于 6%。

## 三、实验设置

| 项目 | 设置 |
|---|---|
| 数据集 | SWE-Bench Verified（500 实例） |
| 下游智能体 | OpenHands + Qwen3-30B-A3B-Instruct-2507（temperature 0，最多 100 轮） |
| 报告指标 | resolve rate；效率指标（已解决实例的平均轮次与总 token；未解决实例的平均轮次） |
| 基线复现 | 无检索基线解决 25.8%（23.0 轮、631K tokens/已解决 issue），与公开报告的 25.2±0.7% 在一个标准差内 |
| 对比配置（6 个） | baseline、BM25（top-2）、Jina-1.5B（top-2，稠密嵌入检索器）、CodeGrep v1/v2/v3 |

检索质量内部评估集：从 held-out 的 swe-rebench 实例构建，使用 CATM 挖掘的标签并由团队高级软件工程师审计。评分用 F_β（β=0.5）+ 文件级 precision/recall。选 β<1 的原因：假阳性（注入无关文件）比假阴性代价更高——前者膨胀上下文、用干扰代码稀释注意力，后者常能被智能体自身工具调用恢复。

## 四、实验结果

### 4.1 检索质量（表 2，内部评估集）

| Retriever | F_β mean | F_β median | Precision | Recall | F_β≥0.8 | Turns |
|---|---|---|---|---|---|---|
| BM25 | 0.359 | 0.455 | 0.375 | 0.386 | 7.0% | – |
| Jina-1.5B | 0.427 | 0.500 | 0.445 | 0.468 | 7.0% | – |
| CodeGrep v1 | 0.562 | 0.556 | 0.641 | 0.486 | 36.7% | 3.8 |
| CodeGrep v2 | 0.526 | 0.556 | 0.589 | 0.483 | 31.7% | 2.9 |
| CodeGrep v3 | **0.576** | **0.714** | **0.677** | 0.435 | **43.0%** | **2.3** |

三个观察：

1. **CodeGrep 全面压制 BM25 和 Jina，差距主要由 precision 驱动**：v3 的 mean F_β 是 BM25 的 1.6×、Jina 的 1.35×；v3 在 43.0% 的实例上产出高质量检索（F_β≥0.8），是 BM25/Jina（各 7.0%）的 6.1×——这正是下游 token 节省最关键的注入区间；
2. **v3 在质量与成本上 Pareto 占优**：median F_β 达 0.714（比 v1/v2 高 +28%），平均轮次沿迭代单调下降 3.8→2.9→2.3；
3. **v1 与 v2 聚合质量相当**：v2 的 advantage 层效率信号"用少量检索质量换训练稳定性"；line-range 组件并不妨碍文件定位，只是沿下游从不消费的方向消耗了优化容量。

### 4.2 下游评估（表 3，SWE-Bench Verified 全 500 实例）

| Config | Resolve | 已解决轮次 | 已解决 token | 未解决轮次 |
|---|---|---|---|---|
| Baseline | 25.8% | 23.0 | 631K | 32.0 |
| BM25 | 25.2% | 22.9 | 763K | 29.7 |
| Jina-1.5B | 25.8% | 23.2 | 587K | 27.8 |
| CodeGrep v1 | **27.0%** | 22.7 | 627K | 26.2 |
| CodeGrep v2 | 26.6% | 21.4 | 584K | 26.4 |
| CodeGrep v3 | **27.0%** | **19.6** | **514K** | **27.5** |

四个观察：

1. **v3 实现小而可复现的 resolve 提升 + 大得多的效率红利**：resolve +1.2pp（25.8%→27.0%），已解决轮次 23.0→19.6（-15%），已解决 tokens 631K→514K（-19%）；
2. **BM25 注入损害下游**：resolve 降 0.6pp，已解决 tokens 膨胀 21%；在两者都解决的 94 个实例内，BM25 rollout 多花 6.6% 轮次和 38.6% tokens；
3. **Jina 注入大致中性**：resolve 与基线持平，已解决 tokens 仅降 7%——方向与 CodeGrep 一致但幅度小一个数量级；
4. **v1 追平 v3 的 resolve rate 但效率不行**：v1 也达 27.0%，但仅 -1.3% 轮次、-0.6% tokens；v2 居中。

### 4.3 精度阈值分析（核心洞见）

检索质量不线性地传导为下游效用，而是呈现单调精度梯度 + 三种状态：

| 状态 | 检索器（precision） | 下游表现 |
|---|---|---|
| (i) 阈值以下：检索有害 | BM25（0.375） | resolve -0.6pp，已解决 tokens +21%；低精度候选成为智能体必须摆脱的"干扰项" |
| (ii) 阈值附近：大致中性 | Jina-1.5B（0.445） | resolve 持平 25.8%，已解决 tokens 仅 -7% |
| (iii) 阈值以上：边际质量增益被 rollout 效率吸收 | CodeGrep（0.677） | v2→v3 median F_β +28%，但 resolve 仅 +0.4pp，而轮次 -8%、tokens -12% |

两个一致机制：

- (a) **文件定位不是 resolve rate 的瓶颈**——下游智能体在能解决实例上本就能通过自己的 grep/view 恢复大部分相关上下文，在不能解决的实例上失败通常在补丁合成而非定位；因此更锐利的检索压缩 rollout 而非扩大可解决集；
- (b) 假阳性的边际危害与真阳性的边际收益量级相当，故低精度检索器总体上可为净负面。

**要点**：三个数据点——BM25（0.375，hurts）、Jina（0.445，neutral）、CodeGrep（0.677，buys efficiency）——描绘出单调梯度，**中性到净正面的转变点位于 precision 0.45 与 0.68 之间**。该阈值是否迁移到其他下游智能体尚待研究。

### 4.4 奖励迭代的三种失败模式

**v1→v2：reward 层缩放破坏训练稳定性且无法转化为下游效率**

1. 优势估计失真：GRPO 优势是组内相对的，reward 上乘性缩放重塑组内方差、扭曲优势的符号结构。可观测后果：策略漂移三倍——v1 的 KL 达 ~0.31，是 v2（~0.09）的 3.4×；
2. 推理成本更高：v1 平均每次推理 3.8 轮，是 v3（2.3）的 1.65×；
3. 下游效率未兑现：v1 虽有 27.0% resolve，但轮次仅 -1.3%、tokens 仅 -0.6%。

**v2→v3：修复漂移后暴露长度利用（length exploitation）**

1. 胜利：v2 最终 KL 稳定在 ~0.09；
2. 新失败模式：平均 completion 长度冲到 ~2000 tokens，clipping ratio 接近 20%——advantage 层缩放隐式奖励那些"碰巧得分好的长 completion"；
3. 检索质量无净增益：v2 的文件级 mean F_β 0.526 反低于 v1 的 0.562；reward 侧的稳定性改善被消耗在 completion 长度而非文件定位上。

**v3 闭环**：移除 F_β^lr 消除了一个"无下游消费者"且在 CATM-only 条目上系统性为零的信号。相同超参下，v3 奖励收敛更高更快，rollout 轮次稳定在 ~2.1，并交付头条下游收益。

### 4.5 Pooled vs Paired 效率核算

表 3 的效率数字是汇集（pooled）在各自配置的已解决集合上（baseline 129 个、v3 135 个），混合了真实效率效应与选择效应。限定到**两者都解决的 96 个实例**做配对比较：轮次 20.1→18.3（-9%），tokens 525K→448K（-15%）——保留了汇集效应的 60–80%，证实改进是真实的逐 issue 加速。

### 4.6 案例研究：django-15278（表 8）

**问题**：通过 migration 添加 nullable OneToOneField 在 SQLite 上崩溃（*Cannot add a UNIQUE column*）：`django/db/models/fields/related.py` 中 OneToOneField 构造函数无条件设 `unique=True`。

| | Baseline | CodeGrep 注入 |
|---|---|---|
| 动作数 | 74（未解决） | 15（patch 通过） |
| Tokens | 3.2M | 307K（10× 减少） |
| 行为 | 35 次终端调用、17 轮 CoT、18 次文件编辑触及 13 个文件，无一包含修复；前 18 次调用在搜索 oauth2（issue 第三方复现器中的 token，Django 代码库中不存在）；最终误改 `django/db/migrations/operations/fields.py` | 注入 3 个候选文件（含 related.py）；动作 6 打开 related.py，动作 8 grep `OneToOneField`，动作 12 做最小编辑，动作 14 调用 finish；仅触及 1 个工作区文件 |

**机制总结**：CodeGrep 并不提供修复本身——智能体仍需 grep 定位确切定义。它压缩的是冻结智能体否则要燃烧的**探索性前缀**（针对 issue 散文中从不出现于代码库的 token 做 repo 级 find/grep 扫射），token 节省正源于此前缀。

## 五、补充对比表

**表 4：CATM vs LRAT**

| 维度 | LRAT | CATM（本文） |
|---|---|---|
| 目标场景 | Web 检索（Wikipedia QA） | 代码检索（SWE-Bench） |
| 动作模式 | 顺序，每轮 1 次 browse | 并行，每轮最多 8 次 read |
| 推理归因 | 每文档一条推理（一对一） | 一条推理共享给 N 个并行读取（一对多） |
| 裁判模型 | Qwen3-30B-A3B-Thinking-2507 | GLM-5.1-FP8 |
| 数据规模 | 26K 轨迹 → 91K 对 | 67K 轨迹 → 32K 样本（47.7% 保留） |
| 训练方法 | Weighted InfoNCE（稠密检索器） | GRPO（多轮工具调用 RL） |

**表 5：关键工程数字**

| 项目 | 数值 |
|---|---|
| 工具并发上限 | 64（asyncio 信号量） |
| 每轮最大并行工具数 | 8 |
| 每次 rollout 最大工具执行数 | 32（4 轮 × 8） |
| 原始轨迹 | 67,074 |
| 裁判阈值 | reasoning_tokens ≥ 30，w ≥ 0.15 |
| 有效训练样本 | 31,977（47.7% 保留） |
| 裁判阶段运行时间 | ~5 小时 |
| 训练总时长 | 27 小时（8×B200） |

**表 6：与 Cognition SWE-grep 的奖励设计对比**（n/s = 未披露）

| 奖励组件 | Cognition SWE-grep | 本文（v3） |
|---|---|---|
| 文件分数 β | 精度偏置（β 未披露） | β=0.5（精度偏置） |
| 行范围评分规则 | n/s | Jaccard ≥50% 重叠 → F_{β=0.5} |
| 基础分数构成 | 文件 F1 与行 F1 的平均 | 仅文件 F_β；行范围已移除 |
| 训练数据/标签 | 专有 | 开放无监督 CATM 流水线 |
| Advantage 层效率信号 | "按每轮平均工具调用数缩放"（函数形式 n/s） | s(c̄)=√min(c̄/4,1)，monkey-patch 到 GRPO advantage |
| 显式奖励设计研究 | 无（单一最终配置） | v1→v2→v3 三次训练运行 |

**表 7：传统 RLHF vs 本文 Agent RL**

| 维度 | 传统 RLHF | Agent RL（本文） |
|---|---|---|
| Rollout 结构 | 单轮 | 多轮（最多 8 轮） |
| 动作空间 | 单个 completion | 每轮最多 8 个并行工具调用 |
| 奖励来源 | 人类偏好模型 | 环境 + 基于规则的 F_β |
| 环境 | 无 | 带真实代码执行的文件系统沙箱 |
| Loss mask | 整个 completion | 工具响应必须被 mask 掉 |
| 数据构造成本 | 低 | 高（裁判推理 + 全局 β 统计） |

## 六、图表说明

| 图/表 | 内容 |
|---|---|
| 图 1 | 系统总览：训练（蓝）从开源智能体轨迹经 GRPO 产出 CodeGrep 权重；推理（橙）将训练好的 CodeGrep 作为检索子模块，输出注入冻结下游智能体的 prompt |
| 图 2 | 三次奖励迭代的训练动态：(a) Reward：v3 爬升更快、平台更高；(b) KL-to-reference：v1 的 reward 层缩放将 KL 推至 ~0.31，v2/v3 保持 0.09/0.15；(c) 工具使用轮次：v3 稳定在 ~2.1 |
| 图 3 | 三层 RL 环境架构（repo/commit 提取 → bare clone + worktree → 原生工具子进程），右侧与 Docker 沙箱对比 |
| 图 4 | 训练稳定性诊断：(a) 平均 completion 长度（v2 剧烈膨胀、v3 稳定）；(b) 梯度范数（三者有界）；(c) clipping ratio（v2 ~20%、v3 <6%） |
| 表 1 | 三次奖励设计迭代总结 |
| 表 2 | 内部评估集上的检索质量 |
| 表 3 | 六种配置的下游对比 |
| 表 8 | django-15278 的动作级轨迹对比 |

## 七、局限与结论

**局限性**：CodeGrep 的贡献集中在 rollout 成本而非 resolve rate 上；自然的下一步是针对下游 resolve-rate 信号做端到端协同训练，检验检索能否同时买到效率与有效性。精度阈值是否迁移到其他下游智能体也仍是开放问题。

**结论**：本文提出 CodeGrep——一个开源的 14B RL 训练 agent-style 代码检索器。注入冻结的 OpenHands 下游智能体后，它在 SWE-Bench Verified 上提升 resolve rate +1.2pp，并在已解决实例上削减 15% 轮次和 19% tokens。研究还揭示了：(1) 支配"检索何时有助于下游"的精度阈值；(2) 在本文训练设定下，advantage 层（而非 reward 层）是 GRPO 风格多轮训练中效率信号的恰当施加位置。所有工件（模型、训练流水线、RL 环境、评估工具）将被开源。

## 八、个人点评

1. **务实的问题定位**：把"检索"从编码智能体中解耦出来单独训练、单独归因，下游智能体完全冻结，实验归因干净，是这篇论文方法论上最值得借鉴的地方。
2. **精度阈值是核心概念贡献**：检索对下游智能体不是线性增益，而是 hurts / neutral / buys efficiency 三态，转变点在 precision 0.45–0.68 之间。这解释了为什么很多"给 agent 加 RAG"的工作效果不显著。
3. **奖励工程的三次迭代**（v1→v2→v3）是少见的公开失败案例记录：reward 层缩放导致 KL 漂移 3.4×，advantage 层缩放又引发长度利用，最终靠"砍掉下游不消费的监督信号"闭环——对做 agentic RL 的人有直接参考价值。
4. **工程细节扎实**：git-worktree 沙箱把环境搭建从分钟级降到毫秒级、CATM 从 67K 真实轨迹挖掘弱监督标签，两者都是可复用的基础设施贡献。
