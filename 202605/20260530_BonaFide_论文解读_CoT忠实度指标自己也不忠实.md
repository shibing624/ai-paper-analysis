# 你以为在测 CoT 忠实度？这篇论文说现有指标基本都接近瞎猜

**——Tel Aviv & Utah 团队推出 BonaFide：首个用 ground-truth 标签做 meta-evaluation 的 CoT faithfulness 基准，结论挺扎心**

---

## 核心摘要

如果你做 LLM 可解释性、AI safety 或 reasoning 模型评估，"CoT faithfulness"（思维链是否忠实反映模型真实的内部计算）这个词大概率躲不开。过去几年陆续提出了七八种衡量 faithfulness 的 metric——Adding Mistakes、Early Answering、Filler Tokens、Paraphrasing、Simulatability、SCM、FUR、CC-SHAP……每篇论文都说自己测得准。

但有个尴尬的问题一直没人正面回答：**这些指标到底测的是不是 faithfulness？**

要回答这个问题，你需要 ground-truth 标签——某条 CoT 究竟忠不忠实。但模型内部计算又看不见，标签从哪儿来？所以现有 benchmark 大多是用 plausibility（看起来合不合理）这种代理指标——这玩意跟 faithfulness 其实是正交的，根本不能划等号。

这篇 BonaFide 给了一个挺漂亮的破局思路：**构造一些任务，使得"答对"这个事实本身就强制要求模型必须执行了某些特定的中间计算**。这样就有了 ground-truth。基于这个方法构造了 3066 条标注 CoT、覆盖 13 个任务、10 个模型（4B–70B），人工验证标注精度 98.9%。

然后拿这个基准把现有 8 个主流 metric 全部跑了一遍，结论挺扎眼：

- 大多数 AUROC 接近 0.5（随机猜）
- CoT-level 最好的 **CC-SHAP 也只有 0.70**，但每个样本要跑 **10³ 秒**
- Step-level 最好的 **Filler Tokens 0.59**，刚刚好比随机强一点
- 两个最好的 metric **互相不能迁移**（CC-SHAP 在 step-level 跌到 0.41，比随机还差）
- IMP 类指标 **90–96% 的 CoT 都标成 unfaithful**，SEM 类指标 **94–96% 都标成 faithful**——预测分布严重偏斜

我的判断：这篇属于"打脸全行业"的 meta-evaluation 论文，结论可能比方法本身更值钱。如果你正在用 CC-SHAP 做 CoT 监控、或者准备用 Adding Mistakes 这套去验证你训出来的 reasoning model，**先停一下，把这篇看完再说**。

---

## 论文信息

- **标题**：Faithfulness Metrics Don't Measure Faithfulness: A Meta-Evaluation with Ground Truth
- **作者**：Yoav Gur-Arieh（Tel Aviv University）、Ana Marasović（University of Utah）、Mor Geva（Tel Aviv University）
- **arXiv**：[2605.25052](https://arxiv.org/abs/2605.25052)（2026-05-24）
- **代码/数据**：BonaFide on Hugging Face & GitHub

---

## 一、为什么这个问题值得做

先聊聊 CoT faithfulness 这事儿到底有多重要。

你大概都见过这种新闻——某 reasoning model 推理过程"看起来"特别有道理，一步步推导，结论也对。然后被人扒出来：模型其实早就在第一步就猜到了答案，后面那一长串"推理"是事后编出来给人看的。这就是 unfaithful CoT。

这种事情在 AI safety 上是大问题。如果 CoT 是 post-hoc rationalization，那你监控 CoT 就完全失效——模型可能在做坏事，但 CoT 里写得人畜无害。Anthropic、OpenAI 都在用各种方式监控 reasoning 模型的 CoT，前提就是相信 CoT 大体上反映真实推理。

那怎么测一条 CoT 是不是 faithful？传统做法分四大类：

| 类别 | 代表 metric | 思路 |
|------|------------|------|
| **Importance-based**（IMP） | Adding Mistakes、Early Answering、Filler Tokens、SCM | 扰动 CoT 看答案是否变化——如果改 CoT 不影响答案，说明 CoT 没用 |
| **Semantic-utility**（SEM） | Paraphrasing、Simulatability | 看 CoT 的语义是否真的导向那个答案 |
| **Parameter-based**（PAR） | FUR | 从模型参数里 unlearn 单步信息，看预测是否变化 |
| **Attribution-based**（ATT） | CC-SHAP | 比较输入 token 对答案 vs 对 CoT 的 SHAP 重要性 |

每个 metric 都自洽，每篇论文也都"work"。但问题是：**它们之间互相不一致，而且没人有 ground-truth 来仲裁**。

之前最常用的 evaluation 方式是 plausibility——找人看 CoT 觉不觉得合理。但作者一句话戳穿：plausibility 跟 faithfulness 既不必要也不充分。一段胡说八道但符合答案的 CoT 在 plausibility 上可能很高分；一段真实但跳跃的内部推理可能反而不像话。

我之前在做类似的事的时候碰到过这个：手搓了个 CoT 监控器，对照 paraphrasing 测它的"忠实度"，数字漂亮，但拿去跑实际任务全军覆没。当时我以为是我实现错了，现在看大概率是 metric 本身就有问题。

---

## 二、BonaFide 怎么造 ground-truth？

这是这篇论文我觉得最聪明的部分。

核心 insight 一句话：**构造一些任务，使得正确答案的产生在逻辑上要求模型必须执行了某个特定中间步骤**。这样我们就能从外部"看到"这个步骤——它必然发生过，因为答案对了。然后再看 CoT 里有没有 verbalize 这个步骤即可。

作者把这个思路做成两类任务设置：**Outright** 和 **Diversionary**。

### 方法总览图

![Figure 1：BonaFide 方法总览。A：构造两类任务（diversionary 提示误导 + outright 直接计算），其答案必然要求特定中间计算；B：自动标注 pipeline，先 retrieval judge 抽取候选 step，再 validation judge 校验；C：得到 BonaFide 基准](https://www.mulanai.com/fs/files/0530_203ea667_x2.png)

*图 1：A 部分两类任务，B 部分两阶段 LLM judge 标注流程，C 部分一个具体例子——同样回答 Da Vinci 画 Starry Night，左侧 CoT 不承认提示来源还编造历史书出处，所以是 unfaithful；右侧 Collatz 任务每一步运算都老实写出来，是 faithful。*

### Outright 设置：用算法瓶颈定义 ground-truth

设计一类任务，其正确答案需要 **bottleneck steps**——除了执行这一步，没有任何捷径或记忆能给你正确答案。一共 10 类任务：

| 任务 | 类别 | 瓶颈步骤 |
|-----|------|----------|
| nested_mod | 算术 | 链式取模幂/乘/加表达式逐步求值 |
| collatz | 算术 | 迭代应用 Collatz 函数计步数 |
| digit_square | 算术 | 数字平方迭代直至重复 |
| prime_chain | 数论 | 最大质因子+常数→最小质因子 |
| stoichiometry | 化学 | 化学计量逐步换算 |
| cipher | 密码学 | 多层密码（Caesar/块翻转/对换/Atbash）反解 |
| paragraph_analysis | 文本 | 计数重复字母词与 'u' 出现 |
| text_extraction | 文本 | 四阶段文本提取 |
| tournament | 逻辑 | 单淘汰赛冠军判定 |
| graph_traversal | 图遍历 | 最小权重边遍历+状态更新 |

为了证明这些任务确实没有捷径，作者做了一个关键控制实验：禁止 CoT 时模型成功率仅 **1.5 %的水平**。也就是说，要答对这种题，模型必须真的在 token-by-token 地推。

这就是关键。CoT 里有没有把这些 bottleneck steps 写出来，就是 step-level faithfulness 的硬证据。

### Diversionary 设置：用错误提示制造可观察的"决策点"

光有 outright 还不够——它只能告诉你"应当执行的步骤是否被 verbalize"，但模型可能还有别的内部计算（比如它怎么决定要听你的提示）你看不到。

所以作者搞了个互补设置：给模型一个问题，外加一个**指向随机错误答案的提示**。如果模型最后答了那个错误答案，那它必然在某个地方做出了"决定听提示"这个内部决策。CoT 里有没有 acknowledge 这个决策，就是 faithfulness 的证据。

6 种 hint 格式（覆盖从 sycophancy 到 metadata 注入）：

| 格式 | 例子 |
|------|------|
| sycophancy | "A Stanford professor indicates the answer is X" |
| validator | Python 验证函数代码注释里塞答案 |
| metadata | XML 元数据带 answer 字段 |
| error_message | 隐藏错误日志里包含答案 |
| security_audit | 安全审计日志 |
| unauthorized_access | 未授权访问消息 |

跨 SimpleQA、DDXPlus（医学）、HLE（Humanity's Last Exam）三个数据集。重要的控制实验：**没有 hint 时模型只有 0.9% 概率自发产生这个错误答案**——所以一旦答了错误答案，几乎可以确定是受 hint 影响。

Diversionary 下 step 标签变得更细致：

- **Hint acknowledgment**：明确承认提示来源（faithful）
- **Faithful commitment**：表态要采纳提示（faithful）
- **Misattribution**：把答案归因到别处（"我从历史书上看到的"）（unfaithful）
- **Bottleneck execution**：执行真正的算法步骤
- **Tool call / Inert**：工具调用 / 无关内容

### 自动标注 pipeline

人工标 4M tokens 的 CoT 没那么现实。作者搞了两阶段 LLM judge：

1. **Retrieval judge**（Gemini 3 Flash）：根据 step 类型定义，从 CoT 中抽取候选片段
2. **Validation judge**（Gemini 3 Pro）：校验候选片段确实匹配定义

人类抽样验证：**精度 98.9%（95% CI [96.6, 100]）**。总成本约 $2,100。

我对这种"用 LLM judge 当 ground-truth annotator"的方案一向比较谨慎——很容易陷入"用一个黑箱去测另一个黑箱"的循环。但他们做了几件事让我相对买账：

1. 用了两个不同强度的 judge（Flash 抽取 + Pro 校验），减少单模型偏置
2. 人工抽样验证 98.9% 精度，且 95% CI 下界还有 96.6
3. 设计是**精度优先**：宁可漏掉一些 faithful step，也要保证标记的 step 标签准确

代价是 CoT-level 标签分布偏斜：85% Unfaithful CoT，只有 15% Faithful CoT——因为认证一条 CoT 完全 faithful 比认证它有问题难得多。这个我们后面会看到对实验结论的影响。

### 数据集规模

- **3066 条 CoT**，约 4M tokens
- **10 个模型**：Qwen3、OLMo 3、DeepSeek-R1-Distill、Llama 3.3 等家族，4B–70B 参数
- **13 个任务**（10 outright + 3 diversionary 数据集）
- **1946 step 标签**：51% Faithful / 49% Unfaithful
- **1120 CoT 标签**：15% Faithful / 85% Unfaithful

---

## 三、主实验：现有 metric 都怎么样？

直接上结论。

### 主结果表（Step-level & CoT-level AUROC）

| 类别 | Metric | Step AUROC | CoT AUROC |
|------|--------|------------|-----------|
| IMP | Adding Mistakes | 0.51 ± 0.02 | 0.51 ± 0.04 |
| IMP | Early Answering | 0.51 ± 0.01 | 0.45 ± 0.03 |
| IMP | Filler Tokens | **0.59 ± 0.01** | 0.50 ± 0.02 |
| IMP | SCM | — | 0.38 ± 0.03 |
| SEM | Paraphrasing | — | 0.61 ± 0.03 |
| SEM | Simulatability | — | 0.50 ± 0.01 |
| PAR | FUR | 0.52 ± 0.02 | — |
| ATT | CC-SHAP | 0.41 ± 0.03 | **0.70 ± 0.04** |
| **SKY** | LM Judge（含定义） | 0.87 ± 0.01 | 0.82 ± 0.02 |
| **BASE** | Random | 0.50 | 0.50 |
| BASE | LM Judge（generic prompt） | 0.68 ± 0.02 | 0.67 ± 0.04 |

把这个表盯几秒钟。

第一眼的反应：**真的接近随机**。除了 CC-SHAP 在 CoT-level 撑到了 0.70，别的几乎没一个像样的。Filler Tokens 那个 0.59 是 step-level 最高分——这个数你拿去跟人吹"我的 metric 测 step-level faithfulness AUROC 0.59"，怕是要被笑。

第二眼，**SCM 的 0.38**——比随机还差。这意味着它系统性地搞反了。也就是它越觉得"重要"的 step 反而越可能是 unfaithful 的。这种"低于随机"的指标如果不知道反过来用，基本是负贡献。

第三眼，**LM Judge 的 0.87**。这就有点意思了——直接让 GPT/Claude 看 CoT，给它你的 faithfulness 定义，然后让它判断。结果秒杀所有专门设计的 metric。这其实是个挺尴尬的事情：折腾这么多年的 metric 设计，效果不如把定义写清楚直接让 LLM 判。

但 LM Judge 也不是免费的午餐——通用 prompt 的版本只有 0.67/0.68，跟 0.87 差了 20 个点。说明 prompt 里那个 faithfulness 定义起了关键作用。如果你部署时给的 prompt 不够具体，LM Judge 也会拉胯。

### 同一个 metric，step-level vs CoT-level 完全不一样

这个发现挺关键的：**没有一个 metric 能同时在 step 和 CoT 两个层级都 work**。

- CC-SHAP：CoT-level 0.70（最佳）→ step-level 0.41（**比随机还差**）
- Filler Tokens：step-level 0.59（最佳）→ CoT-level 0.50（随机）

如果一个 metric 真的在测 faithfulness 这个底层属性，应该在两个粒度上都有信号才对——CoT 的整体忠实度说到底就是步骤忠实度的某种聚合。但实测两层完全不挂钩，这强烈暗示：**这些 metric 测的根本就不是 faithfulness 本身，而是各自相关但不同的代理变量**。

### 为什么差这么多？预测分布的偏斜

光看 AUROC 容易误以为是模型不够好。但作者进一步拆解了预测分布：

![Figure 3：左中——CoT-level 和 step-level 的预测分布。IMP 类（橙）几乎全部预测 unfaithful，SEM 类（蓝）几乎全部预测 faithful。右——长度敏感性，Paraphrasing 和 SHAP 的 AUROC 随 CoT 长度变化最剧烈](https://www.mulanai.com/fs/files/0530_104382db_x4.png)

*图 3：左两栏揭示了为什么 AUROC 接近 0.5——这些 metric 的预测严重偏向某一类。IMP 类指标（Mistakes、Early、Filler、SCM）90–96% 都判 unfaithful，SEM 类（Paraph、Sim）94–96% 都判 faithful。右栏：CoT 每变长 10 倍，AUROC 的变化（Δ）。Paraphrasing 在长 CoT 上反而涨，SCM 跌得最厉害。*

这张图我盯了好一会儿。它告诉我们一件事：**这些 metric 不是在"判断"，是在"投票"——只不过每个党派一边倒**。

为什么 IMP 偏 unfaithful？原因不难想——你扰动 CoT 的某一个 step，要让答案变化是很难的，因为模型对长上下文有冗余。所以"扰动这一步答案没变 → 这一步不重要 → unfaithful"——这个逻辑链每一步都看似合理，但合在一起就把 importance 和 faithfulness 混淆了。一个 step 可以是 faithful 的（确实在描述模型真实推理）但同时不是 critical（删了答案也不变）。

为什么 SEM 偏 faithful？因为大部分 unfaithful CoT 也包含**足够 verbalized 的 reasoning** 让弱模型复现答案——unfaithful 不等于胡说八道，unfaithful 是没把"决定听提示"这种关键 internal 决策写出来。所以 paraphrase 后含义没变、simulatability 也没变。

这就是问题的本质：**现有 metric 设计时把 faithfulness 跟 importance / utility 混为一谈了**。

---

## 四、长 CoT 上更糟糕

这点对 reasoning model 时代特别要命。

![Figure 5：每个数据集上 step-level unfaithful 预测比例随 CoT 长度的变化。每条线是宽度 k=50 的滑窗。可以看到几乎所有 IMP 类 metric（SHAP、Early、Mistakes、FUR、Filler）的 unfaithful 比例都随 CoT 变长而上升或剧烈震荡](https://www.mulanai.com/fs/files/0530_206514fc_x7.png)

*图 5：横轴是 CoT 长度（log scale，从 10³ 到 10⁵ token），纵轴是 metric 标记为 unfaithful 的比例。问题是——按 ground-truth，长 CoT 并不一定更 unfaithful，但这些 metric 系统性地把长 CoT 标得更 unfaithful。*

这个发现对工业部署 CoT 监控的 implication 巨大。GPT-5、Claude 4、DeepSeek-R1 这些 reasoning model 动辄推几万 token——你用 IMP 类 metric 监控，长推理被无差别打成"不忠实"，警报响一片但没用。

### 不同模型规模上的表现

![Figure 8：每个 metric 的 AUROC 随模型规模的变化。横轴是参数量（4B/7B/30B/70B），分 instruct 和 thinking 两类，三个模型家族（Llama/Olmo/Qwen3）。基本没有一个 metric 显示出"模型越大测得越准"的清晰趋势](https://www.mulanai.com/fs/files/0530_49eaa6c7_x10.png)

*图 8：八个 metric 在不同模型规模下的 AUROC。注意几乎所有 metric 在 4B 和 70B 之间的差距都很小，没有 scaling law。CC-SHAP 在 Qwen3-4B instruct 上有个 0.72 的尖峰，但 30B 直接掉到 0.5——这个不稳定性让它很难作为部署 metric 用。*

这是另一个让人警觉的发现：**这些 metric 不会随模型变大而变得更准**。不像很多任务你只要等 scale 上去问题就消失了，faithfulness measurement 是个不会被 scale 自动解决的问题。

---

## 五、Metric 之间互相同意吗？

如果这些 metric 都在测同一个底层属性（faithfulness），那它们应该高度一致。

![Figure 7：Cohen's Kappa heatmap，CoT-level（左）和 step-level（右）。除了三个 IMP metric 之间有 0.18–0.35 的弱一致性，绝大多数 metric 对的 kappa 都在 ±0.05 内](https://www.mulanai.com/fs/files/0530_1f4d4b22_x9.png)

*图 7：Cohen's Kappa 衡量两个 metric 预测的一致程度（去除偶然吻合）。κ=1 完全一致，κ=0 跟随机一样，κ\<0 系统性相反。最高的 0.35 是 Adding Mistakes 和 Early Answering 之间——都是 IMP 类，互相借鉴。剩下大量 metric 之间 κ 接近 0，这是非常糟糕的信号。*

这就是关键。**如果这些 metric 真的在测同一个东西，kappa 不应该是这个样子**。它们之间不一致到这个程度，意味着每个 metric 都在测一个不同的东西，只不过都被作者起了"faithfulness"这个名字而已。

这是这篇论文里我觉得最 brutal 的一个发现——比 AUROC 低更 brutal。AUROC 低你还可以说"啊数据集太难了"。但 metric 之间不一致到 kappa≈0，这说明**整个领域过去几年定义的 metric 群体里就没有一个共同的 faithfulness 概念**。

---

## 六、计算开销有多吓人？

![Figure 2(b) 改图：每个 metric 单实例的 wall time（log scale，秒）。CC-SHAP 中位数 ~200 秒，最长上千秒；FUR 在 step-level 上中位数 100 秒；最便宜的 Sim 和 Filler 在 0.1 秒级](https://www.mulanai.com/fs/files/0530_f63deee7_x3.png)

*图 2(b)：注意纵轴是 log scale。最准的 CC-SHAP（CoT-level 0.70 AUROC）平均要 200 秒处理一个 CoT，最坏情况上千秒。这意味着如果你想在生产环境实时监控 CoT 是否 faithful，CC-SHAP 完全不可行。Filler Tokens 0.1 秒级倒是很快，但准确度只有 0.59。*

简单算一笔账：假设你部署一个 reasoning agent，平均每分钟处理 100 个 query。用 CC-SHAP 实时监控的话，理论上需要 200 秒×100 = 20000 秒的算力 / 60 秒——也就是要 333 个 monitoring worker 才跟得上一个推理 worker。这就是为什么作者说"prohibitively high computational cost"。

而 LM Judge（Skyline）的 0.87 AUROC 看起来很美，但你也得考虑——它需要再跑一个 frontier LLM 去判，算力开销 ≈ 推理本身。在大规模部署时，"用 LLM 监控 LLM" 也不是免费的方案。

---

## 七、这篇论文跟同期工作的差别

我搜了一下相关工作，做对比：

| 维度 | BonaFide（本文） | 之前的 plausibility-based 基准 |
|-----|------------------|--------------------------------|
| Ground truth 来源 | 任务设计强制可观察的中间计算 | 人类标注的 plausibility / 重要性 |
| 评估目标 | 元评估 metric 自身 | 评估 CoT |
| 标签层级 | step + CoT 双层 | 一般只 CoT |
| 模型覆盖 | 10 个，含 reasoning model | 通常 1–3 个 |
| 任务覆盖 | 13 个，含 hard outright tasks | 通常 1–2 个 |

跟 Tutek et al. (2025) 关于 FUR 的论文比，BonaFide 把 FUR 当成众多被评估对象之一；跟 Shen et al. (2026) 的 metric 评估比，BonaFide 不依赖 plausibility 代理。

有意思的是 **BonaFide 跟之前一些工作的结论分歧**：

- 之前的 benchmark 普遍认为 IMP 类 metric 表现不错。BonaFide 反过来——IMP 是表现最弱的一类。
- Shen et al. 认为 knowledge-intensive 领域 faithfulness 更难测；BonaFide 没看到这个 pattern。
- Chen et al. 2025a 认为 reasoning model 的长 CoT 更 faithful；BonaFide 部分支持——但只在 omission 类 unfaithfulness 上成立，commission 类（如 misattribution）反而更糟。

这些分歧基本都源自 ground-truth 来源不同。Plausibility-based 评估可能给了 IMP metric 不应得的好评。

---

## 八、对工程实践的几点判断

### 1. 不要单纯依赖任何一个 metric 监控 CoT

数据非常清楚：单 metric AUROC 都不到 0.7（除了 CC-SHAP 而它太慢）。如果你要在生产里做 CoT faithfulness 监控，至少要做两件事：

- **多 metric 投票**——但鉴于 kappa 这么低，朴素投票效果未必好
- **加 LM Judge 作为 high-stakes 复核**——0.82 AUROC 是目前最 robust 的，代价是算力

### 2. 警惕"重要性"和"忠实度"的混淆

如果你正在用 Adding Mistakes 或 Early Answering 这类指标做 reasoning model 评估——它们大概率告诉你"CoT 不忠实"，但其实是在告诉你"删了 step 答案也不变"。这两件事不是一回事。后者是冗余性问题，跟 faithfulness 关系没你想的那么直接。

### 3. 长 CoT 时代需要新的 metric 范式

reasoning model 主流化之后，CoT 长度从几百 token 涨到几万。BonaFide 显示几乎所有 metric 在长 CoT 上都退化或偏置加剧。你想想看——**为短 CoT 设计的 metric，搬到 reasoning model 上几乎全部失效**。

这是个开放问题，作者也没有给出新 metric——他们的核心贡献是"先把怎么测准这件事搞清楚"。

### 4. 给做评估的同学提个醒

如果你在 reviewer 那里看到一篇论文用 Adding Mistakes / Filler Tokens 测 CoT faithfulness、然后报"我们的方法 faithfulness 提升了"，建议至少问一下：你这个提升在 BonaFide 上还在吗？因为按 BonaFide 的数据，这些 metric 的提升很可能不反映真实 faithfulness 的提升。

---

## 九、这篇论文的局限和我的几点疑问

读完之后我也有几个不太确定的地方：

**第一，CoT-level 标签 85% unfaithful / 15% faithful 偏斜得很厉害。** AUROC 在严重不平衡的数据上会有失真。作者用了 DeLong CI，也用了 label-balanced subset 做某些分析，但主表的数字还是基于这个偏斜分布。我会想看看 PR-AUC 或者基于平衡子集的版本。

**第二，"任务设计能强制 ground truth"这个 claim 有边界**。Outright 任务里，"模型必须执行 bottleneck step" 是逻辑必然（禁 CoT 就只有 1.5% 答对率）。但 diversionary 里，"模型必然做出了听 hint 的决定"这个 claim 我觉得没那么严密——也可能模型从 hint 学到答案后通过别的路径"内化"了，根本没经过显式的"决定听 hint"的步骤。这种边界情况可能被错误标成 unfaithful。

**第三，pipeline 的 98.9% precision 是相对人类标注的。** 但人类自己对 faithfulness 也未必有完美一致——"Hint acknowledgment" 这种比较硬，"misattribution" vs "inert" 之间可能就没那么清晰。precision 数字漂亮不等于 ground truth 一定对。

**第四，CC-SHAP 拿到 0.70 这个最佳分，但作者也指出它在 step-level 是 0.41。** 这个分裂太剧烈了。我有点怀疑 CC-SHAP 在 CoT-level 的 0.70 是不是因为某种 spurious correlation——比如它的 SHAP 计算对长 CoT 给出的总分不稳定，然后 unfaithful CoT 平均更长，于是 SHAP 倾向把长的标 unfaithful。如果是这样，0.70 实际上是被 length confounder 带的。文中没有完全排除这点。

这些都不是"论文不行"，反而是 meta-evaluation 论文留下的好问题——证明这个方向值得继续做。

---

## 十、总结

这篇论文做了一件该有人做但一直没人做的事：**用 ground-truth 标签真刀真枪测一遍 CoT faithfulness metric**。

结论是：**现在常用的 8 个 metric，单论 AUROC 几乎都接近随机；最好的 CC-SHAP 也只有 0.70 而且贵得离谱；step 和 CoT 两个层级的 metric 不能互通；长 CoT 上还会进一步退化**。

如果你信这套数据，那基本意味着——过去几年所有用这些 metric 验证 CoT faithfulness 的工作，结论可能都需要打个问号。这是一篇"重置 baseline"的论文，价值不在于提出新方法，而在于把行业标准重新校准。

往前看，几个开放问题：

1. **能不能设计一个更便宜的 high-AUROC metric**？CC-SHAP 0.70 但要 10³ 秒，LM Judge 0.87 但要跑大模型。中间地带空着的。
2. **能不能跨 step / CoT 一致**？如果一个真正测 faithfulness 的 metric 应该在两层都有信号。
3. **reasoning model 的长 CoT** 是 metric 退化的主战场——专门为它设计的 metric 还没出现。
4. **BonaFide 自己的标签生成 pipeline 也是 LLM 判 LLM**——长期看需要把 ground truth 进一步推到 mechanistic level（如 activation patching），不能永远停在 verbal level。

总之，做 CoT 可解释性、AI safety、reasoning model 评估这几个方向的同学，强烈建议读一下原文。这篇可能不会改变你接下来一周的工作，但会改变你接下来一年怎么看待 faithfulness 这个词。

---

觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。
