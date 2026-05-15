# SkillOS：与其训练 Agent 自己变强，不如训练一个专门管"技能库"的小模型

## 核心摘要

Agent "自我进化"的主流套路是从过去轨迹里 distill 出 reusable skill 丢回 prompt。但**谁来管理这个 skill 库？** 大部分方案靠人工策展、固定启发式，或者用 RL 教 agent 怎么用 skill。SkillOS 走了一条不一样的路：**冻住 executor，单独训一个"技能策展员" Skill Curator，用 GRPO 让它学会 insert/update/delete**——什么时候写新 skill、什么时候改旧的、什么时候删过时的。结果是 8B 的 curator 接到 Gemini-2.5-Pro 这种 frontier executor 上，ALFWorld 平均成功率从 66.4% 涨到 80.2%，AIME+GPQA 平均准确率从 81.8 涨到 88.6。更有意思的是，这个 8B curator 比直接拿 Gemini-2.5-Pro 当 curator 还要强。问题不算"颠覆性"，但它把"agent 自我进化"拆出了一个工程化的中间层——**技能管家本身可以被独立训练**——值得做长期任务系统的人花时间看。

---

## 论文信息

- **标题**：SkillOS: Learning Skill Curation for Self-Evolving Agents
- **作者**：Siru Ouyang, Jun Yan, Yanfei Chen, Rujun Han, Zifeng Wang, Bhavana Dalvi Mishra, Rui Meng, Chun-Liang Li, Yizhu Jiao, Kaiwen Zha, Maohao Shen, Vishy Tirumalashetty, George Lee, Jiawei Han, Tomas Pfister, Chen-Yu Lee
- **机构**：Google Cloud AI Research、University of Illinois Urbana-Champaign、MIT
- **arXiv**：[2605.06614](https://arxiv.org/abs/2605.06614)

---

## 一、从一个真实的工程痛点聊起

我之前做 long-horizon agent 项目的时候碰到过一个特别窝火的现象：agent 上线一个月，犯过的错还是那几个。日志里清清楚楚记着——上次在某个工具调用上栽过跟头，这次换个相似 case，照样栽。

这就是当前 LLM agent 最难受的一点：**每次都是 one-off 的问题求解器，没法真正从经验里学东西**。给它再多的工具、再长的 context window，下次冷启动还是从零开始。

学术界给这个问题取了个名字叫 **self-evolution**。落地层面比较公认的一条路是：把过去的执行轨迹蒸馏成可复用的 **skill**，存到一个外部库里，下次碰到相似任务先去检索。Anthropic 2025 年开源的 Agent Skills 用的就是这个范式——一个 skill 就是一个 Markdown 文件，YAML frontmatter 写描述，body 写步骤和约束。

但这套范式有个绕不开的问题：**谁来决定 skill 库里写什么、改什么、删什么？** 主流方案三种：

1. **人工策展**——Anthropic 官方 skills 库的玩法，质量高但 scale 不动。
2. **启发式规则**——"碰到失败 case 就总结一条经验插进去"，A-Mem、Alita 走的路子。规则是死的，没法根据 executor 真实需求调整。
3. **RL 优化 skill 系统**——SkillRL、D2Skill 训的是"如何使用 skill"。最近也有工作训"如何管理 skill"，但 horizon 太短，reward 集中在"插入"这种立竿见影的操作上，**对"修改"和"删除"这种延迟反馈的复杂操作，几乎没有学习信号**。

SkillOS 想解决的就是第三类问题：**如何让一个模型真正学会"长程的、有判断力的"技能库管理。**

---

## 二、核心架构：把 Executor 和 Curator 拆成两个 Agent

SkillOS 的整体设计就一张图：

![图1：SkillOS 整体架构与 skill 文件格式](https://arxiv.org/html/2605.06614v1/x1.png)

*图1：左边是流式技能策展闭环，右边是 skill 的具体表示。Agent Executor 是冻结的（雪花标记），Skill Curator 是可训练的（火焰标记），中间通过 SkillRepo 这个外部 Markdown 文件库通信。*

系统就两个角色：

**Agent Executor（冻结）**：每来一个任务 $x_t$，用 BM25 从 SkillRepo 检索相关 skill 子集 $\tilde{\mathcal{S}}_t$，基于 task 描述和 skill 执行，产生轨迹 $\xi_t$。

**Skill Curator（可训练）**：任务结束后看到执行轨迹 $\xi_t$、自我判断的对错信号 $\mathbb{1}_{\xi_t}$、相关 skill 子集 $\tilde{\mathcal{S}}_t$，输出一串 curation 操作 $c_t = (u_t^1, \dots, u_t^{M_t})$——每个操作是 `insert_skill` / `update_skill` / `delete_skill` 三选一的函数调用。

skill 格式遵循 Anthropic 的 SKILL.md 范式：单文件 Markdown，YAML frontmatter 写 name 和 description，body 写 workflow、约束、注意事项。论文做了简化——单文件，无支持脚本和层级化组织（Limitation 里坦诚提到了）。

为什么叫 SkillOS？因为他们把 SkillRepo 当文件系统管，curator 只能通过三个 file I/O 风格的 function call 改它：

![图2：三种 skill 操作的函数签名](https://arxiv.org/html/2605.06614v1/x8.png)

*图2：`new_skill_insert`、`skill_update`、`skill_delete` 三个工具的参数定义——标准的 file system 风格 API。*

这个 action space 是离散、可枚举、可校验的，所以**reward 层面可以单独给"是否生成了合法 function call"一个信号**。把 curator 和 executor 解耦的另一个好处更工程化——**curator 是 executor-agnostic 的**，训出来的 8B curator 可以直接接到 Gemini-2.5-Pro 上，相当于一个"通用技能管家"。

---

## 三、训练配方：把延迟反馈变成可学习的信号

SkillOS 最核心的贡献其实不是架构，而是**怎么给 curator 提供学习信号**。

难点在哪？说实话我第一反应就是：**curator 的决策反馈是高度延迟和间接的**。你今天往库里插一条 skill，它的好坏要等下周某个相关任务来了、executor 检索到、用它解题、最后看对错——才能算出"这条到底有没有用"。

经典的 credit assignment 问题。论文用两个组合拳应对：**任务分组** + **复合 reward**。

### 3.1 训练实例就是一组"相关任务流"

![图3：SkillOS 训练流水线](https://arxiv.org/html/2605.06614v1/x2.png)

*图3：每个训练 step 采样一个 task group（左下方），里面的任务有强 skill 依赖。SkillRepo 从空开始，curator 顺序处理每个任务，policy 用 GRPO 优化。右下方曲线展示了 curator 行为如何从"盲目插入"逐渐演变到"删除冗余、更新已有、组织高质量内容"。*

三步走：先用 Gemini-2.5-Pro 给每条任务打**多维属性标签**（topic、common pitfall、需要的技能、依赖概念）；然后按属性相似度把任务集切成 task group，每组内任务有非平凡的 skill 依赖；训练时每 step 抽一个 group，**SkillRepo 从空开始**，curator 顺序处理。

关键在于：第一个任务没法借力（库是空的），但从第二个开始，前面 curation 的好坏会直接影响后面任务的成败——**"延迟反馈"被压缩成了一个 group 内可观测的信号**。同期一些工作（abs-2512-17102 之类）horizon 太短或任务独立，curator 学不到"修改旧 skill"、"删除有害 skill"这种长程操作。SkillOS 的 group 设计让"之前插的那条其实写错了，得改一下"这种判断变成可被 reward 覆盖的行为。

### 3.2 复合 reward：把 task outcome 拆成四个信号

光一个 task outcome reward 还不够，GRPO 的方差会把模型带偏。论文设计了四项复合 reward：

$$r = r^{\text{task}} + \lambda_f \cdot r^{\text{fc}} + \lambda_u \cdot r^{\text{cnt}} + \lambda_c \cdot r^{\text{comp}}$$

- **$r^{\text{task}}$（任务结果）**：从第二个任务开始算平均成功率，$r^{\text{task}} = \frac{1}{|G|-1}\sum_{i=2}^{|G|} \mathbb{1}(\xi_i)$。最直接的 executor-grounded 信号。
- **$r^{\text{fc}}$（function call 合法性）**：curator 生成的工具调用中合法且能执行的比例。cheap 但有效防止早期乱来。
- **$r^{\text{cnt}}$（内容质量）**：用外部 judge（Qwen3-32B）给每次 curation 的 skill 质量打分。
- **$r^{\text{comp}}$（压缩率）**：$r^{\text{comp}} = \frac{1}{|G|}\sum_{i=1}^{|G|}\left(1 - \frac{|\mathcal{S}_i|}{|\chi_i|}\right)$，鼓励 curator **不要把原始轨迹一字不漏复制进库**，而是真正抽象出可复用 skill。

这四项分工很清楚：$r^{\text{cnt}}$ 和 $r^{\text{comp}}$ 是中间反馈，缓解 task outcome 的延迟稀疏；$r^{\text{fc}}$ 防 reward hacking；$r^{\text{task}}$ 才是下游对齐目标。算法用 GRPO，主要图稳定性和 sample efficiency。

我自己看到这个 reward 设计的时候，感觉作者是真的在做工程——每一项都对应一个具体的失败模式（行为崩溃、格式错误、低质 skill、把轨迹当 skill 存）。这种拆解能力是 RL 系统能跑起来的关键。

---

## 四、实验：8B Curator 居然比 Gemini-2.5-Pro Curator 还强

实验跨了两个大类任务、三个 executor backbone，数据量很扎实。先看主结果。

### 4.1 ALFWorld：跨 executor 全面提升

![图4：跨任务训练-测试矩阵的相对提升](https://arxiv.org/html/2605.06614v1/x3.png)

*图4：三个 executor backbone 上的训练任务 × 测试任务转移矩阵。绿色越深表示相对 No Memory 提升越大。对角线和多数跨任务格子都是绿色——curator 不仅训练域有效，还能跨任务迁移。*

ALFWorld 关键数据（Avg. SR / Steps，三 seed 平均）：

| Executor | No Memory | ReasoningBank | MemP | SkillOS-gemini | **SkillOS** |
|---|---|---|---|---|---|
| Qwen3-8B | 47.9 / 21.1 | 55.7 / 20.1 | 49.7 / 21.0 | 50.7 / 20.8 | **61.2 / 18.9** |
| Qwen3-32B | 54.5 / 20.3 | 61.4 / 18.7 | 55.7 / 20.0 | 63.6 / 18.1 | **68.6 / 17.3** |
| Gemini-2.5-Pro | 66.4 / 17.7 | 71.4 / 16.0 | 74.3 / 15.2 | 79.3 / 14.9 | **80.2 / 14.8** |

几个值得停下来想的点：

Gemini-2.5-Pro 这种 frontier model 也能从 +13.8 个点的提升中受益（66.4 → 80.2），说明**好的 procedural memory 对再强的 executor 都有边际价值**——单纯 scale up executor 替代不了经验积累。

8B 的 SkillOS 比 SkillOS-gemini（直接拿 Gemini-2.5-Pro 当 curator）平均都强：Qwen3-8B 上 61.2 vs 50.7，Qwen3-32B 上 68.6 vs 63.6。**RL 训练带来的提升超过了 curator backbone 从 8B 到 frontier 的提升。** 挺打脸"模型越大越好"直觉的。

MemP 在 Qwen3-8B 上甚至不如 No Memory（49.7 vs 47.9）。**手工启发式在 executor 能力变弱时是脆的**——它假设 executor 能"理解"那些被精心管理的 memory，弱 executor 根本用不好。

### 4.2 WebShop & 单轮推理：跨任务类型也成立

| Executor | 任务 | No Memory | ReasoningBank | **SkillOS** |
|---|---|---|---|---|
| Qwen3-8B | WebShop SR | 9.8 | 11.4 | **16.5** |
| Qwen3-8B | Reasoning Avg Acc | 69.6 | 69.6 | **73.8** |
| Qwen3-32B | Reasoning Avg Acc | 74.0 | 74.9 | **79.7** |
| Gemini-2.5-Pro | Reasoning Avg Acc | 81.8 | 83.5 | **88.6** |

WebShop Qwen3-8B SR 从 9.8 涨到 16.5 差不多翻倍；reasoning 在 Gemini-2.5-Pro 上 81.8 → 88.6，对 frontier 模型来说算很能打了。

我有点皱眉的地方是 WebShop 在 Gemini-2.5-Pro 上 SR 只涨了 +2.9 个点（38.4 → 41.3）。可能是因为 WebShop 的成功比较依赖"商品检索 + 属性匹配"这种细粒度交互能力，curator 能给的指导有上限。

### 4.3 Ablation：哪个设计真的关键

| 配置 | Avg. SR | Steps |
|---|---|---|
| **SkillOS-GRPO（完整）** | **61.2** | 18.9 |
| w/o $r^{\text{cnt}}$ | 58.6 | 20.1 |
| w/o $r^{\text{comp}}$ | 60.0 | 19.3 |
| w/o grouping | 57.3 | 20.6 |

**最大掉点来自去掉 task grouping**（-3.9）。这印证前面分析的：grouped task stream 才是 SkillOS 能学会"长程 curation"的真正引擎。如果任务彼此独立，curator 永远学不会"修改旧 skill"和"删除冗余 skill"。去掉 content quality reward 也掉 2.6 个点，纯靠下游 task outcome 的稀疏信号不够，中间 judge 反馈是有效补强。

---

## 五、Curator 学到了什么？四张图看 RL 怎么改变行为

这部分是论文最有意思的地方，比主表更说明问题。

### 5.1 操作分布：从"疯狂插入"到"主动 update"

![图5：训练过程中 curator 的三种 skill 操作占比](https://arxiv.org/html/2605.06614v1/x4.png)

*图5：横轴训练 step，纵轴 insert/update/delete 占比。训练初期 insert 占绝对主导（~80%），update 从 ~10% 涨到 ~25%，insert 相应下降到 ~60%。delete 始终小比例但有缓慢上升。*

这张图告诉你的是：**curator 的行为模式是真的在 RL 过程中"涌现"的**。早期看到啥都觉得"这是新 skill，插进去"；中期发现老 skill 可以改得更好，于是更多 update；delete 一直是小动作，但能慢慢学会——这跟 $r^{\text{comp}}$ 鼓励紧凑 SkillRepo 的设计吻合。这种"行为演化"在启发式 memory 系统里是不可能出现的，因为规则是写死的。

### 5.2 Skill 内容的演化：从废话到 meta-skill

![图6：训练过程中 skill markdown sections 与类型的演化](https://arxiv.org/html/2605.06614v1/x5.png)

*图6：（a）新出现的 markdown section 占比——早期"额外指导/提示"这种泛泛 section 占比高，随训练推进显著下降；（b）SkillRepo 里 skill 类别构成——"任务对象特定"这种琐碎 skill 在 20 step 后萎缩，"meta strategy"和"通用 action"稳步上升，最终 meta strategy 占约 29%。*

RL 训练不仅改变了 curator 做了多少操作，还改变了它**写 skill 的"风格"**——从堆砌"建议、提示、注意事项"，慢慢学会抽象出"系统性搜索失败时该怎么处理"这种 meta-strategy。

### 5.3 Skill 利用率：训练后的 skill 真的更好用

![图7：四类 skill 利用率指标的对比](https://arxiv.org/html/2605.06614v1/x6.png)

*图7：(i) skill 被引用率：SkillOS-base 87.9% → SkillOS 100%；(ii) 平均调用频率：53.6 → 61.2；(iii) skill 引用准确率：72.9% → 88.6%；(iv) 任务平均使用 skill 数：2.24 → 1.55。*

最反直觉的是 (iv)：**SkillOS 平均每任务用的 skill 反而更少了**（2.24 → 1.55），但前三项说明用得更准、更频。合起来读就是：SkillRepo 不再是"一堆模糊重叠的 skill 让 executor 自己挑"，而是"少数高质量、目标明确的 skill 被精准调用"。这种"密度上去、冗余下来"的状态，对 BM25 这种简单检索特别友好。

### 5.4 Case study：curator 学到的是怎样的 skill？

![图8：SkillOS 为不同任务类型策展的 skill 案例](https://arxiv.org/html/2605.06614v1/x17.png)

*图8：（a）agent 任务的 skill——"用替代品处理缺失对象"的 fallback 策略，分四步流程（穷尽搜索 → 确认不可得 → 选择替代品 → 用替代品继续），且显式引用其他已有 skill；（b）推理任务的 skill——"内心率/外接圆/半周长关系"，不是单一公式，而是包含两条解题路径（角度切入、边长切入），每条标了公式、应用场景和前置约束。*

这两个例子比所有数字都说明 SkillOS 的价值。agent skill 学到的不是"找苹果就去厨房"这种 task-specific 知识，而是"找不到目标对象时的通用恢复 workflow"；reasoning skill 也类似——不是"题目 X 的解答步骤"，而是"几何关系的多种切入路径"。这才是 procedural memory 该有的样子：可复用、可组合、显式约束。SkillOS-base（只 prompting）产生的 skill 更像"把解答抄一遍稍微改改命名"，根本没法复用。

---

## 六、几个需要冷静看的点

写到这儿，得给这篇论文降降温。

**Executor 冻结，既是优势也是上限**。Curator 只能通过 SkillRepo 间接影响 executor，没法解决"executor 本身就理解错了"的问题。这个解耦让方法变得 modular，但天花板被 frozen executor 钉死。

**BM25 检索是严重的简化**。SkillRepo 一旦扩展到几千个 skill，检索本身会成为新瓶颈。论文 Future Direction 里提到了 agentic search 这条路，但那就是另一篇论文了。

**Task grouping 依赖 Gemini-2.5-Pro 打多维属性标签**。这个标注成本论文没怎么强调，工业界落地时要么有现成标签，要么自建标签体系——都不便宜。

**评测公平性的小瑕疵**：SkillOS 8B 比 SkillOS-gemini 强这件事，是"训练 vs 不训练"的对比，不是"模型能力对比"。如果 SFT 一下 Gemini，能不能追上？论文没回答。

但这些问题不影响整体判断：**它把"agent 自我进化"拆出了一个干净、可独立优化的模块**，而且证明了——**专门训一个小 curator，比把 frontier 大模型当 curator prompt 还要好。**

---

## 七、对工程实践的启发

如果你在做长程 agent 系统，几条可以直接拿走：

- **把 memory 管理拆成独立 service**。哪怕暂时不训 RL，未来想升级（人工 → 启发式 → RL）也不用动 executor。
- **task grouping 这个数据组织方式可以借用**。即使不训练，把测试集组织成"相关任务群"做评估，更能反映 memory 系统的真实价值。
- **内容质量信号别只靠 task outcome**。用一个独立 LLM judge 给 memory 操作打分作为中间 reward，能显著降低低质 memory 比例。
- **弱 executor 更需要 learned curation**。MemP 在 Qwen3-8B 上甚至不如 No Memory 这件事说明，手工启发式 memory 在小模型上大概率是负向的。
- **8B curator 能提升 frontier executor 这件事告诉你：memory 是一个"通用 capability layer"**。可以训一个跨业务的 curator，接到不同业务线的 executor 上——modular 性的工程价值可能比刷 benchmark 更值钱。

---

## 八、写在最后

看到 SkillOS 之前，我对"agent 自我进化"这类工作有点疲劳——大多数论文要么把过去的 trace 塞回 prompt 换个名字，要么给 memory 加几条启发式规则。

SkillOS 让我重新打起精神在于：**它没有发明新的 memory 表征，也没有设计更复杂的检索算法**。它只是认真问了一个问题——"既然需要一个会管理技能的 agent，为什么不直接把它训出来？" 然后用最朴素的工具（GRPO + 复合 reward + grouped task stream）给出了能跑通的答案。

技术上没有惊天动地的创新，但**问题分解漂亮、工程扎实、结果说明问题**。如果你在做 long-horizon agent 或 procedural memory，这篇论文值得花一个下午把第三章的 reward 设计和第五章的行为分析嚼透。**它给出的不只是一个 SOTA 数字，而是一种"如何把延迟反馈的复杂决策训练出来"的可复用方法论。**

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
