# Prefix Sliding for efficient test-time scaling 论文解读

> **arXiv**: [2608.26070](https://arxiv.org/abs/2608.26070)（2026-08-26 提交）
> **作者**: Niklas Muennighoff, Zhengyang Wang, Zeyi Chen, Weijia Shi, Binyuan Hui, John Yang, Dapeng Jiang, Mika Senghaas, Fares Obeid, Johannes Hagemann, Sami Jaghouar, Ludwig Schmidt, Percy Liang, Jason Wei, Andrew Y. Ng, Luke Zettlemoyer, Yejin Choi, Mike Lewis
> **代码**: https://github.com/Muennighoff/prefix-sliding
> **一句话总结**: 推理时只保留"前缀（指令/工具）+ 最近滑动窗口"的注意力，让每生成一个新 token 的成本恒定，免训练提速 3 倍且性能不降，配合 RL 训练可扩展至 10 万 token 以上的推理轨迹。

---

## 1. 动机：长推理的内存与算力困境

Test-time scaling（测试时扩展）通过让模型"想得更久"来提升性能，是目前提升推理能力的主流路径之一。但标准 Transformer 使用全注意力（full attention），必须把**整条推理轨迹保留在 KV cache 中**，每生成一个新 token 的计算成本随已生成 token 数**线性增长**。遇到需要极长思考（几万甚至几十万 token）的难题时，成本将变得难以承受。

长上下文还会带来附加问题：被旧的无关节 token 干扰（distraction）、context poisoning、重复循环、知识丢失等。

### 两个关键观察

1. **中间推理 token 很快失去重要性**。例如计算 `((42 + 84) × 4) - 5`：一旦完成 `42 + 84`，该步骤的推理过程就不再需要，只有结果对后续步骤重要。
2. **前缀和最近的推理 token 至关重要**。前缀包含系统指令与任务 prompt（工具信息、任务目标），同时充当 attention sink；最近的 token 则反映模型当前的工作状态。

### 证据：注意力分布分析（Figure 2）

对 Qwen3-1.7B 在 AIME25 推理轨迹上、跨层与注意力头平均的 post-softmax 注意力概率分析显示：

- **前 4 个 token** 吸收了大量概率质量（attention sink 作用）；
- prompt 中其他 token 也比中间推理 token 获得更多注意力；
- `<think>` 分隔符获得高注意力（可能因其持续指示"思考模式"）；
- 注意力在推理轨迹中段持续走低，**在末尾急剧上升**，尤其是紧邻待生成 token 的前一个 token。

这直接指向一个结论：**保留全部中间推理 token 并不值得其内存与算力代价**。

---

## 2. 方法：Prefix Sliding

### 核心机制

推理过程中，内存中只保留两部分：

- **Prefix（前缀）**：系统指令 + 任务 prompt，含工具信息与任务目标；
- **Sliding window（滑动窗口）**：最近生成的几千个 token。

随着生成推进，窗口前移并丢弃窗口之外的旧中间 token。示例：40-token 系统指令 + 60-token 任务 prompt 构成 100-token 前缀；配 4096-token 窗口，内存中最多保留 **4196 个 token**。

无论模型已生成多少 token——百万级甚至十亿级——**每个新 token 的成本都是恒定的**（Figure 3）。这是实现超长时程 test-time scaling 的必要条件。

### 位置嵌入处理（Figure 4）

丢弃中间 token 后，剩余 token 的原始位置编号出现空洞，需要重新处理位置嵌入（PE）。两种方案：

- **Reset PE**：对剩余 token 重新应用位置嵌入；
- **Continue PE**：沿用 cache 中已应用 PE 的表示，更高效。

两者性能差异不显著（附录 D），作者采用 **Continue PE**。

### 免训练版本

开箱即用于现有生成式模型：与全注意力性能相当的同时**快 3 倍**。自定义 FlashAttention 内核的速度与常规滑动窗口内核大致相同（略慢源于前缀的额外内存）。

**吞吐行为（Figure 6）**：Prefix Sliding 与滑动窗口的 tokens/s 先下降后**稳定在约 5,000 tokens/s**——初期下降是因为生成初期"warm-up 阶段"（token 数尚未达到窗口大小）占比逐渐减少；达到窗口大小后每 token 成本恒定。而全注意力会**无限变慢**。

**性能来源（Figure 1）**：Prefix Sliding 表现更好并非因为每个 token 更优，而是**相同思考时间内能生成更多 token**。

**等内存对比（Figure 7）**：在相近内存预算下（全注意力上限 8,192 token vs. 8,192 窗口），Prefix Sliding 可**提升**性能——相同预算能支撑更长的推理。

### RL 训练版本

Prefix Sliding 也支持超长 RL rollout，避免常见的"截断并丢弃过长生成"做法。

**理论依据**：多层滑动窗口的理论感受野为 W×L（W=窗口，L=层数），但因信息瓶颈实际约为 **1.5×W**。因此反向传播 W 个 token 只需传入约 1.5×W 的前置 token 加前缀。

**两种反传方案（Figure 5）**：

- **Chunked backpropagation**：分块反传并累积梯度，接近全量反传；
- **Truncated backpropagation（实际采用）**：只对最后一个块反传。

具体例子：10 万 token 推理轨迹 + 2048 窗口时，只将最后 **8192 个 token** 送入 trainer；前 6144 个仅作上下文（通过 **loss mask** 置零损失），只对最后 2048 个计算 RL 损失。因梯度基于 4 倍窗口大小计算，与全 10 万 token 生成相比非常准确。训练侧同样使用 Continue PE。

### 自定义内核实现（两级过滤）

- **Intra-tile masking（块内掩码）**：对部分覆盖允许区域（prefix ∪ 窗口）的 tile 施加逐元素掩码，保证数学正确性且不改变 FlashAttention 分块策略；
- **Inter-tile skipping（块间跳过）**：完全落在允许区域外的 tile 直接跳过，重构 producer–consumer 流水线，分别迭代 prefix 块和窗口块两个不相交区间，效率基本对齐标准滑动窗口注意力。

---

## 3. 实验

### 评测设置

- **Benchmark**：GPQA、MATH500、AIME25（主实验）；LiveCodeBench（LCB）与 HealthBench 用于局限性分析。
- **协议**：64 次运行取平均（accuracy / avg@64）；temperature 0.6、top-p 0.95；simpleverify 验证答案；budget forcing 控制思考预算（不用 "Wait" token）；核心效率指标为**每样本平均思考时间（秒）**。
- **窗口扫描**：512 / 1024 / 2048 / 4096 / 8192 / 16384。
- **硬件/推理**：vLLM + FlashAttention，单块 80GB Nvidia H100，为 Hopper 架构编写自定义内核。

### 免训练结果

- 与全注意力**性能相当，速度快 3 倍**；
- 吞吐稳定在约 5,000 tokens/s，全注意力持续变慢；
- 等内存预算下（8,192 token）性能优于全注意力。

### RL 训练设置与结果

| 项目 | 内容 |
|---|---|
| 模型 | Qwen3-1.7B（默认）；附录 E 用 7B 模型验证 |
| 算法 | GRPO：trl（同步）与 prime-rl（异步） |
| 反传策略 | 全量反传 vs. 只反传最后一个滑动窗口（向 trainer 传 4× 窗口大小的末尾 token） |
| 数据 | 自建数学题数据集，按可猜测性、可验证性、难度三标准过滤 |
| 超参 | 与 Prefix Sliding 无关的超参不调优、跨对比固定 |

**结果**：

- RL + Prefix Sliding 在相近内存预算下支持**超过 100,000 token 的推理长度**，从而获得更高奖励；
- **Trainer token 数消融（Figure 8）**：窗口 2048、最大生成 16,384。只传 2K 时 KL 散度 >0.1；传 4K（2×）显著降低失配；**8K（4×）**与 16K（8×）大致相当，故主实验用 4×。残余 KL 源于自定义 FlashAttention（生成端）与 FlexAttention（训练端）的微小数值差异；
- 附录 E：7B 模型、窗口 8192、乘数 4 的 RL 训练，控制序列长度时**性能与全注意力相当**。

### 消融：对比其他压缩方案（Figure 9，AIME25；最大生成 262,144，窗口 4096）

| 方法 | 机制 | 优缺点 |
|---|---|---|
| **Last k** | 达到阈值后删除除最后 k 个 token 外的所有内容 | 快但可能浪费 token；k 大需重复处理，k 小丢失有用近期 token；内存占用锯齿状波动，难以充分利用算力。代表：agent 删旧轮次、Markovian Thinking / Delethink |
| **Summary** | 达到阈值后总结全部文本，以总结+prompt 开启新窗口 | 理论上可保留整个上下文信息，但实践中模型难以跨多轮保留关键信息；引入更多超参（阈值、总结长度、总结 prompt/模型/位置）；额外总结开销。代表：Opus 4.6、GPT 5.4、Composer 的 compaction |
| **Vanilla sliding window** | 等价于无前缀的 Prefix Sliding | 最简单，但丢失任务信息（忘了在解什么题、可用什么工具），长推理任务上表现差。gpt-neo、gpt-oss 等靠穿插全注意力层补偿 |

**结论**：Prefix Sliding 提供**最佳的性能-效率权衡**，且只增加一个超参（窗口大小）。纯滑动窗口在长思考时间下迅速触顶；Last k 和 Summary 能达到不错性能，但受限于固有的 token 重复处理与额外总结步骤，且有更多超参与生成重启的复杂性开销。

---

## 4. 关键图表解读

| 图号 | 主题 | 含义 |
|---|---|---|
| Figure 1 | 免训练即比全注意力高效 | 相同思考时间生成更多 token 故性能更好；Qwen3，窗口 4096 |
| Figure 2 | 开头与结尾 token 获得最多注意力 | 核心动机证据：前 4 token 为 attention sink，末尾注意力急剧上升 |
| Figure 3 | 高效长时程 test-time scaling | 每步成本恒定 vs. 全注意力持续增长至不可承受 |
| Figure 4 | 位置嵌入处理 | Reset PE 与 Continue PE 两方案 |
| Figure 5 | 长推理轨迹的反传 | chunked 与 truncated 两种反传方案 |
| Figure 6 | 内核吞吐对比 | 滑窗类稳定约 5,000 tokens/s，全注意力持续变慢（H100，窗口 4096） |
| Figure 7 | 等内存预算提升性能 | 8,192 预算下 Prefix Sliding 可推理更久从而更好 |
| Figure 8 | 截断反传数值分析 | 4×（8K）trainer token 为性价比最优 |
| Figure 9 | 优于其他替代方案 | AIME25 上对比 Last k / Summary / 纯滑窗 / 全注意力 |
| Figure 10 | 极限下成本恒定 | "L" 点后成本有界；Last k/Summary 呈锯齿形成本尖峰 |
| Figure 11 | LCB 需窗口 ≥16,384 | 局限性：代码任务中函数开头可能滑出窗口 |
| Figure 12 | 快速任务获益少 | HealthBench 平均仅 2,086 token，窗口 2048 时几乎无加速空间 |

---

## 5. 局限性（论文自述）

1. **对比范围有限**：仅与"开箱即用 + 单 token 成本有界"的方法比较，未含 RNN/SSM 等替代架构；
2. **信息丢失风险**：某些任务（如 LiveCodeBench）中间 token 仍重要，需窗口 ≥16,384 才能匹配全注意力；
3. **短生成收益小**：存在滑动窗口 warm-up 阶段；HealthBench 平均仅需 2,086 token，窗口 2048 时几乎不滑动、等同全注意力；
4. **Agent/多轮场景**：超长工具输出可能淹没窗口，未来用户指令的归属（进 prefix 还是被滑走）尚无定论；
5. **规模上限**：当前验证至 7B 参数与数十万 token，更大规模留待未来工作。

---

## 6. 总结与思考

Prefix Sliding（arXiv: 2608.26070）用一个极简的观察——**推理中间 token 很快失去重要性，前缀与最近 token 才是关键**——把长时程 test-time scaling 的成本从"随轨迹长度线性增长"变为"每 token 恒定"。方法本身只引入一个超参（窗口大小），免训练即有 3 倍提速且性能不降，配合 RL 训练可把推理轨迹扩展到 10 万 token 以上。

值得关注的几点：

- **与滑动窗口注意力的本质区别在于保留 prefix**。实验表明纯滑窗在长推理中会因丢失任务目标而迅速触顶，prefix 保留是性能保持的关键；
- **训练-推理一致性**处理得很务实：4× 窗口大小的截断反传 + loss mask，在梯度准确性与训练成本间取得平衡；
- **局限也很真实**：对需要回溯长程中间信息的任务（如代码生成中的函数头），窗口必须足够大，否则会丢失关键内容；对短任务几乎无收益。

这一工作与 KV cache 压缩、context compaction、线性注意力等方向互补，为"无限长推理"提供了一个工程上立即可用的方案。
