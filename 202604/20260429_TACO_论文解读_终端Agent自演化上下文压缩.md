# 终端Agent的"上下文垃圾场"清理工：TACO 用一个不停进化的规则池替代手写 prompt

## 核心摘要

如果你做过终端Agent，应该对这个画面不陌生：一个 `apt-get install` 喷出几百行 Unpacking、一个 `objdump -d` 吐出几千行汇编、一个 `make` 把同样的 cp 命令打印 100 遍——这些噪声塞进上下文，token 二次方往上涨，模型反而越跑越糊涂。已有的解法要么是写死的 prompt 截断（generalize 不动）、要么是专门 finetune 一个 SWE-Pruner（只对 SWE-Bench 这种工作流好用）。

TACO 的思路挺干脆：把"压缩规则"本身做成一个会自我进化的池子。每个规则是一段结构化 JSON（trigger regex + keep/strip patterns + summary header），由 LLM 在执行轨迹里自动发现、修订、淘汰。一个全局规则池（Global Rule Pool）跨任务共享，每个任务再从池子里挑 Top-30 做局部细化。整个过程**完全 training-free**，不动模型权重。

效果上：在 TerminalBench 1.0/2.0 上给 DeepSeek-V3.2、MiniMax-M2.5、Qwen3-Coder-480B 等几乎所有主流大模型挂上去都涨 1–4 个点；MiniMax-M2.5 在 TB 1.0 上从 42.30 → 45.25，token 还省了 21%；同 token 预算下相对 baseline 也稳定多 2–3 个点。我的判断是这篇论文最值钱的不是某个具体规则，而是把"prompt 工程在线化、跨任务沉淀化"做成了一个可工程化的 loop——这个范式比方法本身更有借鉴意义。

---

## 论文信息

- **标题**：A Self-Evolving Framework for Efficient Terminal Agents via Observational Context Compression
- **作者**：Jincheng Ren, Siwei Wu, Yizhi Li, Kang Zhu, Shu Xu, Boyu Feng, Ruibin Yuan, Wei Zhang, Riza Batista-Navarro, Jian Yang, Chenghua Lin
- **机构**：University of Manchester、MAP、HKUST(GZ)、HKUST、Beihang University
- **arXiv**：[2604.19572](https://arxiv.org/abs/2604.19572)
- **代码**：[MAP/TACO](https://github.com/multimodal-art-projection/TACO)

---

## 为什么终端Agent会被自己的输出"淹死"

先讲一个我之前调试编译类 Agent 时碰到的真事。让模型跑 `make -j8` 编一个中等规模的 C++ 项目，本意是想看 link 阶段有没有 undefined symbol。结果模型把上千行 `g++ -c -O2 ...` 的命令展开当宝贝塞回上下文，第二轮想做下一步推理时，prompt 里 70% 都是无信息量的编译命令，剩下 30% 才是它真正该看的错误。

这种现象在论文里给了一个挺扎心的数据。Figure 1 (a) 把 50 条 TB 2.0 的轨迹拿出来人工标注"真正有用的文本"，然后跟原始 prompt 比 token 数：

![图1：上下文冗余度量与压缩方法对比](https://arxiv.org/html/2604.19572v1/x1.png)

*图1：(a) 三个主流模型在 TB 2.0 上原始 prompt vs 人工提取的有效文本，Qwen3-Coder-480B 冗余 30.5%、DeepSeek-V3.2 冗余 44.1%、MiniMax-M2.5 冗余 24.6%。(b) 在 TB 2.0 上，Baseline 23.9% → Seed Rule 21.6%（更差）→ High-Quality Rule 24.3%（小幅）→ LLM Summarize 20.3%（更差）→ TACO 27.6%（明显领先）。*

44.1% 的冗余率，意味着 DeepSeek 每跑一步终端命令，差不多有快一半的 token 是在"喂垃圾"。这不是边际优化的问题，是结构性问题。

更让我意外的是 Figure 1 (b)。作者拿 200 条人工写的高质量规则去跑 TB 2.0，accuracy 居然只有 24.3%，比 baseline 23.9% 高不到一个点。再看 LLM Summarize（用大模型现场总结终端输出）的 20.3%，比 baseline 还差 3.6%。说实话，看到这两个数我愣了一下——直觉上人写规则应该非常 work，LLM 现场总结也不至于亏 3 个点。

为什么会这样？作者的解释是终端环境太异质了，人写的规则在 A 仓库 work、到 B 仓库可能就过度压缩了；LLM 现场总结又会把关键的 error 和栈帧也给 paraphrase 掉，相当于丢信息。这个观察很重要——它直接告诉你"静态压缩 + 现场总结"这两条路都走不通，必须有一种**能根据当前任务动态调整、又能在跨任务上累积经验**的机制。

这就是 TACO 切入的位置。

---

## 方法：把 prompt 工程做成一个永不停下的训练循环

TACO 的核心抽象是**规则**（Rule）。每个规则是一个结构化 JSON，长这样：

```json
{
  "rule_id": "seven_zip_extraction",
  "trigger_regex": "\\b7z\\b.*\\s+secrets\\.7z\\b",
  "keep_patterns": ["\\bError:", "Wrong password", "Everything is Ok",
                     "Extracting\\s+.*secret_file\\.txt"],
  "strip_patterns": ["^\\s*[0-9]+ files?,", "^\\s*Size:\\s+",
                      "^\\s*Processing archive:"],
  "keep_first_n": 5, "keep_last_n": 5,
  "confidence": 1.0, "times_applied": 126, "times_complained": 0
}
```

注意一下，规则本身不是一段神秘的神经网络权重，就是一组 regex + 保留/丢弃模式。**真正的"自演化"发生在这些 JSON 上**——LLM 负责生成、修订、淘汰这些规则，整个过程不更新任何模型权重。

整个 pipeline 长这样：

![图2：TACO 整体架构概览](https://arxiv.org/html/2604.19572v1/x2.png)

*图2：TACO 三大模块。左边是规则初始化（从全局规则池 Top-k 检索 + LLM 选择/精炼），中间上面是 Agentic Workflow（agent 执行命令拿到原始观察，TACO 做匹配-过滤-输出三步压缩），中间下面是规则更新（spawn or modify rules），底下是 Global Rule Pool Update（任务结束后回写有效规则到全局池）。*

这张图信息量蛮大的，我拆成几块讲。

### 任务开始：从全局池里挑规则

有新任务进来时，TACO 不会从零开始。它先从**全局规则池** $R_g$（持久化存储所有跨任务积累下来的规则）里取 Top-30——按 ranking score $R_{gs}(r) = c_r^g \cdot (n_r + 1)$ 排序，其中 $c_r^g$ 是全局 confidence，$n_r$ 是历史成功应用次数。

挑出来这 30 条候选之后，再让一个 LLM 看着任务描述去**选择和精炼**。比如这个任务是"用 R 跑一个统计采样代码"，LLM 就会保留 `pip_install`、`apt_install` 这种通用规则，再生成一条 task-specific 的 `pytest_stochastic_testing_mod`，专门用来过滤 pytest 的 PASSED 行但保留 KS 检验的 p-value。

这一步其实很关键。前面 Figure 1 (b) 证明了人写的高质量规则不 work，但 LLM 在**当前任务上下文**里现场挑选规则、做 task-specific 修改，就能避免"通用规则过度激进"的问题。这是一个挺漂亮的设计——把 LLM 的"智能"用在它真正擅长的地方（理解上下文做适配），而不是用在它不擅长的地方（手撕几千字符的日志）。

### 执行中：匹配 → 过滤 → 输出 + 错误时绕开

每一步 agent 跑完命令拿到 raw observation $O_t$，TACO 干两件事：

1. 检测 $O_t$ 是不是 **Critical**（含明显的 error/traceback/syntax error 这类信号）。是的话直接放行，**不压缩**。
2. 不是 Critical 才走规则匹配——遍历当前任务的 active rule set $R_t$，找到 trigger 匹配的规则，按 keep/strip patterns 过滤，输出 $\tilde O_t$。

公式上就是：

$$\tilde{O}_{t}=\begin{cases}O_{t},&\text{if }O_{t}\text{ is Critical},\\ F_{R_{t}}(O_{t}\mid C_{t}),&\text{otherwise.}\end{cases}$$

这个 Critical-bypass 其实是工程上很值得注意的细节。压缩 + 错误信息 = 灾难——你错误堆栈给截了，agent 下一步根本无法定位 bug。所以无论规则有多激进，只要看到 error/traceback 这种红线模式，立刻让原文通过。这是个非常老实的、贴近真实工程经验的设计。

### 执行中：规则的在线进化（Intra-Task Rule Set Evolution）

执行过程中规则不是一成不变的。TACO 有两个触发机制：

**Add（新增规则）**：如果某条 raw observation 没有任何 active rule 能匹配，就调 LLM 生成一条新规则，加进 $R_t$。

**Update（更新规则）**：通过 agent 的"隐式反馈"判断规则是不是过度压缩了。比如 agent 在压缩输出后**重新执行了同一条命令**、或者**显式说"我需要看完整输出"**，TACO 就把刚才触发并改动了输出的规则定为"被投诉"，suppress 掉，并让 LLM 生成更保守的替代版本。

我特别欣赏这个"隐式反馈"的设计。你不需要让 agent 多说一句"我抗议这条规则"，只要它的行为模式表明"信息丢了"，TACO 就能自动捕捉。这是把 RL 里的 reward signal 概念翻译到 prompt-engineering 世界的一个聪明做法。

### 任务结束：写回全局池 + 收敛检测

任务跑完，TACO 把符合两个条件的规则回写到全局池 $R_g$：
- $\Delta n_r \geq 1$：在本次任务里至少成功应用了一次
- $c_r^t \geq \tau$：任务结束时的 confidence 大于阈值

被投诉过的规则 $c_r^t = 0$，**直接从全局池删除**，不让它污染未来任务。

收敛靠一个叫 **Retention** 的指标判断：

$$\mathrm{Retention}^{(i)}_{K}=\frac{\left|\mathrm{TopK}\!\left(R_{g}^{(i-1)}\right)\cap\mathrm{TopK}\!\left(R_{g}^{(i)}\right)\right|}{K}\times 100\%$$

意思是连续两轮训练后，全局池里 Top-30 规则的重合度。重合度高说明"什么是好规则"已经稳定下来了。论文里 K=30、阈值 90%，跑到这个水平就停。

---

## 实验：哪些惊喜，哪些值得扣字眼

### 主实验：所有模型都涨，但涨幅有点意思

![图3：同 token 预算下 TACO 与 baseline 的 accuracy 对比](https://arxiv.org/html/2604.19572v1/x3.png)

*图3：六个模型在 TB 2.0 上同 token 预算（横轴）下的准确率对比。上面三个是大模型（Qwen3-Coder-480B / DeepSeek-V3.2 / MiniMax-M2.5），下面三个是小模型（Qwen3 8B/14B/32B）。橙色 TACO 在所有 budget 下都明显高于蓝色 baseline。*

把 TerminalBench 主表抄一下，挑核心几条对比：

| 模型 | TB 1.0 baseline | TB 1.0 +TACO | TB 2.0 baseline | TB 2.0 +TACO |
|------|---------------|--------------|---------------|--------------|
| Qwen3-Coder-480B | 37.50 | 38.50 (+1.00) | 23.90 | 25.86 (+1.96) |
| MiniMax-M2.5 | 42.30 | 45.25 (+2.95) | 42.80 | 44.16 (+1.36) |
| DeepSeek-V3.2 | 43.93 | 46.25 (+2.32) | 40.62 | 42.77 (+2.15) |
| Qwen3-32B-Instruct | 11.25 | 14.13 (+2.88) | 3.92 | 7.48 (+3.56) |
| Qwen3-14B-Instruct | 5.23 | 11.25 (+6.02) | 4.04 | 6.15 (+2.11) |
| Qwen3-8B-Instruct | 8.86 | 9.22 (+0.36) | 1.43 | 3.67 (+2.24) |

我看到 Qwen3-14B 在 TB 1.0 涨了 **6.02 个点**（5.23 → 11.25），第一反应是"等等，这是不是 baseline 选得有点惨？"——5.23% 的基线意味着模型几乎啥都做不出来，加任何辅助都容易看到大幅提升。但反过来想，**这说明 TACO 对小模型的边际效用更明显**：小模型本身上下文窗口小、推理力弱，被噪声淹得更彻底，把噪声清掉就能从"完全卡住"变成"能跑完几步"。

大模型这边的涨幅就温和很多——MiniMax-M2.5 涨 2.95、DeepSeek 涨 2.32、Qwen3-Coder-480B 只涨 1.00。这其实是合理的：480B 这种规模的模型对噪声本来就有更强的鲁棒性，但即使如此还能拿到 2 个点左右的稳定增益，工程上已经很值。

### Token 效率：大模型省 token，小模型多走几步

| 模型 | Acc baseline | Acc +TACO | Avg Step baseline | Avg Step +TACO | Token/Step baseline | Token/Step +TACO |
|------|------------|-----------|------------------|----------------|--------------------|------------------|
| Qwen3-Coder-480B | 23.3 | 25.8 | 45.7 | 47.0 | 21,718 | 19,965 (-8%) |
| DeepSeek-V3.2 | 40.6 | 42.7 | 29.5 | 30.6 | 35,038 | 30,939 (-12%) |
| MiniMax-M2.5 | 42.8 | 44.1 | 43.2 | 42.6 | 28,631 | 28,559 (-0.3%) |
| Qwen3-32B | 3.9 | 7.4 | 15.7 | 19.6 | 8,472 | 8,735 (+3%) |
| Qwen3-8B | 1.4 | 3.6 | 44.3 | 68.5 | 9,579 | 9,583 (持平) |

这张表挺有意思。DeepSeek-V3.2 每步省了 4099 token (12%)，这是真实的效率提升。但 Qwen3-8B 平均步数从 44.3 → **68.5**，每步 token 没省，**总 token 反而涨了 50%+**——只不过涨的是因为它现在能跑得更远了，原来跑 44 步就 fail 了，现在能跑到 68 步把任务做完。

作者很坦诚地承认了这点。这也就是为什么后面 6.2 节专门做了"同 token 预算下的精度比较"——直接比 token/step 对小模型不公平。

### Pass@k：不止单次更准，多次采样也更稳

![图4：TACO 在 pass@k 上的提升](https://arxiv.org/html/2604.19572v1/x4.png)

*图4：六个模型 pass@k（k=4 到 8）对比。橙色 Self-Evo (TACO) 在所有模型、所有 k 上都稳定高于蓝色 Baseline。Qwen3-Coder-480B 的 pass@8 从 ~45% 提到 ~46%，DeepSeek-V3.2 的 pass@8 从 ~67% 提到 ~69%。*

这个结果对工程的意义比单 run 提升还重要。Pass@k 提升说明 TACO 不仅让单次轨迹更准，还**扩大了 agent 的"潜在解空间"**——同样跑 8 次，TACO 能命中更多的解。这通常意味着压缩后的上下文给了模型更多探索不同策略的"认知带宽"，而不是被噪声困在同一类失败模式里。

### 收敛性：自演化什么时候能停？

![图5：自演化收敛性与性能稳定性](https://arxiv.org/html/2604.19572v1/x5.png)

*图5：(a) Top-30 规则的 Retention（连续轮次重合度）随 run 数变化，三个模型都在第 8 轮左右达到 90% 阈值。(b) 任务准确率的滑动标准差，收敛之后从 2.0+ 降到 1.0 以内，明显低于 baseline 的横线。*

这是论文里我觉得设计得最聪明的一个工程细节。**自演化系统最棘手的问题是不知道什么时候停**——如果一直跑下去，规则池可能在某个时刻已经稳定但你还在烧钱训练。Retention 这个指标用的就是"高频规则的重合度"作为收敛信号，避开了直接看任务 accuracy 抖动那种含噪声的方法。

下面那张图是验证：一旦 Retention 超过 90%，accuracy 的滑动标准差就明显下降，说明这个收敛信号是真实可靠的。这种"用 metadata 而非 metric 判断收敛"的思路，做 RL 训练管理的同学应该挺熟悉。

### 消融：两个进化模块缺一不可

| 方法 | Acc | ΔAcc | ΔTok |
|------|-----|------|------|
| Baseline | 40.6% | 0.0% | 0.0% |
| TACO w/o Global Rule Pool Evolution | 40.4% | -0.2% | 18.1% |
| TACO w/o Intra-Task Rule Set Evolution | 38.9% | -1.7% | 30.7% |
| TACO (Full) | 42.7% | +2.1% | 12.2% |

这个消融实验讲了三件事：

1. **只有任务内进化没有全局池**：accuracy 持平 baseline，token 省了 18%——意味着每个任务自己摸索一套规则的效率不差，但**没法跨任务积累**。
2. **只有全局池没有任务内进化**：accuracy 反而掉 1.7 个点，但 token 省 30%——固定的高质量规则压得太狠，丢了关键信息。
3. **两个一起上**：accuracy +2.1%，token -12%——精度和效率双赢。

最反直觉的是中间那条："w/o Intra-Task" 是 token 省得最多的，但精度反而最差。这正是论文 Figure 1 (b) 那个 High-Quality Rule 现象的另一种表达——**静态高质量规则会把信息压缩过头**，必须有任务内的在线调整作为"安全阀"。

### 跨基准泛化：从 TerminalBench 到 SWE-Bench

| 基准 | Baseline Acc | TACO Acc | Baseline Tokens (M) | TACO Tokens (M) |
|------|------------|----------|---------------------|-----------------|
| SWE-Bench Lite | 56.30 | 57.12 (+0.82) | 307.61 | 270.53 (-12%) |
| CompileBench | 75.00 | 75.00 (+0) | 14.55 | 11.41 (-22%) |
| DevEval | 38.10 | 39.74 (+1.64) | 36.72 | 26.82 (-27%) |
| CRUST-Bench | 47.00 | 48.05 (+1.05) | 163.53 | 134.97 (-17%) |
| TB 1.0 | 42.30 | 45.25 (+2.95) | 29.74 | 23.43 (-21%) |
| TB 2.0 | 42.80 | 44.16 (+1.36) | 113.74 | 110.63 (-3%) |

CompileBench 上 accuracy 持平但 token 省 22%——意思是任务都能做完，只是更省钱。DevEval 涨 1.64 个点的同时 token 省 27%。这种跨任务的稳定泛化，确实说明规则池里沉淀下来的"通用压缩知识"是有迁移价值的。

### 超参选择：top-k=30 和 batch=4 的工程取舍

![图6：超参选择消融](https://arxiv.org/html/2604.19572v1/x6.png)

*图6：左边 Top-k 选规则的影响，k 从 10 到 50，accuracy 在 k=30 附近达到最高点（25.9%），k>30 后 self-evo 的 token cost 还在涨但精度反而下滑；右边 batch size N，N=2 的精度最高（27%）但 runtime 是 N=4 的 1.6 倍，N=4 是甜蜜点。*

这两张图揭示了一个冷酷的现实：**TACO 一轮完整自演化要 4 天**（在 N=4 的设置下，跑完 TB 2.0 全部任务）。你想让规则收敛得更稳（用更小的 batch size），就要付出更多时间。这是任何"在线 learning" 系统都绕不开的精度-时间 tradeoff。

---

## 案例：从"截断 100 字符"到"理解 apt-get 在干嘛"

附录 A 里的三个案例研究是这篇论文里最有说服力的部分。挑两个最典型的讲。

### 案例一：apt-get install 的极致压缩

任务是用 R 跑统计采样，第一步要 `apt-get install -y r-base`，安装 200+ 依赖包。**原始输出 10071 字符**：

```
Unpacking libc6-dev:amd64 (2.39-0ubuntu8.7) ...
Unpacking gcc-13-base:amd64 (13.3.0-6ubuntu2~24.04.1) ...
Unpacking libisl23:amd64 (0.26-3build1.1) ...
[... 150+ Unpacking/Setting up lines ...]
Setting up r-base (4.3.3-2build2) ...
```

TACO 的 `apt_install_unpacked_packages` 规则把它压成 **73 字符**（压缩比 0.007）：

```
[WAITING] apt-get install -y r-base
Current status: Setting up x11-utils
```

99.3% 的减少。我看到这个数的时候有点意外，但仔细想想完全合理——agent 在等待 apt 装包的时候，需要的就是"还在跑"和"跑到哪了"两个信号，谁会去关心 libisl23 那个版本号是 0.26-3build1.1 还是别的。

这条规则不是种子规则，**是 TACO 在跑这个任务的时候自动发现的**。当 agent 第一次撞到几千行 Unpacking 没规则匹配，TACO 让 LLM 看着这个输出生成了一条新规则——这就是 Intra-Task Rule Evolution 的实际效果。

### 案例二：objdump 反汇编的语义过滤

这个例子更精彩。任务是 reverse-engineering 一个二进制文件找 flag，agent 用 `objdump -d` 反汇编。原始输出 5619 字符的汇编代码：

```
4011e5: 31 f6                    xor %esi,%esi
4011e7: bf 11 00 00 00          mov $0x11,%edi
4011ec: 31 c0                    xor %eax,%eax
4011ee: e8 7d fe ff ff          call 401070 
[... 90+ more instruction lines ...]
```

TACO 的 `objdump_disassembly_rule` 把它压到 821 字符（压缩比 0.146），**保留所有 call 指令**（揭示 API 调用比如 `signal`/`ptrace`），**保留 section header 和 symbol label**，但去掉那些纯算术运算的 mov/xor/test 行。

整个 101-episode 的轨迹，这一条规则就触发了 18 次，省了 29464 字符——占总压缩量的 54%。**一条 reactively 生成的规则覆盖了一整类高频场景**，这正是规则池能 amortize 成本的关键。

更妙的是，这条规则不是 hardcode 在种子里的（哪个工程师会预先写一条 objdump 压缩规则？），是 agent 实际撞到 objdump 输出之后才即兴生成的。这是 LLM 生成规则的**杠杆效应**——一次 LLM 调用产生的规则，可能在后续 100 次调用中被反复使用，分摊下来成本极低。

---

## 我的判断：好东西，但也别神化

### 三个真实价值

**第一，把 prompt 工程做成了在线 loop。** 业界已经有一堆"agent 自演化"的工作（Memento-Skills、SAGE、Symbolic Learning），但 TACO 是第一个把这套框架认真落到"上下文压缩"这个具体场景的。它的价值不是某条具体规则，而是证明了**"LLM 生成结构化规则 → 规则在执行中被验证 → 高质量规则跨任务沉淀"**这个 loop 在工程上是可跑通的。

**第二，规则池 + 隐式反馈 = 可持续维护的压缩机制。** 这套架构最让我觉得"对，就该这么做"的地方是隐式反馈机制。你不用明确告诉系统"这条规则不好"，agent 行为模式（重跑命令、要求完整输出）就是天然的 reward signal。这相当于把 RL 里的 outcome supervision 翻译成了 prompt 世界的语言。

**第三，跨基准泛化是实打实的。** Section 6.6 那张表，TACO 在 5 个不同基准上都有提升或者持平，token 都有节省。这种迁移不是论文里的常见叙事话术——它是建立在"全局规则池里的高频规则确实捕获了通用模式（git/pip/apt/compiler/openssl）"这个具体事实上的。

### 几个需要画问号的地方

**问号一：1-4 个点的提升，是真"上下文质量"提升还是"prompt 长度"效应？**

这是我最大的质疑。终端 Agent 在 Qwen3-Coder-480B 这种长上下文模型上仍然只涨 1 个点（TB 1.0），到 Qwen3-8B 涨 0.36 个点。这些幅度跟单纯把上下文从 60k 截到 30k 比起来，区别有多大？论文做了 fixed-budget 比较（Figure 3），证明同 token 下 TACO 更好，这个比较是有效的，但**没回答"如果 baseline 也用一个简单的 truncate-tail 策略，差距还会这么大吗"**。

**问号二：4 天一轮的训练成本对工业落地是个挑战。**

文章里说"在 N=4 的设置下一轮 self-evolution 大约 4 天"，意思是你想用这套规则池，要么等公开发布的预训练规则池（论文 GitHub 上应该会放），要么自己跑 4 天。这对 startup 或者要在新 codebase 上快速落地的团队来说是个不小的门槛。

**问号三：规则的可解释性是优势也是天花板。**

把压缩规则做成 regex JSON 的好处是**完全可审计**——你能看到每一条压缩在做什么，错了能 debug。但这也意味着 TACO 的能力上界被 regex 的表达能力限制住了。比如多行上下文的语义关联（"这个 error 跟 50 行前那个 warning 有关"）regex 是搞不定的，必须靠 Critical-bypass 直接放原文，这种情况下 TACO 退化成了"什么都不做"。

**问号四：Critical 检测的鲁棒性论文没充分讨论。**

整个系统的安全网是 Critical-bypass——一旦检测到 error/traceback 就放原文。但 Critical 检测本身是怎么做的？论文里只提到"明显的 error 或 failure 信号"，没有给出明确的判断逻辑。如果一个错误日志没有标准的 ERROR 关键词（比如某些 Java 应用用 `[FAIL]` 标记），TACO 会不会把关键错误也压掉？这是工程落地必须做大量回归测试的地方。

### 跟同期工作的位置

TACO 跟 SWE-Pruner（Wang et al., 2026）的对比是有意思的——SWE-Pruner 是 finetune 一个小模型做 pruning，针对 SWE-Bench 优化；TACO 是 training-free 的规则池，跨多个终端基准都能用。**两者其实不是替代关系，是互补的**：SWE-Pruner 在 SWE-Bench 这个特定 distribution 上能更细粒度地学习什么该剪；TACO 更适合那些没有大规模标注数据、需要快速适配新环境的场景。

跟 Memento-Skills 这类"技能进化"的工作比，TACO 的位置更窄但更扎实。Memento-Skills 那种"让 agent 沉淀通用技能"的目标更宏大但也更模糊；TACO 把范围限定在"上下文压缩规则"这一件事上，反而做得更深、更可验证。

---

## 工程启发：能不能在你的 agent 系统里复刻？

如果你也在做长程 agent，特别是终端类、文件 IO 类、网络爬虫类这种**输出量级不可控**的场景，TACO 这套框架是非常值得抄的。我个人会这么落地：

1. **从静态规则起步**：先把项目里高频出现的命令（比如你的 build pipeline 用的 mvn/gradle/npm/yarn 等）写一组种子规则，这是 0 成本的优化。
2. **加 Critical-bypass**：对 error/exception/stacktrace 这些关键信号做白名单，永远放行。这一步能避免上下文压缩的"失之毫厘谬以千里"。
3. **逐步引入 LLM-based 规则生成**：当遇到某个命令的输出没有规则覆盖时，调一次 LLM 生成规则，存到本地池里。**每次 LLM 调用是有成本的，但生成的规则可以复用 N 次，分摊下来很便宜。**
4. **隐式反馈做规则淘汰**：观察你的 agent 是否重复执行同一条命令、是否要求看完整输出，这些都是天然的"该规则太激进"信号。

最关键的一点：**别想着一开始就做出"完美的规则"**，TACO 的核心价值是这个 loop 本身能让规则池随着使用越来越好。这跟 RLHF 在过去几年的演化逻辑其实很像——一开始大家都想做完美的 reward model，后来发现"持续收集偏好数据 + 持续微调"的工程 pipeline 才是真正的 moat。

终端 Agent 的上下文压缩，可能正在迎来它自己的"工程 pipeline 时代"。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
