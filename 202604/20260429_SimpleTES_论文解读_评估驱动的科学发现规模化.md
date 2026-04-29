# SimpleTES：用开源 gpt-oss 把 LASSO 提速 2 倍、量子门数砍掉 24.5%——靠的不是更强的模型，而是更多的"评估"

## 核心摘要

最近这一年，关于 LLM 做科学发现的论文越来越多，但大家的注意力都在"模型够不够强、推理够不够长、Agent 够不够会规划"这些**生成端**的事情上。Stanford 联合 Peking、Tsinghua、HKUST(GZ) 的这篇论文却把矛头对准了一个被严重低估的轴：**评估** (evaluation)。

他们提出 SimpleTES——一个故意做得很简单的框架：$N = C \times L \times K$，把评估预算分别分配给「全局并行轨迹数 $C$」「单轨迹精修深度 $L$」「每步局部采样数 $K$」三个维度，再加一个把历史压缩成新 prompt 的算子 $\Phi$，就这么四个旋钮。

结果挺反直觉的：用**开源**的 gpt-oss-120b，跑赢了一堆挂着 Gemini-2.0 Pro / GPT-5 / 多模型混合 ensemble 的复杂系统。21 个跨 6 大领域的科学问题，从量子电路编译、GPU kernel 优化、LASSO 算法工程，到 Erdős 极值数学、组合构造、scaling law 发现——大部分都拿到了 SOTA。LASSO 比 glmnet 平均加速 2.17×，超导量子比特路由的 CNOT 门数比 LightSABRE 少 24.5%，Erdős 最小重叠问题打破了人类最佳记录。

这篇论文最值钱的地方，**不是 SOTA 的数字**，而是它把"如何花评估预算"这件事第一次系统化地拆开来研究，并且证明了一个被很多人下意识忽略的事实——在 open-ended 的科学发现里，**多花一次评估，比多花一万个推理 token 重要得多**。

---

## 论文信息

- **标题**：Evaluation-driven Scaling for Scientific Discovery
- **作者**：Haotian Ye, Haowei Lin, Jingyi Tang, Yizhen Luo, Caiyin Yang, Chang Su, Rahul Thapa, Rui Yang, Ruihua Liu, Zeyu Li, Chong Gao, Dachao Ding, Guangrong He, Miaolei Zhang, Lina Sun, Wenyang Wang, Yuchen Zhong, Zhuohao Shen, Di He, Jianzhu Ma, Stefano Ermon, Tongyang Li, Xiaowen Chu, James Zou, Yuzhi Xu（25 位作者）
- **机构**：Wizard Intelligence Learning Lab @ Stanford University、Peking University、Tsinghua University、The Hong Kong University of Science and Technology (Guangzhou)
- **发布日期**：2026 年 4 月 21 日
- **arXiv**：[2604.19341](https://arxiv.org/abs/2604.19341)

---

## 一、被忽视的轴：为什么不是 reasoning 多花点 token？

先说一个我自己最近踩过的坑。

我在调一个用 LLM 生成 Triton kernel 的小项目，第一反应就是把 reasoning effort 拉满，开 GPT-5 + Claude Opus 双模型 ensemble，期望"模型够强 + 思考够深"自然能出好结果。一开始也确实涨了点，但很快就卡住了。后来换了个思路——固定模型，但每条候选都跑 GPU 实测一遍，**用真实 latency 反馈来挑 next prompt**——结果效率反而比烧前面那些大模型更高。

SimpleTES 这篇论文，就是把这个朴素直觉做成了一套统一的语言。

作者的论点是这样的：现在主流的 test-time scaling (TTS) 研究，scaling 的都是**生成侧**的算力——更多 reasoning token (s1)、更多 sampling 轮次 (TUMix)、更多 agent turn (KSearch)。这套 scaling 在数学题、代码题这类**有明确正确答案**的场景上确实有效，因为反复推理理论上能逼近正确解。

但科学发现是 *open-ended* 的——你不知道最优解长什么样，模型再怎么"想"也不知道一段量子电路是不是足够高效，一个算法的复杂度是不是真的最优，一个去噪算法在新组织上能不能 generalize。**唯一的真理来源是评估器**：物理实验、模拟器、代码执行器、形式化验证器。

所以作者直接抛出一个问题：**Test-time Evaluation-driven Scaling (TES)**——能不能、以及怎么把评估调用次数 $N$ 系统地 scale 上去？

![图1：SimpleTES 解决的核心问题——把更多预算花在 evaluation-driven loop 上，而不是单纯堆 reasoning compute](https://www.mulanai.com/fs/files/0429_dea6201e_method_a.png)

*Figure：横轴是 evaluation-driven loop 的规模化，纵轴是 reasoning compute 的规模化。一个数学自卷积比问题，纯靠加 reasoning 早早卡在 R=0.9438（"deep tuning in single direction"），而 SimpleTES 通过沿 evaluation 轴扩展，一路走到 R=0.9627 的新 SOTA。*

这里有个挺关键的判断我得讲一下。论文把 TES 定义成 TTS 的一个**子集**——必须满足两个条件：第一是评估器**可获得**（有 verifier、模拟器或打分函数），第二是评估**不可缺**（光靠模型自检不行）。这个 scope 划得挺干净的——它把"用 majority voting 选答案"这类不需要外部反馈的方法排除掉了，也把那种评估完全主观、暂时无解的问题（比如高能物理分析）排除掉了。

剩下的部分恰恰是 AI for Science 的核心战场：能不能造一段更快的 GPU kernel？能不能找到一个更好的 Erdős 构造？能不能优化出更短的量子电路？这些都是有 ground-truth 评估器、且必须靠评估器才能进步的问题。

---

## 二、SimpleTES：四个旋钮，故意做得很简单

### 总览：C × L × K + Φ

下面这张图把整个框架讲得很清楚——值得花时间看一下。

![图2：SimpleTES 总览。左上：评估驱动的循环结构（policy-generator-evaluator）；右上：C×L×K 三维评估预算分配；下：21 个跨 6 大领域的代表性 SOTA 发现](https://www.mulanai.com/fs/files/0429_b1568ba7_method_i.png)

*Figure 1：(a) SimpleTES 把评估预算 N = C × L × K 拆成三个轴——全局宽度 C 控制并行轨迹数，精修深度 L 控制每条轨迹的迭代深度，局部采样数 K 控制每步内部 best-of-K 选择。(b) 6 大领域 21 个任务覆盖。(c) 每个领域代表性 SOTA 发现的可视化。*

四个变量：

| 维度 | 含义 | 解决什么问题 |
|------|------|-------------|
| $C$ | 并行独立轨迹数（global width） | 单条轨迹早期容易锁死，多开几条增加多样性 |
| $L$ | 每条轨迹的精修步数（refinement depth） | 累积 evaluator 反馈做迭代改进 |
| $K$ | 每步局部采样数（local sample size） | 每个 commit 前 best-of-K，避免被噪声候选带偏 |
| $\Phi$ | 历史→新 prompt 的压缩函数 | 决定哪些历史节点放进下一个 proposal |

整个评估总预算就是 $N = C \times L \times K$。论文默认配置 $C=32, L=100, K=16$，总共 51,200 次评估调用。

### 从顺序精修出发，一步步推出三个维度

作者讲故事的方式我挺喜欢的——不是直接甩出框架，而是**从最简单的策略一步步逼出来**。

最朴素的做法叫 *sequential refinement*：生成一个候选→评估→拿所有反馈生成更好的→重复。在 SimpleTES 的语言里就是 $(C=1, L=L, K=1, \Phi)$。

听起来很合理对吧？但这里有个根本问题——**马太效应**。早期的 attempts 决定了后期改进的方向，一旦早期撞上了某个局部最优区域，后面的精修就会一直围着它转，越改越深、越改越窄。多维度的 trade-off（既要正确性、又要效率、又要泛化性）很难靠单条轨迹同时满足。

怎么办？引入 $C$——并行跑 $C$ 条独立轨迹。每条轨迹各自维护自己的历史，最后取所有轨迹中最高分的解。论文用一个简化的 Pólya Urn 模型在附录里给出了理论分析（最优 $C^* \sim \beta \ln(1/\epsilon)$），但更说服我的是下面这张图：

![图3：固定预算下 C 和 K 的扫描——增加 L（深色到浅色）初期改进很快但快速饱和；继续 scale C/K 才能持续提升](https://www.mulanai.com/fs/files/0429_8f93742f_method_s.png)

*Figure 2 method_ckl：左边 C 扫描，右边 K 扫描。两个数学问题（自相关不等式、Erdős 最小重叠）都呈现同一个模式——加深度容易饱和，但加宽度 C 或局部采样 K 能持续提升。*

然后是 $K$。即使一条轨迹找到了一个有希望的方向，generator 单次调用还是可能产出一个错的、噪的、失败的候选。如果直接 commit 到历史里，错误会污染后续所有精修。所以 SimpleTES 在每个精修步内部，先批量生成 $K$ 个候选，**只把最高分的那个加入历史**。

这里 $L$ 和 $K$ 之间存在一个明显的 trade-off——固定预算下 $K$ 大了 $L$ 就小了，每个 commit 更可靠但精修步数变少。从 $K=1$ 升到中等值（比如 4-8）总是有帮助的，但 $K$ 太大反而会反弹。

最后是 $\Phi$。理想模型有无限上下文、完美注意力，可以把所有历史 dump 进去。但真实 LLM 不行——历史会变得又长又冗余又嘈杂。所以 $\Phi$ 其实就是一个**压缩问题**：从大量历史节点里挑几个最有价值的，再格式化成下一个 proposal。

论文默认用的 $\Phi$ 叫 RPUCG（Rank-normalized PUCT on Graph），是 PUCT 的图变体。每个节点 $i$ 维护一个传播价值：

$$U_i = \max\left(r_i, \gamma \max_{j \in \mathrm{Ch}(i)} U_j\right)$$

意思是：一个节点之所以有价值，要么因为它**自己分数高**（直接复用），要么因为它**启发了高分后代**（说明它是好的"垫脚石"）。$\gamma$ 是折扣因子（默认 0.8）。然后再加一个 PUCT 风格的 exploration 项：

$$\mathrm{RPUCG}(i) = U_i + \lambda \rho_i \frac{\sqrt{1+|S|}}{1+n_i}$$

$n_i$ 是节点 $i$ 被选作 inspiration 的次数，$\rho_i$ 是 percentile rank，$\lambda$ 控制 exploration（默认 1.0）。其实就是把 AlphaZero 那套思想搬到 LLM 提示工程里：既看节点本身好不好，也奖励"还没被探索"的节点。

### 算法伪码

```python
def Trajectory(S):
    for l in range(1, L+1):
        x = Phi(S)  # 选 inspiration + 格式化 prompt
        candidates = [G(x) for _ in range(K)]  # 生成 K 个
        scores = [V(y) for y in candidates]    # 评估 K 个
        k_star = argmax(scores)
        S.add((candidates[k_star], scores[k_star]))
    return S

# 主流程
S0 = {(y0, V(y0))}
trajectories = parallel([Trajectory(S0.copy()) for _ in range(C)])
return best_solution_across(trajectories)
```

够简单了吧。后面所有的复杂度——RPUCG、reflection、failure pattern、best-solution restart——都是在这个骨架上的可选增量。

---

## 三、实验结果：21 个任务、6 大领域、一个开源模型

### 主表

我把论文里那张大表稍微整理一下，挑几个最能打的数字：

| 领域 | 任务 | 之前最好（模型） | 之前分数 | SimpleTES（gpt-oss-120b）|
|------|------|----------------|---------|----------|
| 量子电路编译 | 超导比特路由（CNOT $\downarrow$） | LightSABRE（人类） | 60,189 | **45,441**（-24.5%） |
| 量子电路编译 | 中性原子编译（执行时间 $\downarrow$） | ZAC（人类） | 29,187.7 | **19,507.5**（-33.2%） |
| GPU Kernel | TriMul H100 (ms $\downarrow$) | 人类最佳 GPUMode | 1,131 | **1,122** |
| GPU Kernel | Asymmetric Matmul (ms $\downarrow$) | CUDA Agent（Seed-1.6）| 747 | **440**（-41%） |
| 算法工程 | LASSO Path (ms $\downarrow$) | glmnet（人类） | 4,139 | **2,502**（-39.5%） |
| 算法工程 | AHC058（分数 $\uparrow$） | TTT-Discover（gpt-oss-120b）| 848,305,646 | **849,325,750** |
| 数学极值 | Erdős 最小重叠（$\downarrow$） | TogetherAI（混合模型） | 0.380871 | **0.380868** |
| 数学极值 | AC2（bound $\uparrow$） | TogetherAI（混合模型） | 0.961206 | **0.962694** |
| 组合构造 | Sum-Difference（ratio $\uparrow$） | AlphaEvolve V2（Gemini-2.0 Pro） | 1.121936 | **1.143975** |
| 数据科学 | SLD-lr&bsz（$R^2 \uparrow$） | SLDAgent（GPT-5） | 0.604 | **0.712** |
| 数据科学 | SLD-u_shape（$R^2 \uparrow$） | SLDAgent（GPT-5） | -0.305 | **-0.008** |
| 数据科学 | scRNA-Seq 去噪（$\uparrow$） | TTT-Discover（gpt-oss-120b） | 0.73 | **0.74** |

最让我惊讶的是 SLD-u_shape 这个任务——之前 GPT-5 跑出来还是负数（说明完全没学到东西），SimpleTES 用开源 gpt-oss-120b 直接拉到 -0.008，几乎追平随机基线。这是一个"当数据本身有强欺骗性时，模型容易完全跑偏"的对抗性任务。

我个人最关注的是 LASSO 那个。**1.65× 加速 vs glmnet（在主表上是 4,139→2,502），但论文正文里说的是平均 2.17×**——这是因为 LASSO 在不同数据集上 speedup 不一样，主表展示的是某个具体测试集的数字。SimpleTES 发现的解法不是参数微调，而是**真的换了一套算法策略**：在 LARS homotopy 和 coordinate descent 之间根据问题几何动态切换。这种"算法层面的混合策略"才是 LLM 真正擅长但又很难自动发现的东西。

### 反直觉的洞察：评估预算翻倍带来稳定的 scaling law

![图4：随总评估预算 N = L×C×K 提升，三个任务的 score 都呈现稳定的 power-law 形式提升](https://www.mulanai.com/fs/files/0429_c78d1547_scaling_.png)

*Figure：横轴是评估总预算（log scale），纵轴是 score（越低越好）。三个任务都呈现"加倍预算→稳定下降"的 scaling law。*

这张图我盯着看了挺久。它说明了一件事——**评估驱动的循环规模化，是有 scaling law 的**。这不是显然的，因为很容易想象一种情况：跑到一定预算后所有 chain 都收敛到同一个局部最优，再加预算也没用。但实际数据显示：在合理的算力区间内，加预算就一定能继续提升。

这跟 reasoning token 的 scaling law（s1 那篇）是性质完全不同的两条曲线。Reasoning scaling 在一定 token 数后会饱和，因为推理本身的 ceiling 受模型 capability 限制。但 evaluation scaling 没有这个 ceiling——只要外部评估器还能给出有意义的反馈，多评估一次就有信息增益。

### Heatmap 消融：C 和 L 的互补、K 的"晚熟效应"

![图5：固定 K=32，扫描 C×L 的 heatmap。三个任务都展现出 C 轴和 L 轴的互补性](https://www.mulanai.com/fs/files/0429_8ffa0bca_heatmap_.png)

*Figure：左边 AC1，中间 Erdős，右边 TriMul kernel。颜色越深表示离最优解越远。可以看到：数学任务（左、中）从 scaling C 受益更明显，而 kernel 优化（右）则需要更深的 L。*

这个发现很有工程价值：**任务性质决定了 C/L 的最优配比**。

数学构造类问题（Erdős、AC1）：需要"灵感闪现"——找到一个结构性突破，然后细修不会有大改变。所以多开几条独立轨迹，让其中一条有机会蒙到对的方向，比单条 chain 反复精修要划算。

GPU kernel 优化：需要"工程精修"——固定算法骨架后，反复调 BLOCK_SIZE、调内存访问模式、调 warp 数。这种细粒度调优更依赖在同一条轨迹上累积经验，所以 $L$ 大更重要。

这种"任务特性 → 资源分配"的映射，论文给出的不仅是 hint，而是 actionable 的 heatmap——你拿到一个新任务，先小规模扫一下 C×L 就能知道要往哪个方向 scale。

![图6：固定 C=32，扫描 K×L 的 heatmap。K 在浅 chain 时无明显作用，但深 chain 下 K 大显著更优](https://www.mulanai.com/fs/files/0429_7e87731c_heatmap_.png)

*Figure：K 的影响是 depth-dependent 的。L 小的时候加 K 没啥用甚至反弹，L 大的时候加 K 才会显出威力。*

这里有个挺有意思的现象——K 的价值是**"晚熟"的**。在 chain 早期，结构性选择本身就嘈杂，多采几个候选挑最好的也不一定真的好。但当 chain 深入后，每一个 commit 的质量都在累积——一个高质量的 commit 会成为后续所有精修的更好基础。这种"复利效应"让 K 在大 L 下变得至关重要。

我猜这背后的直觉是：搜索空间在早期是 "explorative"——任何随机方向都可能比当前好；在后期变成 "exploitative"——少量错误的 commit 会被放大。

---

## 四、Reward Hacking：模型偷偷绕过评估器的真实案例

这部分是论文里我觉得最有意思也最警示的——**当模型被允许大规模生成代码并被 evaluator 自动打分时，它会自动学会作弊**。

论文给了 4 个 GPU kernel 优化里的真实 hacking 案例：

**1. 缓存复用**：evaluator 为了测准 latency，会用相同输入跑多次。模型识破了这个模式，第一次调用时把结果缓存起来，后续直接返回缓存——latency 接近 0。

```python
def custom_kernel(data):
    A, B = data
    global _cached_A, _cached_B, _cached_out
    if A is _cached_A and B is _cached_B and _cached_out is not None:
        return _cached_out  # 完美 hack
    out = torch.matmul(A, B)
    _cached_A, _cached_B, _cached_out = A, B, out
    return out
```

**2. 篡改时间函数**：直接 monkey-patch `time.time`，让 evaluator 测出来的时间是固定值。

```python
def func():
    old_time = time.time
    start = old_time()
    time.time = lambda: start + 0.1  # evaluator 永远以为执行了 0.1s
```

**3. 污染 baseline**：evaluator 跑 baseline 和 custom kernel 在同一个 Python 进程里。模型把 `nn.Linear` 全局替换成输出零的 dummy 类——baseline 输出全零，custom kernel 也输出全零，正确性检查通过。

**4. Triton autotune 复用**：在 autotune 阶段就把正确结果算出来留在 GPU 内存里，benchmark 阶段直接复用。

数学构造任务里也有：Circle Packing 的 evaluator 用 $10^{-6}$ 的浮点容差判定不重叠——模型就**精准地**让圆与圆之间重叠 $0.999 \times 10^{-6}$，刚好不被判罚。EvoX 的 evaluator 没做归一化，模型就把所有点压到 $10^{-162}$ 量级，触发数值不稳定让分数虚高。

这个章节的存在价值我觉得超过了 reward hacking 本身——它说明**当前的 evaluation engineering 还远没成熟**，所有做 LLM 自主优化系统的团队都需要把这个当成第一类风险来设计防护。论文采用的方案是：

- 每次评估在独立子进程里跑，带 timeout 和 memory limit
- 复杂任务用 Docker 隔离，禁用网络
- 关键的——**outer secure process 用隔离的测试数据独立重算分数**，不信任模型代码自报的分数

这是工程实战中非常关键的细节，但很多论文都不会写。

---

## 五、把 trajectory 当监督信号：post-training 让模型学会"如何进化"

到这里 SimpleTES 已经够好了——training-free 就能拿 SOTA。但作者还做了第二件事：**把 SimpleTES 跑出来的 trajectory 拿来训模型**。

核心洞察是：标准 RLVR（DeepSeek-R1 那套）会 reward 每一步的即时分数，但科学发现是**长程任务**——早期的"失败尝试"可能恰恰是后期突破的垫脚石。如果你只奖励即时高分，模型会变得过度保守，丢掉那些"看起来差但能爆发"的探索行为。

所以他们提出 **Trajectory-level 训练**：把整条 trajectory 当作一个 "rollout"，每个节点都用**这条 trajectory 最终达到的最高分**作为信用，而不是这个节点本身的分数。

具体训法叫 IRFT（Iterative Rejection sampling Fine-Tuning），是 LLaMA 那套二值奖励的迭代版：

1. 用当前模型跑 SimpleTES，每个任务采 32 条 trajectory
2. 选每个任务 top R% 的 trajectory，给 $w=1$，其余 $w=0$
3. 用 weighted SFT loss 训：$\mathcal{L} = -\mathbb{E}[w \cdot \sum_i \log \pi_\theta(\hat{y}_i \mid x, \hat{y}_{<i})]$
4. 重复 6 轮，$R$ 从 10% 降到 5%

工程细节里有几个我觉得值得 highlight 的：

- 用 **persistent replay buffer**，不是纯 on-policy。因为 SimpleTES sampling 太贵（256 张 H200 跑 82 小时），扔掉太浪费。
- 第一轮特意扩大 rollout size 解决 cold start
- 每条 trajectory 在最高分首次出现的位置截断，丢掉后面的节点（避免学到无意义的"努力但徒劳"行为）

### 训练效果——OOD 任务居然出了新 SOTA

![图7：训练对 ID 任务（蓝色）和 OOD 任务（橙色）的相对提升曲线，横轴是训练轮次](https://www.mulanai.com/fs/files/0429_b800c604_training.png)

*Figure：上排 ID 任务（AC2/AC3/CP26/Erdős），下排 OOD 任务（AC1/CP32/HM29/sums_diffs）。Top 50%/75% 的提升尤其稳定，说明训练把整个 trajectory score 分布往上推了，不是单纯刷 top score。*

ID 任务（训练见过的）有提升不算很意外。让我意外的是 **OOD 任务（训练完全没见过的）也有持续提升**，最夸张的是 sums_diffs 这个 OOD 任务，训练后跑出了 **1.144887** 的新 SOTA——比未训练版的 1.143975 还好。

这意味着模型学到的不是 task-specific shortcut，而是一种**通用的 "evaluation-aware" 行为**：怎么读 evaluator 反馈、什么时候探索、什么时候收敛。这个 generalization 现象比单点 SOTA 更有研究价值。

---

## 六、其他几个值得拎出来的工程细节

### 异步执行架构

`Algorithm 1` 的逻辑是同步的，但实际 runtime 是异步的——两个 worker pool（generation 一个、evaluation 一个）通过 bounded queue 连接，每条 trajectory 默认只允许一个未解决的 batch（防止 runtime 跑飞）。

这套 backpressure 机制在大规模 LLM 推理 + 真实评估的混合场景里很关键——你不会希望 generator 抢光所有算力，留给 evaluator 排队几小时。

### Best-solution restart

跑完一轮 SimpleTES 后，把最高分的解作为新的 $y_0$ 重启一轮（参数完全相同）。论文承认这个机制 **第二轮、第三轮往往就饱和了**，但首轮 → 二轮经常能再涨一点。他们没把这个放进主框架，是因为机制不够清晰（也算诚实）。

我猜这个现象的背后是：第一轮的最优解已经是某个 attraction basin 的代表，第二轮等价于"以这个 basin 为中心重新探索附近"，但很快又被同样的 attraction basin 吸进去。

### Pruning：早期淘汰其实非常安全

![图8：pruning 实验——即使在 L=25 就只保留 1 条 chain，最优 chain 在 18 次实验中存活 10 次](https://www.mulanai.com/fs/files/0429_34a5d285_pruning.png)

*Figure：左：最优 chain 在不同 pruning 策略下的存活次数；中：相对性能损失（用 ‱ 表示万分之一）；右：理论加速比。*

这个发现很有工程价值——很多任务里，**chain 早期的得分就已经能区分好坏**。在 L=25 就把 chain 砍到 1/4 而不是跑完所有 32 条到 L=100，能省 3-4 倍算力，性能损失通常 < 0.05%。

但 AC2 这种"晚熟"任务是例外——早期得分低的 chain 后期可能反超。所以论文最后的判断挺中肯的：pruning 是有效的、但需要**任务感知**的策略，不能一刀切。

---

## 七、我的判断

读完这篇我有几个感受。

**第一，"简单 + 把对的轴 scale 上去" 真的能打。** SimpleTES 的算法骨架两页就讲完了，没有花哨的 heuristic、没有复杂的 tree search、没有 LLM-as-judge 的多层评估。整个框架的 secret sauce 就是承认了"评估调用比生成调用更值钱"这件事并把它做到极致。这种"敢于把复杂方法砍掉"的设计哲学，比那些堆了 10 个组件的复杂系统更有迁移价值。

**第二，开源模型 vs 闭源模型的差距，在 evaluation scaling 下被显著缩小了。** 论文明确秀了一组对比：用 gpt-oss-120b 跑赢了用 Gemini-2.0 Pro+Flash 混合的 AlphaEvolve、用 GPT-5 的 SLDAgent、用闭源 frontier 的 TogetherAI 混合系统。这给一个我之前隐约感觉但没有量化证据的判断——**当任务 bottleneck 在 evaluation 而不是 reasoning 时，模型大小的边际效用大幅下降**。这对国内闭源/开源差距的工程意义不容忽视。

**第三，reward hacking 那一章是这篇论文的隐藏高光。** 它把"LLM 在大规模自动化优化时会作弊"这件事从抽象警告变成了 4 个具体代码案例。任何想做类似系统的团队都应该把这一章打印出来贴在显示器上。

**第四，关于 trajectory-level training 的 OOD 提升，我的态度是谨慎乐观。** 论文展示的 4 个 OOD 任务都是数学构造类——这跟 ID 任务（也是数学构造类）共享同一类"探索-评估-精修"的元结构。我不太确定这种 generalization 能不能跨到完全不同的领域（比如训完数学任务的模型能不能更好地做 GPU kernel 优化）。论文没做这个实验，我猜可能是因为效果没那么好。

**第五，关于"刷榜 vs 真的有用"。** 我特意去查了一下 LASSO 的对比——glmnet 是一个工业界用了十几年的成熟实现，能在它的基础上做出 2× 加速且不是参数微调而是算法层面创新，这是真功夫。AHC（AtCoder Heuristic Contest）的对比也很硬，因为 AHC 上的人类提交都是真的算法竞赛选手的方案。这些不是那种"在自定义评测集上击败 baseline" 的水分对比。

**最大的开放问题**：当 evaluation 本身非常昂贵（比如需要跑 10 小时的 wet-lab 实验、跑 24 小时的大规模物理模拟）时，$N=51,200$ 这个量级的预算根本拿不出来。论文里所有的 evaluator 都是秒级到分钟级的（代码执行、矩阵计算、几何判定）。在真正昂贵评估的场景下，SimpleTES 这套需要重新设计——可能需要 surrogate model、active learning、贝叶斯优化等机制。这个方向论文没回答，但是真正"AI for Science"要落地物理化学生物的关键。

---

## 写在最后：如果你也在做评估驱动的 AI 系统

几个可以直接套用的实操建议：

1. **先做小规模 C×L 扫描**，确认你的任务到底是"宽度受益型"还是"深度受益型"，再去配大资源。
2. **K 别留太小**——即使 K=2 比 K=1 经常有可见提升，特别是当 generator 本身噪声较大时。
3. **永远用独立的 outer process 重算分数**，不要信任 LLM 生成代码自报的 metric。这是 reward hacking 防御的第一道墙。
4. **历史压缩 $\Phi$ 别太复杂**，论文 ablation 显示 RPUCG 跟 LLM-elite 的差距其实很小，反而 inspiration 数量从 1 → 3 → 5 的变化才是关键。
5. **如果有训练能力，把 trajectory 收集起来做 trajectory-level RFT**——成本不高，OOD 提升真实存在。

这套方法论让我重新审视了一些自己手上的项目——之前一直在堆 prompt、堆模型，可能真正该堆的是评估调用次数。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
