---
title: "2.8T 参数全开源：Kimi K3 把 KDA 与 AttnRes 推到 3T 级 MoE"
date: 2026-07-28
arxiv: 2607.24653
authors: Kimi Team
subject: cs.CL / cs.LG
---

# 2.8T 参数全开源：Kimi K3 把 KDA 与 AttnRes 推到 3T 级 MoE

> arXiv 2607.24653 ｜ Moonshot AI ｜ 2026-07-27

## 核心摘要

如果让我用一句话讲清楚 Kimi K3 干了什么：**这是第一个把"3T 级 MoE + 1M 上下文 + 完整原生视觉"做出来还把权重全开源的模型，整体 scaling efficiency 比 K2 提升约 2.5×。**

架构上的关键招数有三个：把 Kimi Linear 里的 KDA 升级为带下界衰减 + 全秩门控的工业版（让长序列的 chunkwise 计算能纯跑 Tensor Core），把残差连接换成"每层用 learned query 去看前面所有层输出"的 Attention Residuals（8 个 block，2.8T 深度通信开销可控），以及把 MoE 扩到 896 routed / 16 active 的极端稀疏但用 Stable LatentMoE + Quantile Balancing 稳住训练。后训练是分三档 reasoning effort（low / high / max）× 三个领域（通用 / 智能体 / 编码）共 9 个专家，再做 Multi-Teacher On-Policy Distillation 收口。

效果上，K3 在 BrowseComp（91.2%）、AutomationBench（30.8%）、MCPMark-Verified（94.5%）、SWE-Marathon（42.0%）、ProgramBench（77.8%）上拿到所有对比模型里的第一，FrontierSWE（81.2%）和 OSWorld-Verified（84.8%）是第二；价格段在 KCB 2.0 / BrowseComp / GDPval / AA-Briefcase 四张图上几乎贴着 Pareto 前沿走。整体上"开放式完胜"其他开源（GLM-5.2），但仍然落后闭源最顶的 Claude Fable 5 和 GPT-5.6 Sol。

坦率讲，这篇 technical report 的工程量是惊人的——光是 sandbox 就造了 5120 万个、图像 150 万张，预训练阶段把 KDA 上下文并行、FlashKDA kernel、MoonEP 平衡 EP 等十几个底层组件重新打了一遍。**它的价值不在于某一个"颠覆性 idea"，而在于把所有这些"工业级细节"集成到 2.8T 规模并 release 出来——这才是开源社区真正稀缺的。**

---

## 论文信息

- **标题**：Kimi K3: Open Frontier Intelligence
- **作者**：Kimi Team（302+ 位作者，Moonshot AI / 月之暗面）
- **发布时间**：2026-07-27
- **arXiv**：2607.24653（[abs](https://arxiv.org/abs/2607.24653) ｜ [pdf](https://arxiv.org/pdf/2607.24653)）
- **权重**：已开源，https://huggingface.co/moonshotai/Kimi-K3
- **技术报告**，cs.CL / cs.LG

---

## 为什么这篇值得读：开源前沿被悄悄卡在 1T 附近

2026 上半年的一个明显趋势是：闭源阵营把 reasoning 模型的 test-time scaling 玩得越来越花——OpenAI 的 o 系列 / GPT-5.6 Sol、Anthropic 的 extended-thinking、DeepSeek-R1、Kimi 1.5/2.5 都在 RL 和工具使用上狠狠拉参数。

但 **pre-trained foundation 那一侧，开源阵营基本卡在 1T 上下**（GLM-5、Mimo V2.5 Pro、DeepSeek V4 这一档），只有 K2（1.04T）算是少有的"1T+"开源选手。这种差距带来的真正问题是：再花哨的 RL 后训练，其上限都受制于 base model 的容量；同样 RL 算法套在 1T 和 3T 上，得到的不是一个量级。

K3 直接把基础模型推到 **2.8T 总参 / 104B 激活**、原生视觉、1M context。这件事的意义不在于"我比你大"，而在于把开源的 base frontier 抬到了 3T 区间。后面所有的 reasoning、agentic、code RL 才有意义。

---

## 架构总览：三维信息流（token / channel / depth）

K3 的架构思想非常清晰：**用三个独立维度分别解决 sequence length、network depth、model width 的信息混合问题**——一个 block 内 3 层 KDA + 1 层 Gated MLA（token 维度），每个 KDA/MLA 后面跟一个 Stable LatentMoE（channel 维度），所有层用 Block Attention Residuals（depth 维度）连接。这三套机制各自负责一个轴，互不抢资源。

| 维度 | 机制 | 关键设计 |
|---|---|---|
| 序列长度（token） | Hybrid KDA-MLA（3 KDA + 1 Gated MLA 块内循环） | MLA 层用 NoPE，由 KDA 提供位置/近因信息 |
| 深度（layer） | Attention Residuals（AttnRes） | 每层一个伪 query，去 softmax 前面所有 block 输出 |
| 通道（channel） | Stable LatentMoE | 896 routed / 16 active，sparsity 56，latent 维度 3584 |

具体到结构：93 层、hidden 7168、96 头、1 个 dense layer、1 个 MTP 层。MoE 方面 routed expert 896 个、每 token 激活 16 个、共享 expert 2 个，每个 routed expert 内部 hidden 3072（K2 是 2048，提升 50%）。Latent MoE 维度是 3584（0.5× hidden），训练上下文 1M token（K2 是 128K，提升 8×）。

这些数字拼起来告诉你一件事：**K3 不是简单把 K2 拉大，而是从架构、MoE、训练上下文、视觉路径上几乎全部重做了一遍。**

---

## KDA 升级：把"下界衰减 + 全秩门控"做到能纯跑 Tensor Core

KDA（Kimi Delta Attention）来自 Kimi Linear 这条线，思路是 **delta-rule 递推 + 通道级遗忘门**，把 softmax attention 的增长 KV cache 替换成固定大小的 recurrent state $S \in \mathbb{R}^{d_k \times d_v}$：

$$
S_t = (I - \beta_t k_t k_t^\top) \cdot \text{Diag}(\alpha_t) S_{t-1} + \beta_t k_t v_t^\top,\quad \tilde o_t = S_t^\top q_t
$$

训练 / prefill 阶段用 chunkwise 并行（chunk 内 parallel、chunk 间 recurrent）。K2 的 Kimi Linear 做法是用 negative-Softplus 把 log-decay 映射到 $(-\infty, 0)$，导致 chunk 内需要分 secondary 16-token tile，**对角 tile 仍然要走 position-pair 计算**，这是主要 intra-chunk 瓶颈。

K3 换了个工程上很漂亮的做法：**用 scaled sigmoid 给 log-decay 一个下界** $g_{\min} = -5$：

$$
g_t^h = g_{\min} \cdot \sigma(e^{A_h} z_t^h) \in (g_{\min}, 0)^{d_k},\quad \alpha_t^h = \exp(g_t^h) \in (e^{-5}, 1)^{d_k}
$$

这一个 $\alpha > e^{-5} \approx 6.7 \times 10^{-3}$ 的下界带来的实际效果是：**16-token tile 内的累积 log-decay 落在 $(-80, 0)$，reciprocal rescaling 永远在 BF16 范围内。** 这意味着对角 tile 也不再需要 position-pair 路径，所有 causal tile 全都可以用 Tensor Core 跑 dense matmul。论文里的 fig:kda-lower-bound 把这件事讲得很清楚——A 面板的曲线对比、B 面板的"对角 vs 非对角 tile 全 Tensor Core"对比。

**坦白说，这是个很务实的小改。** 单独看 magnitude 上不是"颠覆"，但放到 1M 上下文 + 2.8T 规模下，"能不能把对角 tile 也纯跑 Tensor Core"是绝对决定 throughput 数字的关键——这直接关系到 K3 能不能在合理时间内把 1M context 训出来。

另一个改动是把 output gate 从低秩换成全秩（full-rank gated attention），让每个 token 都能 channel-wise 调制 attention 输出。同时 MLA 端也加了同样的全秩 gate。训练时 attention output 保留在 FP32 以规避 flash attention 的 rounding error，并把 doubling 的 on-chip footprint 通过重新设计 kernel（KV staging buffer）抵消。

---

## Attention Residuals：让深度也变成 attention

标准残差把前面所有信息压成一个 hidden state $h_l$——这其实是个"时间维度上的 RNN"瓶颈。AttnRes 直接把这个机制换掉：

**给每层一个 learned pseudo-query $w_l \in \mathbb{R}^d$**，key / value 用前面所有层（或 block）输出，attention kernel 是 softmax 形式但加 RMSNorm 防止大范数层压制权重：

$$
\alpha_{i \to l} = \frac{\exp(q_l^\top \text{RMSNorm}(k_i))}{\sum_j \exp(q_l^\top \text{RMSNorm}(k_j))},\quad h_l = \sum_i \alpha_{i \to l} v_i
$$

Full AttnRes 算术量是 $O(L^2 d)$，K3 选了 **Block AttnRes** 把 $L$ 层分成 8 个 block（12 层/block + 嵌入层 = 9 个 block），同一 block 内的输出先做 partial sum，跨 block 走 full attention。**这样 memory + 通信开销从 $O(Ld)$ 降到 $O(Nd)$，推理时还能用 online softmax 把 block 间和 block 内合并掉，** 推理延迟有显著下降。论文说 $N \approx 8$ 几乎能恢复 full 的全部收益（出自同期 AttnRes 论文）。

这个想法坦白说不是 K3 首创，但把它落到 2.8T + 8 个 block + 配套 kernel（prefill 用 sequence parallelism，decoding 用 stream overlap + 在线 softmax 融合）是另一个量级的工程。

---

## Stable LatentMoE：极端稀疏下的三个稳定器

把 routed expert 从 K2 的 384 拉到 **896，激活数 8 → 16（sparsity 56）**，单纯堆规模会出两个事故：

1. **路由路径有 4 个串联的 matmul**（$W^\downarrow$、GLU 内部、$W^\uparrow$、再叠加 shared 路径），2.8T 规模下激活会爆炸；
2. **~10³ 数量级的 expert load balancing** 远超 DeepSeek-V3 那种 auxiliary-loss-free 偏置更新的舒适区。

K3 的三个稳定器分别打这两个问题：

- **Normalized LatentMoE**：up-projection 前加 RMSNorm，抑制 routed path 激活爆炸，顺带还能改善 val loss；
- **SiTU-GLU**：把 SwiGLU 的两个无界因子都用 $\beta \tanh(x/\beta)$ softcap，gate branch $\beta_1 = 4$、up branch $\beta_2 = 25$。论文里的 fig:situglu 把 GLU / SwiGLU / SiTU-GLU 三家放在一起比——**SiTU-GLU 保留了 SwiGLU 在 0 附近的局部响应，但 $|f(x)| \le \beta_1 \beta_2 = 100$，大坐标不会溢出**。这是把"低精度算 + 大稀疏 MoE"能训得动的关键补丁；
- **Quantile Balancing（QB）**：替代 DeepSeek-V3 的固定步长偏置更新，**直接从 token-wise margin 的 $(1 - k/n)$-quantile 反推每个 expert 的偏置**——

$$
\hat b_j^{(t+1)} \leftarrow -\text{quantile}_{1-k/n}(s_{:,j} - \alpha^{(t)})
$$

意思是要让 expert $j$ 拿到目标 $q = mk/n$ 个 token，就把"score 减去 top-$(k+1)$ 截断"后的 margin 的 $(q+1)$-th 最大值取负。**一次性 forward 就能确定下一轮的偏置**，而 V3 的 sign-based 更新实质上是个 PI 控制器（$\gamma$ 调小慢、 $\gamma$ 调大抖）。当 expert 池到 896，PI 控起来就开始力不从心——所以用 quantile-based 一次性收敛更合适。工程上用 histogram 估计来做跨 rank 聚合（几个百量级的 bin 而非上百万个 margin）。

---

## 视觉：MoonViT-V2 从头训，不要 SigLIP 初始化

这是个我觉得很有意思的小决策：**MoonViT-V2 不再以 SigLIP 对比预训练做初始化，而是完全用 next-token prediction 从头训**。论文给出的理由不是性能而是稳定性——

![MoonViT-3D（SigLIP 初始化）和 MoonViT-V2（从头训）的 vision-tower 梯度范数对比](https://www.mulanai.com/fs/files/0729_2bd93d3f_vt-grad-.png) vt-grad-norm 这张图（左：完整 7k–30k 步，右：14k–16k 步 zoom-in）一目了然：MoonViT-3D（蓝）持续出现高于 0.4 的梯度范数尖峰，而 MoonViT-V2（红）基本贴在 0.05 以下。**这是 SigLIP loss 目标和 next-token prediction loss 目标方向不一致导致的训练抖动**。更关键的是去掉这个初始化后视觉任务成绩没有掉，意味着对比预训练做初始化在 3T 规模上"不必要且有害"。

ViT 本身是 27 层、约 0.4B 参数、12 头、patch 14，attention 拆成 intra-frame 空间 + inter-frame 时序两个 pass，2×2 pixel shuffle 降采样，最多支持 3584×3584 像素的图（在 1M 上下文里 afford 得起）。

---

## Pre-Training：cosine schedule、1M 上下文、2.5× scaling efficiency

**Scaling law** 单独重做：K2 系列的架构、数据、超参都变了，所以 batch size、learning rate、TPP、shape 全部重 search。

![K2 vs K3 缩放律曲线：K3 在同等 FLOPs 下 validation loss 明显更低](https://www.mulanai.com/fs/files/0729_09ed3827_scaling-.png) 拟合结果显示 K3 比 K2 **scaling efficiency 提升约 2.5×**——同样算力下 loss 显著更低，或者同样 loss 所需 FLOPs 大幅下降。

几个细节值得指出：
- **cosine 调度胜过 WSD**——但 K3 给出了一个诚实的 caveat：两者的 optimal hyperparam（peak LR、batch size）差异很大，共享超参对比对 WSD 不公平。**独立 search 之后 cosine 才一致胜出**；
- 训练从 8K context 起，扩展到 64K，然后 cooldown 阶段再从 256K → 1M（"four-stage curriculum"）；
- 1M 上下文靠 **NoPE + KDA 的递推门控**提供位置信息，省去了 RoPE 重缩放 / YaRN 之类的招；
- 长上下文数据通过"打散 + 拼接"合成，确保模型必须 attend 全 1M 上下文才能答对。

这些都已经是行业标配，但 K3 在 2.8T 规模上把整套配方重跑了，**经验本身比纸面结论更值钱**。

---

## 后训练：SFT → 三档 RL → MOPD

后训练是分三段：SFT 冷启动 → 多档 RL → Multi-Teacher On-Policy Distillation（MOPD）收口。

### RL 的关键招：partial rollout + reasoning effort 分档

长程任务的 RL 最大痛点是 **stragglers**——同一批 prompt 里 90% rollout 跑完了，剩下 10% 长 trajectory 还在慢慢跑，整个 iteration 都在等。K3 把 K1.5 时代就有的 partial rollout 进一步扩展：fraction $\lambda$ 完成的 prompt 立刻去 policy update，剩下的排队等下一轮。

> 但 partial rollout 天然引入 data staleness（一条 trajectory 跨多个 iteration），所以 policy optimization 必须"per-token 局部约束"——把更新限制在 policy 的局部邻域内，对 stale 数据天然鲁棒。

分档方面：把 reasoning effort 设成 low / high / max 三档，每档配 $\tau \cdot b_0(x)$ 的 token budget，**超出 budget 的 trajectory 直接 -1 reward**。三个领域（general / agentic / coding）× 三档 effort = 9 个 expert。

**对不可验证的通用任务**用 Agentic Generative Reward Model（GRM），强制 judge 按 protocol 走：read outcome → generate rubric → score → write scorepad，避免 reward hacking 出"长就是好"。

![RL FLOPs 增长时各任务 Score 和 Avg. steps 同步上升](https://www.mulanai.com/fs/files/0729_49d7f99d_step_sca.png) step_scaling_merged 这张图分四列（开发经验 / 通用工具 / Web 开发 / 智能体搜索）和四列（专业工作流 / Office 交付物 / 智能体图表理解 / 智能体视觉谜题）。**Score（实线蓝）随 RL FLOPs 单调上升，Avg. steps（虚线红）也在涨**——模型变强的同时能"用更多步去想"。

### MOPD：把 9 个 expert 收口到一个模型

9 个 expert 部署时太贵，**MOPD 把它们收成一个统一模型**。训练时按 prompt 的领域 $d$ + 采样得到的 effort $e$ 选对应 teacher $\pi^{(d,e)}_{\text{teacher}}$，per-token 蒸馏 reward：

$$
r^d_{\text{opd}}(y_t | e, x, y_{<t}) = \text{clip}\left(\text{sg}\left(\log \frac{\pi^{(d,e)}_{\text{teacher}}(y_t|x,y_{<t})}{\pi_\theta(y_t|e,x,y_{<t})}\right), -R_{\max}, R_{\max}\right)
$$

> stop-gradient + clip 让 reward 数值稳定，同时和 partial rollout 兼容。**作者试过更细的 top-$k$ 蒸馏目标，**坦率讲，结果差不多——没看到明显收益。**

### 任务合成：知识图谱 + 白盒环境 + AgentENV

K3 在 post-training 数据上做了两件事值得单独说：

**知识图谱引导的任务合成**：用 agent 在 web 规模上递归扩展一个有向无环概念图（粗粒度种子 → 多个 agent → web search → 细粒度子概念），然后从图中 sample 节点 + 检索真实材料 + 让合成 agent 生成各种类型的训练任务。**这条 pipeline 解决了"高质量长尾任务从哪里来"**——纯爬数据显然不够。

![知识图谱引导的任务合成流程](https://www.mulanai.com/fs/files/0729_ef262ef2_k3_task_.png) 左侧是分层知识图（CS/AI、Biology、Coding、Humanities、Math、Chemistry、Physics 等等级节点），右侧是 keyword set → material retrieval → task synthesis 的 pipeline。**这套机制把"任务从哪里来"从手工标注变成了"图谱 + 检索 + 合成"的自动化**。

**白盒 RL 环境**：训练时不能只用一个 agent harness（Kimi Code / Claude Code / Codex / OpenClaw / Hermes 各自有不同的 system prompt、tool schema、context 管理方式）——会被单一 harness 过拟合。**K3 把 harness 拆成可组合模块**（tool interface / system prompt / context management / skill / memory / subagent），训练时**对不同 task group 动态组合不同配置**，做 cross-harness 泛化。挺务实的设计。

任务类型覆盖：可验证搜索、专业知识工作（投行 / 数据分析 / 法律）、SWE、内核优化、多模态 reasoning with vision-in-the-loop tool、persistent assistant workflow（用 mock 的 Gmail / Notion / Slack / Canvas）、autonomous execution、web development。**这些任务的实际工程量加起来得有几万级别的 environment instance。**

### 部署感知：MXFP4 + 草稿模型

整个 post-training（SFT + RL）都做 **MXFP4 weights / MXFP8 activations 量化感知训练**，rollout 和训练用同一份量化方案——**消除 train-inference mismatch**。

Speculative decoding 走 EAGLE-3 路线，**把 K3 预训练里那个 MTP layer 直接微调成 draft model**。草稿训练目标用 **LK loss**（即"接受率"的负 log）而不是 KL：

$$
\mathcal{L}_{\text{LK}} = -\log \sum_{x \in V} \min(p(x), q(x))
$$

> 作者说，**KL surrogate 并不直接对应"接受率"**，特别是对 capacity-limited draft model 来说，直接优化接受率更直接。这个 loss 在温度 1 下、不加 ground-truth CE term 时效果最好。

草稿 input 拼接 target model 第 1 / 第 4 / 最末 AttnRes block 的输出（低 / 中 / 高层特征），用 $[\,0\;\; 0\;\; I\,]$ 初始化投影矩阵，所以初始 fused rep 等于 high-level feature（MTP layer 预训练时的输入），微调时再慢慢学用上低 / 中层。

---

## 基础设施：1M context + 3T 训练的工程戏法

这一章是 K3 报告里最像"系统论文"的一章。**三个工程挑战**罕见地集中在同一模型上：

| 挑战 | 原因 | 核心方案 |
|---|---|---|
| Hybrid KDA 注意力 | 序列方向上有两种不同 attention 范式 | FlashKDA kernel + KDA Context Parallelism (KCP) |
| 3T 级稀疏多模态训练 | 算力/内存 / 多模态变长 | MoonEP（平衡 EP）+ 统一 activation manager + encoder PP 隐藏 |
| 1M token 智能体 RL | rollout 跨 iteration、state 持久 | 外部 KV cache pool + auto-throttling + AgentENV microVM |

### FlashKDA + KCP

KDA 的递推形式跟 GPU 的 wide-parallel 偏好天然冲突：
- **训练/prefill**：chunkwise，intra-chunk parallel / inter-chunk serial。K3 用 CUTLASS 写了个 FlashKDA，token-parallel + head-parallel 双 schedule 独立调优，**比 Triton 参考实现快很多**；
- **超长 prefill**：单卡 SM 浪费——纯 TP 时每 rank 只分到几个 head。K3 用 SM-level context parallel 在**单卡内**做序列切分，segment 转换可独立算后合并，**无跨设备通信**。

**KDA Context Parallelism（KCP）** 是核心创新点。线性注意力的传统 CP 是"假设 $S=0$ 算 local state，然后跨 rank 求和"——但 KDA 不行，因为递推里有 token-dependent 的 transition matrix $M_t$。K3 把每段的效应拆成 **本地生成的 state** + **cumulative transition 作用在 incoming state**：

$$
S_{[i+1]}^t = \tilde S_{[i+1]}^t + M_{[i+1]}^{t \leftarrow 1} S_{[i]}^{T_i}
$$

两个量都只用本段 token 算得出来，**一次 all-gather + prefix scan 就能在 $O(\text{ranks})$ 时间恢复每个 rank 的 incoming state**。通信量是 fixed-size recurrent state 而不是 KV 块——线性注意力相对 softmax 注意力的"省通信"优势在长序列下被彻底放大了。

### MoonEP：完美平衡的 Expert Parallel

传统 EP 跨 rank 严重不均，K3 的 **MoonEP** 通过"动态冗余 expert + online planning + zero-copy 通信"做到完美平衡。

关键定理：**E 个 expert 分给 R 个 rank 时，每个 rank 预留 $E/R$ 个冗余 expert 就能保证总是存在可行规划**，且这个 bound 基本上 tight。**对比 ECHO / UltraEP 这种"预设冗余数 + 不行就停训"的方案，MoonEP 是数学上保证"永不停"**。

零拷贝通过 fused permute/unpermute 算子 + planning kernel 预算 token 目的地，**通信 buffer 从 DeepEP 的 $S \times K \times R$ 降到 MoonEP 的 $S \times K$**（worst-case 不平衡时也保得住）。完美平衡还带来一个副产物——**所有 rank 收到的 token 数都正好是 $S \times K$，shape 静态已知，省去 per-layer host sync**。

> 这条对 2.8T 训练吞吐是**决定性的**。MoonEP 是开源的（github.com/MoonshotAI/MoonEP），值得一读源码。

### 长程 RL：外部 KV 池 + AgentENV

1M context 多步 rollout 痛点：partial rollout 起始处积压的 unfinished long prefill requests，spec decoding 加速请求轮转的同时带来 prefix-block churn，**导致 cache miss 率飙升**。

K3 把 GPU KV cache 和 CPU 写回解耦：**active decode blocks 留在 GPU KV，idle prefixes 在 evict 时写回外部 KV pool（CPU DRAM），下次 reuse 时 prefetch 回来**。KDA state 跟对应的 MLA KV 块一起 offload / prefetch，lifecycle 保持一致。训练时再把训练 state 写到 NVMe 释放 DRAM 给 rollout。

Sandbox 是个独立巨坑——他们写了个叫 **AgentENV** 的 microVM runtime：
- **Checkpoint 133ms / Resume 49ms**（增量 dirty page）；
- **Pause/Resume/Fork/Snapshot** 四个高层操作；pause 时 sandbox 几乎不占资源，**agent 等待 inference 时（占 sandbox 生命周期的 98%）就可以 pause**；
- 高密度：上万个 sandbox 同时创建，**6.5× 内存 overcommit**。

整个训练过程**造了 51,219,741 个 sandbox，跨 1,505,678 张镜像**。这个数字是相当炸裂的工程量。

### 部署：KDA-aware prefix cache

Hybrid 架构下，**MLA KV cache 随序列增长 + KDA state 固定大小 + 必须同时 restore 才能复用**——K3 用了 unified paged layout 解决。但更细的是 **fine-grained prefix hash 块（512 token）+ coarse physical block（6144 token）解耦**：

MLA 用 512-token hash block 做 hash matching，KDA 只在 hash 端点处（sparse 边界，例如 conversation turn）保存 state checkpoint。**请求在物理块内部的任意 hash 端点命中后，从该端点恢复 KDA state + 复制 MLA 块的剩余部分，跳过对前面 token 的 recompute**。Hash granularity 和 KDA checkpoint 边界统一，cache miss cost 显著降下来。下面这张 K3 预训练 pipeline 图（k3pp）展示了 2.8T 训练时不同 PP 阶段上 Computation、EP Comm、NCCL Comm、Activation Offload 是怎么相互 overlap 的——同样的 overlap 思路在 1M context RL 里也用上了：

![K3 预训练 execution schedule：Computation / EP Comm / NCCL Comm / Offload 在不同 PP 阶段上 overlap](https://www.mulanai.com/fs/files/0729_496731fa_k3pp.png)

并发一致性的三条规则：所有 cache group 共享 free list（hit 块跨组 pin 住防止 evict 竞争）、copy 在 forward 前 GPU 端做（不暴露上一 owner 的旧字节）、跨 KDA group checkpoint atomic 失效（要么都能 hit 要么都不能）。

---

## 评测：开源第一、闭源第二、价格效率贴着 Pareto

### 主表：扫一眼就知道 K3 的位置

K3 跑了 50+ 个 benchmark，涵盖 reasoning / coding / agentic / vision 四块。从结果可以归成几档：

![K3 主结果：coding、agentic、vision 三块共 12 个 sub-benchmark 对比。蓝色 bar 是 K3，星标是第一名](https://www.mulanai.com/fs/files/0729_c371bad0_benchmar.jpg)

| 档次 | 代表 benchmark | 解读 |
|---|---|---|
| K3 **第一** | BrowseComp 91.2 / AutomationBench 30.8 / MCPMark 94.5 / SWE-Marathon 42.0 / ProgramBench 77.8 / DeepSearchQA 95.0 / ResearchRubrics 76.2 / OmniDocBench 91.1 / Video-MME 90.0 / MMVU 82.1 / SpreadsheetBench 2 34.8 / τ³-Banking 33.4 / Harvey Lab-AA 94.6 / ZeroBench w/ tool 41.0 | 智能体 + 视觉 + 部分搜索类，K3 比所有对比模型都强 |
| K3 **第二** | HLE-Full / FrontierSWE 81.2 / Toolathlon / DeepSWE / Agents' Last Exam / OfficeQA Pro / OSWorld-Verified / AA-Briefcase / CorpFin v2 / Math-Vision w/ tool | 略输 Fable 5（有时输给 GPT-5.6 Sol），但明显领先 GLM-5.2、Opus 4.8、GPT-5.5 |
| K3 **第三 / 落后** | HLE-Full / CritPt / OSWorld 2.0 / SaaS-Bench / Legal Research Bench | 仍输给 Fable 5 和 GPT-5.6 Sol，**研究级 reasoning 和 OSWorld 2.0 的 advanced computer use 仍是差距** |

K3 主结果图（fig:kimi-k3-results）把 coding、agentic、vision 三类共 12 个 sub-benchmark 集成到一张 dashboard 上，蓝色 bar 是 K3，星标是第一名。**最显眼的是 BrowseComp、AutomationBench、SWE-Marathon 三个条——K3 都是 #1，且大幅领先第二名**。

### 第三方：WebDev Arena 第一，Intelligence Index 第四

三个独立机构评测：
- **Artificial Analysis**：Intelligence Index v4.1 = **57.1**，580 个模型中排第 4（前 3 是 Fable 5 / GPT-5.6 Sol 各 effort 变体）；
- **Vals AI**：Vals Index **74.7%**，39 个模型中排第 2（输 Fable 5 的 75.1%，赢 GPT-5.6 Sol 的 73.1%）；
- **Arena**：**WebDev Arena 1,678 Elo 第一**（Fable 5 1,634 第二）——**第一个登顶的开源模型**；Text Arena 1,486 Elo 第 8；Agent Arena 第 4。

> **"第一个登顶 WebDev Arena 的开源模型"这条，对中文开源社区意义很大。**以前 WebDev 这种长尾、可视化密集的任务，开源模型基本不在 top 3。

### 价格效率：贴着 Pareto 走

K3 在四个 benchmark 上画了 score-cost 曲线（KCB 2.0 / BrowseComp / GDPval-AA v2 / AA-Briefcase）：

- **KCB 2.0**：K3 比 Fable 5 低 4 分但价格只有 **38%**；high effort 已经和 Opus 4.8 max-effort 持平，价格 **1/3**；
- **BrowseComp**：K3 拿第一 91.2%，**每任务 \$2.03**——GPT-5.6 Sol 90.4% 价格是其 2 倍，Claude 系 max effort 贵一个数量级；
- **GDPval-AA v2**：K3 距 GPT-5.6 Sol 50 Elo 之内，**价格低 13%**，比 Fable 5 便宜 **2.6×**；
- **AA-Briefcase**：K3 第二（1548 Elo），价格约 Fable 5 一半。

> **这是我觉得 K3 这次最值得工程团队关注的数字。** 同样成本预算下你能拿到的智能，现在是 K3 + Kimi Code 这条线最划算，**特别是长程 agentic 任务**。

---

## 内部评测 + 案例研究：K3 的甜点在哪里

### 内部 suite

Kimi 自建了 17 个内部 benchmark（KCB 2.0、Coding Experience、Swarm Bench、Deep Research Bench、Finance Bench 等）。结果：K3 在 **Swarm Bench（76.3）和 Deep Research Bench（90.0）** 领跑，**KCB 2.0** 距 Fable 5 不远，**Coding Experience 拿到第一**（说明实战体感好），Kimi Webdev Bench 凭盲评专家打分**比 Opus 4.8 高 31 分（最大优势在 3D/WebGL/Shader 任务）**。

落后主要在 **Agent Behavior Bench / MIRA / 24-7 ClawBench 2.0 / Agentic Vision Bench / KWV Bench**。**在 KAET / CLIF / Online Experience / DECK / Faithfulness / Chat All-in-One 上 K3 排第一或第二**——比较平均。

### 案例研究

报告里放了 6 个 case study 都很能说明问题：

- **GPU kernel 优化**：4 个代表性 kernel（AttnRes / DSA / KDA / MLA-512），K3 在 AttnRes 上 latency 从 283.6ms → 114.4ms，DSA / KDA runtime 砍 55.1% / 73.6%，**MLA 跑到峰值 TFLOPS 的一半以上**。这些任务给的是 24 小时/任务的预算，K3 单独跑完基本 match Fable 5（带 fallback）并显著赢 Opus 4.8 / GPT-5.6 Sol / GPT-5.5；
- **GPU 编译器开发**：K3 自己写了个叫 **MiniTriton** 的 Triton-like 编译器（带 PTX codegen / NCCL 分布式 / sparse primitive / 可视化），在 L20 上 **90% 接近 cuBLAS roofline**，自训 GPT loss 曲线和 torch 参考曲线**贴在 $10^{-4}$ fp32 精度内**；
- **芯片设计**：48 小时单跑，K3 做出了 **4mm² 面积、100MHz、8,700 tokens/s 解码吞吐**的 inference chip prototype（开源在 MoonshotAI/nano-kpu），用 Nangate45 标准单元库；
- **科研复现**：2 小时复现"Universal Relations in Computational Astrophysics"全套 pipeline，**写了 3000+ 行 Python + 300 多个 EOS 评估 + 互动 HTML dashboard**——同等工作专家通常 1-2 周；
- **知识工作**：在 Kimi Work 里做出覆盖 42 年 AI ASIC 行业的交互式研究网站，**120 轮迭代、87 份季报 + 99 份原始 PDF（>11,000 页）、>2,800 次 web 搜索、>1,100 次 terminal query**；第二例是分析 391 个 GWTC-5 引力波事件，**20+ 并发子智能体**产出 7 张科学可视化、2 张总结表、10+ 文献综述；
- **视频剪辑**：K3 用原生多模态架构 **自己制作了讲自己架构的 3Blue1Brown 风格 motion graphics** + 剪辑了 56 段原始素材的预告片，含 motion-matched 切分、beat 同步、音频处理、多轮修改。

> **这些 case 的共同点是"长程 + 多工具 + 验证闭环"**——不是"问一答一"，而是"目标 → 计划 → 工具调用 → 反馈 → 修正"的真实工程量。**这正是 K3 训练时把 AET 任务和 partial rollout + 推理 effort 调度 + AgentENV microVM 一起做出来的现实回报**。

---

## 网络安全评测：值得单独一节

这部分其实不在主结果图里，但作为"安全 / 红队 / 防御"参考意义很大：

- **Tier 1（漏洞发现）**：在几十个广泛部署的系统里 K3 找出数百个候选漏洞，**人工 review 后约 70% 被确认是真的**——其中 16 个是 6 个项目里**之前未知的漏洞**。亮点是两个 Linux 内核发现：（1）由上游 incomplete fix 引入的远程 OOB write，**影响所有后续版本含最新**；（2）RDMA 子系统的 Dirty-COW-class 漏洞，**一个 upstream fix 误删了权限检查**；
- **Tier 2（端到端 exploit）**：36 任务（16 user-space + 20 kernel），K3 解决 14 任务（**38.9%**）vs GLM-5.2 的 8 任务（22.2%），user-space 10/14 成功；
- **第三方独立评估**：UK AISI + NIST CAISI 联合评估，K3 在 ExploitBench 上 32% vs GLM-5.2 24%，**但 41 任务 end-to-end exploit 上 0/41 拿到 arbitrary code execution**。

作者明确说这是当前 model 版本下的 lower bound，会随版本更新重测。**公平地讲，K3 在 offensive security 维度上"能到"但还没"封顶"，这是合理的——也是负责任的**。

---

## 我的判断

读完整篇报告，**我最大的感受是 K3 是"工业化 frontier 集大成者"，而不是某一个 idea 的颠覆**。它把过去 1-2 年里学界/工业界在 attention scaling、MoE scaling、RL 后训练、long-context serving 几个维度上的"成熟招数"全集成到 2.8T 规模，并且通过开源让整个社区复用。

**强在哪**：
1. **架构三维混合**（KDA 长度方向 / AttnRes 深度方向 / Stable LatentMoE 通道方向）的思路清晰且配套齐全；
2. **3T 级训练效率**：2.5× scaling efficiency 不是"嘴炮"，是从架构、数据、训练 recipe 一起挤出来的；
3. **长程 RL 基础设施**：partial rollout、AgentENV microVM、外部 KV 池、KDA-aware prefix cache——**这堆系统组件联合起来才让 1M context × 9 档 expert 的 RL 跑得起来**，这才是"最稀缺的部分"；
4. **价格/质量 Pareto**：在 4 个 score-cost 图上几乎贴着前沿走，**对工程团队来说 K3 + Kimi Code 是目前性价比最高的组合**；
5. **开源完整性**：weights + 配套 sandbox（AgentENV）+ EDA 设计（nano-kpu）+ 编译器（MiniTriton），**生态完整度罕见**。

**仍有的差距**：
1. **研究级 reasoning 仍弱**（HLE-Full 43.5% / 56.0%，CritPt 23.4%，远输 Fable 5 / GPT-5.6 Sol）；
2. **OSWorld 2.0 / SaaS-Bench / Legal Research 这类 advanced computer use** 上 K3 仍落后；
3. **Tier 2 exploit 的"收尾阶段"和"kernel track"** 都不行，作者列的 4 个 failure mode（chain 收尾、control-flow vs data-only 选择、debug loop、verify-before-submit）都还需要 RL 范式层面的进展；
4. **knowledge reasoning 离人类研究级还差明显**——这不是堆参数能解决的，可能需要新的 search / memory / tool integration 范式。

**对工程团队的启发**：
- 如果你做 1M context / 长程 agent / 多工具编排，**直接试 K3 + Kimi Code**——这是现在性价比最好的方案；
- 如果你做 RL 训练，**partial rollout + per-token 局部约束 + 推理 effort 分档**这套范式值得学，实质上是在 stale data 上做受约束 policy update；
- 如果你做 MoE 训练，**MoonEP（数学保证 + 在线规划 + 静态 shape）是当前最 solid 的 EP 方案**，源码可以读；
- 如果你做 chip / 编译器，**报告里那两个 case study（nano-kpu + MiniTriton）展示了 K3 在"长程 + 工具链 + 验证闭环"任务上的真实能力**——48 小时从 RTL 到 verified chip，从 0 写 compiler 到 90% cuBLAS roofline。

**一句话收尾**：开源大模型这场仗打到 2026 年下半年，**真正决定下一阶段领先程度的不是"参数谁更大"或"数据集谁更新"——而是"在多长程任务里，模型能否自主规划、调用工具、验证结果并修订"**。K3 这篇报告最大的贡献是把这条路上每一段（架构 / 训练 / RL / 基础设施 / 部署）都展示了当前最 solid 的工程实现，并把它们连成端到端。**剩下的差距，坦白讲，是科学问题，不是工程问题**。

---

## 写在收尾

如果觉得这篇解读有启发，欢迎点赞、在看、转发。跟进 K3 后续在 RL 后训练 / 长程 agent 方向上的工作，以及开源社区基于 K3 的微调 / 蒸馏进展——这两条线在 2026 下半年值得持续关注。
