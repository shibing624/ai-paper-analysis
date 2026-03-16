# MiniCPM-SALA：让Transformer在百万token下跑起来

> 一句话总结：混合稀疏注意力和线性注意力（1:3比例），用持续训练降低75%成本，在消费级显卡上支持1M token上下文。

---

## 📖 为什么Transformer"吃"不了长文本

Transformer的自注意力机制有一个致命伤：**复杂度随序列长度平方增长**。

传统注意力的计算量：$O(N^2 \times d)$

当N=128K时，计算量是N=4K时的1024倍。这导致：
- 128K tokens在消费级显卡上直接OOM
- 256K tokens在专业显卡上也吃不消
- 1M tokens？想都别想

现有的解决方案各有取舍：

| 方案 | 复杂度 | 优点 | 缺点 |
|-----|--------|------|------|
| 全注意力 | $O(N^2)$ | 最高精度 | 内存爆炸 |
| 滑动窗口 | $O(NW)$ | 局部高效 | 全局信息丢失 |
| 线性注意力 | $O(N)$ | 全局高效 | 精度下降 |
| 稀疏注意力 | $O(N\sqrt{N})$ | 平衡精度和效率 | 复杂度高 |

**核心矛盾**：线性注意力效率高但精度差，稀疏注意力精度好但实现复杂。有没有办法"鱼和熊掌兼得"？

这就是MiniCPM-SALA要解决的问题。

---

## 🔍 核心设计：1:3混合架构

### 架构概览

![图1：架构概览](https://arxiv.org/html/2602.11761/x1.png)

*图1：MiniCPM-SALA的混合架构设计。25%的层使用稀疏注意力（InfLLM-V2），75%的层使用线性注意力（Lightning Attention）。*

MiniCPM-SALA采用"分工合作"的策略：

```
┌─────────────────────────────────────────────────────┐
│  Layer 0:  Lightning Attention (线性注意力)          │  高效全局建模
├─────────────────────────────────────────────────────┤
│  Layer 1:  Lightning Attention                       │  高效全局建模
├─────────────────────────────────────────────────────┤
│  Layer 2:  Lightning Attention                       │  高效全局建模
├─────────────────────────────────────────────────────┤
│  Layer 3:  InfLLM-V2 (稀疏注意力)                    │  高保真局部精确
├─────────────────────────────────────────────────────┤
│  Layer 4:  Lightning Attention                       │  高效全局建模
├─────────────────────────────────────────────────────┤
│  ... 每4层重复一次 ...                               │
└─────────────────────────────────────────────────────┘
```

**为什么是1:3？**

论文通过实验发现：
- 稀疏注意力太少（如1:7）：长上下文精度不够
- 稀疏注意力太多（如1:1）：效率优势不明显
- 1:3是帕累托最优：精度与效率的最佳平衡

这就像团队分工：
- 25%的"专家"负责高难度细节（稀疏注意力）
- 75%的"工兵"负责大面积覆盖（线性注意力）
- 既保证质量，又保证效率

### 两种注意力机制详解

#### 线性注意力：Lightning Attention

**核心思想**：把注意力的softmax分解，利用矩阵乘法结合律降维

传统注意力：
$$\text{Attention}(Q, K, V) = \text{softmax}(QK^T)V$$
复杂度：$O(N^2 \times d)$

线性注意力：
$$\text{LinearAttn}(Q, K, V) = \phi(Q)(\phi(K)^T V)$$
复杂度：$O(N \times d^2)$

其中$\phi$是特征映射函数（如ELU+1）。

**关键优化**：
- 利用cumsum实现left-to-right扫描，复杂度真正降到$O(N)$
- 在MiniCPM-SALA中，线性注意力层负责全局信息整合

#### 稀疏注意力：InfLLM-V2

**核心思想**：只关注"关键token"，忽略无关信息

![图2：稀疏注意力机制](https://arxiv.org/html/2602.11761/x2.png)

*图2：InfLLM-V2的稀疏注意力模式。每个token只关注局部窗口+全局关键token，大幅减少计算量。*

InfLLM-V2的注意力模式：
1. **局部窗口**：关注最近的W个token（如W=1024）
2. **全局关键token**：通过评分机制选出全局重要的token
3. **分块策略**：将序列分成块，每个块维护一个"代表"

这就像读书：
- 线性注意力：快速浏览全文，抓住大意
- 稀疏注意力：精读关键段落，确保细节

### HyPE：混合位置编码

这是论文的一个关键创新。传统做法：所有层都用RoPE（旋转位置编码）。

**问题**：RoPE在超长上下文中会导致信息衰减。位置越远，信号越弱。

**HyPE的设计**：

| 层类型 | 位置编码 | 原因 |
|-------|---------|------|
| 线性注意力层 | RoPE | 保持位置敏感记忆 |
| 稀疏注意力层 | 无RoPE | 防止长距离信息衰减 |

**直觉理解**：
- 线性注意力需要位置信息来区分不同位置的内容
- 稀疏注意力已经通过"关键token"机制定位重要信息，不需要额外位置编码干扰

这个设计的精妙之处在于**差异化处理**：不同层有不同的"职责"，因此需要不同的工具。

### 其他架构改进

**QK-Normalization**

在长上下文训练中，Q和K的点积会产生巨大的激活值尖峰，导致训练不稳定。论文在所有注意力层加入QK归一化，缓解这个问题。

**输出门（Output Gate）**

每个注意力块后加入输出门，缓解"注意力汇聚"问题——即所有token的注意力都集中在某几个token上，导致信息压缩过度。

---

## 📊 训练策略：75%成本降低的秘密

### HALO框架：Transformer转Hybrid

MiniCPM-SALA不是从头训练的，而是基于MiniCPM-4.0的预训练检查点"改装"。

![图3：训练流程](https://arxiv.org/html/2602.11761/x3.png)

*图3：从Transformer到混合架构的持续训练流程。5个阶段逐步扩展上下文长度，总成本约2T tokens。*

**传统做法**：从头训练一个混合架构模型
- 成本：约8T tokens
- 问题：收敛慢，不稳定

**HALO做法**：将预训练的Transformer转换为混合架构
- 成本：约2T tokens
- 优势：利用已有知识，快速收敛

**转换过程**：

```
原始: [Attention] → [Attention] → [Attention] → [Attention] → ...
                              ↓ HALO转换
混合: [Linear] → [Linear] → [Linear] → [Sparse] → [Linear] → ...
```

### 五阶段训练流程

| 阶段 | 目标 | 序列长度 | 数据量 |
|-----|------|---------|-------|
| 1. 架构转换 | Softmax→线性注意力 | 0.5K | 1.3B tokens |
| 2. 稳定训练 | 协调各组件 | 4K | 314.6B tokens |
| 3. 短衰减训练 | 提升质量 | 4K | 1006.6B tokens |
| 4. 长衰减训练 | 扩展上下文 | 32K→520K | 102.2B-50.6B tokens |
| 5. SFT | 下游任务适应 | 64K→140K | ~100B tokens |

**关键设计**：
1. 先在短序列上稳定混合架构
2. 逐步扩展序列长度，避免一次性跳跃
3. 在超长序列训练阶段才启用稀疏注意力

**成本对比**：
- 从头训练：~8T tokens
- HALO持续训练：~2T tokens
- **节省75%**

---

## 📊 实验结果：效率与精度的双赢

### 通用能力：没有妥协

混合架构会不会牺牲通用能力？论文用标准基准测试验证：

| 任务 | MiniCPM-SALA | Qwen3-8B | 差异 |
|-----|-------------|----------|-----|
| HumanEval (代码) | 95.12 | 92.5 | +2.6 |
| MBPP (代码) | 89.11 | 87.2 | +1.9 |
| AIME24 (数学) | 83.75 | 80.1 | +3.7 |
| AIME25 (数学) | 78.33 | 75.2 | +3.1 |
| CMMLU (知识) | 81.55 | 79.8 | +1.8 |
| BBH (推理) | 81.55 | 80.2 | +1.4 |
| **平均** | **76.53** | **74.2** | **+2.3** |

**结论**：MiniCPM-SALA不仅没有损失通用能力，反而略有提升。这说明混合架构是"加法"，不是"替换"。

### 长上下文能力：惊艳的外推

这是MiniCPM-SALA的核心亮点。

![图4：RULER基准测试结果](https://arxiv.org/html/2602.11761/x4.png)

*图4：RULER（大海捞针类任务）在不同上下文长度下的表现。MiniCPM-SALA训练到520K，但在2048K下仍保持高精度。*

**训练长度 vs 测试长度**：

| 模型 | 训练长度 | 128K | 512K | 1024K | 2048K |
|-----|---------|------|------|-------|-------|
| MiniCPM-SALA | 520K | **89.37** | 86.2 | 86.3 | **81.6** |
| Qwen3-8B | 128K | 85.2 | OOM | OOM | OOM |
| Qwen3-Next-80B | ? | 87.5 | 83.1 | 80.3 | - |

**关键发现**：
1. MiniCPM-SALA训练长度520K，但测试能到2048K（2M tokens）
2. 在1024K下，9B参数的MiniCPM-SALA超过80B参数的Qwen3-Next
3. Qwen3-8B在512K就OOM了

**为什么外推这么强？**

论文归功于HyPE设计：稀疏注意力层移除RoPE后，不再受位置编码的外推限制。这就像取消了"最大读取范围"的限制，模型可以"看得更远"。

### 推理效率：3.5倍速度提升

![图5：推理速度对比](https://arxiv.org/html/2602.11761/x5.png)

*图5：不同序列长度下的首字延迟（TTFT）。MiniCPM-SALA在256K时比Qwen3-8B快3.5倍。*

**A6000D (96GB) 上的结果**：

| 序列长度 | Qwen3-8B | MiniCPM-SALA | 加速比 |
|---------|----------|-------------|-------|
| 64K | 12.3s | 5.8s | 2.1x |
| 128K | 45.6s | 14.2s | 3.2x |
| 256K | 180.8s | 51.6s | **3.5x** |
| 512K | OOM | 142.3s | - |
| 1024K | OOM | 523.1s | - |

**RTX 5090 (32GB消费级显卡)**：

| 序列长度 | Qwen3-8B (量化) | MiniCPM-SALA |
|---------|----------------|-------------|
| 64K | 成功 | 成功 |
| 128K | OOM | 成功 |
| 256K | OOM (量化后) | 成功 |
| 512K | OOM | 成功 |
| 1024K | OOM | **成功** |

**关键突破**：MiniCPM-SALA在消费级显卡上成功运行1M token上下文，无需量化。这是全注意力8B模型无法做到的。

---

## 💡 我的观点和启发

### 混合架构的哲学

MiniCPM-SALA的成功揭示了一个重要原则：**不同任务需要不同的机制**。

线性注意力擅长：
- 全局信息整合
- 快速扫描大量数据
- 效率优先的场景

稀疏注意力擅长：
- 精确检索关键信息
- 高保真局部建模
- 精度优先的场景

与其争论"谁更好"，不如"各取所长"。这让我想到软件开发中的"微服务架构"：不同的服务用不同的技术栈，而不是强求统一。

### HyPE的创新意义

HyPE（混合位置编码）是这篇论文最精妙的创新之一。

传统观念：所有层都要用位置编码，否则模型不知道"位置"。

HyPE的观点：位置编码是双刃剑。在稀疏注意力层，位置编码反而限制了对远距离信息的访问。

这让我想到一个类比：
- 传统做法：给每个人发GPS，但GPS在偏远地区没信号
- HyPE做法：给部分人发GPS，其他人靠"路标"导航

"路标"就是稀疏注意力中的关键token机制，不需要精确位置，只需要知道"方向"。

### 外推能力的工程价值

MiniCPM-SALA训练到520K，但能处理2048K。这种外推能力有巨大的工程价值：

**场景1：长文档处理**
- 训练时只见过~50万字的文档
- 部署时可以处理200万字的书

**场景2：多轮对话**
- 训练时的对话历史有限
- 部署时可以支持更长的对话历史

**场景3：代码仓库分析**
- 训练时的代码文件有限
- 部署时可以分析更大的代码库

这种"训练一次，受益长期"的特性，大大提高了模型的经济效益。

### 局限性和未来方向

论文也坦诚指出了几个局限：

1. **训练成本仍然较高**：2T tokens虽然比8T省很多，但仍然需要大量资源
2. **实现复杂度**：混合架构比单一架构更难实现和调试
3. **超参敏感**：1:3的比例是否对所有模型都最优？需要更多实验

我认为几个值得探索的方向：

**方向1：自适应比例**

目前1:3是固定的。是否可以根据任务动态调整？比如：
- 代码任务：增加稀疏注意力比例（需要精确检索）
- 对话任务：增加线性注意力比例（需要全局理解）

**方向2：更细粒度的混合**

目前是"层级别"混合。是否可以在"注意力头级别"混合？某些头用线性，某些头用稀疏？

**方向3：端到端训练**

目前是HALO转换。是否可以直接从头训练混合架构，同时优化稀疏注意力和线性注意力的参数？

---

## 🔧 技术细节：关键实现

### InfLLM-V2的稀疏注意力实现

```python
import torch
import torch.nn as nn

class SparseAttention(nn.Module):
    """InfLLM-V2风格的稀疏注意力"""
    
    def __init__(self, hidden_dim, num_heads, window_size=1024, num_global_tokens=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 投影
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 选择全局关键token（简化版：随机选择）
        global_indices = torch.randperm(seq_len)[:self.num_global_tokens]
        k_global = k[:, global_indices]
        v_global = v[:, global_indices]
        
        # 局部窗口注意力
        outputs = []
        for i in range(seq_len):
            # 窗口范围
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2)
            
            # 合并局部和全局
            k_local = k[:, start:end]
            v_local = v[:, start:end]
            k_combined = torch.cat([k_local, k_global], dim=1)
            v_combined = torch.cat([v_local, v_global], dim=1)
            
            # 计算注意力
            attn = torch.matmul(q[:, i:i+1], k_combined.transpose(-2, -1))
            attn = torch.softmax(attn / (self.head_dim ** 0.5), dim=-1)
            out = torch.matmul(attn, v_combined)
            outputs.append(out)
        
        output = torch.cat(outputs, dim=1)
        return self.o_proj(output.view(batch_size, seq_len, -1))
```

### Lightning Attention的线性实现

```python
class LightningAttention(nn.Module):
    """Lightning Attention线性注意力实现"""
    
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def feature_map(self, x):
        """特征映射函数：ELU + 1"""
        return torch.nn.functional.elu(x) + 1
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 投影
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 应用特征映射
        q = self.feature_map(q)
        k = self.feature_map(k)
        
        # 线性注意力: (K^T V) 先算，再与 Q 相乘
        # 复杂度: O(N * d^2) 而非 O(N^2 * d)
        
        # 计算累积和（left-to-right scan）
        # kv = cumsum(k * v)
        kv = torch.matmul(k.transpose(-2, -1), v)  # [batch, heads, d, d]
        
        # 应用RoPE（线性注意力层有位置编码）
        # ... RoPE实现省略 ...
        
        # 最终输出
        output = torch.matmul(q, kv)  # [batch, seq, heads, d]
        
        # 归一化
        k_sum = k.sum(dim=1, keepdim=True)  # [batch, 1, heads, d]
        output = output / (torch.matmul(q, k_sum.unsqueeze(-1)) + 1e-6)
        
        return self.o_proj(output.view(batch_size, seq_len, -1))
```

### 混合层的构建

```python
class HybridTransformerBlock(nn.Module):
    """混合Transformer块：根据层索引选择注意力类型"""
    
    def __init__(self, hidden_dim, num_heads, layer_idx, sparse_ratio=4):
        super().__init__()
        
        # 每 sparse_ratio 层使用一次稀疏注意力
        use_sparse = (layer_idx % sparse_ratio == 0)
        
        if use_sparse:
            self.attention = SparseAttention(hidden_dim, num_heads)
            self.use_rope = False  # 稀疏注意力不用RoPE
        else:
            self.attention = LightningAttention(hidden_dim, num_heads)
            self.use_rope = True  # 线性注意力用RoPE
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 输出门
        self.output_gate = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        # 注意力 + 残差
        attn_out = self.attention(self.norm1(x))
        x = x + torch.sigmoid(self.output_gate(attn_out)) * attn_out
        
        # FFN + 残差
        x = x + self.ffn(self.norm2(x))
        
        return x
```

---

## 📊 详细数据：内存与速度分析

### 内存占用对比

| 模型 | 64K内存 | 128K内存 | 256K内存 | 512K内存 | 1M内存 |
|-----|--------|---------|---------|---------|--------|
| Qwen3-8B (全注意力) | 18.2GB | 42.5GB | 128.3GB | OOM | OOM |
| MiniCPM-SALA | 6.8GB | 11.2GB | 20.1GB | 38.6GB | 72.3GB |
| 节省 | 62.6% | 73.6% | 84.3% | - | - |

### 吞吐量对比（tokens/s）

| 序列长度 | Qwen3-8B | MiniCPM-SALA | 加速比 |
|---------|----------|-------------|-------|
| 4K | 2845 | 2921 | 1.03x |
| 16K | 892 | 1056 | 1.18x |
| 64K | 156 | 312 | 2.00x |
| 128K | 42 | 124 | 2.95x |
| 256K | 12 | 42 | 3.50x |

随着序列长度增加，加速比越来越显著。这正是混合架构的优势所在。

---

## 📝 总结

MiniCPM-SALA这篇论文展示了如何通过**架构创新**而非单纯的模型扩大来解决长上下文问题。

核心贡献：

1. **1:3混合架构**：稀疏注意力保精度，线性注意力保效率
2. **HyPE位置编码**：差异化处理，解决外推瓶颈
3. **HALO持续训练**：从Transformer"改装"，节省75%成本
4. **工程落地**：消费级显卡跑1M token

实验结果证明：**效率和精度可以兼得**。MiniCPM-SALA在通用能力上与全注意力模型相当，在长上下文能力上更强，同时推理速度提升3.5倍。

对于LLM开发者，这篇论文的启示是：

- **不要执着于单一架构**：不同机制适合不同任务
- **重视位置编码的影响**：RoPE在长上下文中是双刃剑
- **持续训练比从零开始更高效**：利用已有检查点
- **外推能力有工程价值**：训练一次，长期受益

长上下文是LLM的核心能力之一。MiniCPM-SALA提供了一个可行的技术路径：**通过混合架构突破Transformer的效率瓶颈，让百万token的上下文成为现实**。

---

## 🔗 参考资料

- **论文原文**：[MiniCPM-SALA: Hybridizing Sparse and Linear Attention](https://arxiv.org/abs/2602.11761)
- **InfLLM**：[Leave No Context Behind](https://arxiv.org/abs/2404.07143)
- **Lightning Attention**：[Linear Attention with Efficient Implementation](https://arxiv.org/abs/2312.06828)
- **MiniCPM系列**：[面向端侧的高效LLM](https://github.com/OpenBMB/MiniCPM)
- **RoPE位置编码**：[Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
