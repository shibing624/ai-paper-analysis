# OpenSeeker：首个完全开源训练数据的搜索 Agent，单次 SFT 逆袭复杂工业管线

> **核心摘要**：搜索 Agent 的训练数据长期被大厂视为核心机密，导致学术界难以复现 DeepResearch 等前沿工作。上海交大团队开源的 OpenSeeker 打破了这一局面：通过“事实驱动的 QA 合成”和“去噪轨迹合成”构建了仅 11.7k 的高质量样本。令人惊讶的是，仅凭单次 SFT 训练，OpenSeeker 就在 BrowseComp-ZH 基准上取得了 48.4% 的准确率，超越了通义 DeepResearch 等采用复杂 RL 管线的工业级系统。

---

## 📋 论文信息

- **标题**：OpenSeeker: Democratizing Frontier Search Agents by Fully Open-Sourcing Training Data
- **作者**：Yuwen Du, Rui Ye, Shuo Tang, Xinyu Zhu, Yijun Lu, Yuzhu Cai, Siheng Chen
- **机构**：上海交通大学
- **发布日期**：2026年3月
- **开源资源**：
  - 🔗 代码：[GitHub - OpenSeeker](https://github.com/rui-ye/OpenSeeker)
  - 📊 数据：[HuggingFace - OpenSeeker-v1-Data](https://huggingface.co/datasets/OpenSeeker/OpenSeeker-v1-Data)
  - 🤖 模型：[HuggingFace - OpenSeeker-v1-30B-SFT](https://huggingface.co/OpenSeeker/OpenSeeker-v1-30B-SFT)

---

## 🎯 研究背景与动机

构建具备深度搜索能力的 Agent 面临高质量训练数据稀缺的问题。传统的 QA 数据集（如 SQuAD、TriviaQA）问题复杂度不足，无法满足多轮搜索、跨页面信息关联的需求。

虽然阿里通义的 WebSailor 等系列工作已采用递归构造知识图谱的方法生成复杂数据，但这些高质量合成数据通常未开源。大厂倾向于保持数据封闭，仅开源模型。OpenSeeker 的主要贡献在于开源了完整的训练数据和模型，为学术界研究搜索 Agent 提供了可复现的基线。

![图1：OpenSeeker 方法总览](https://arxiv.org/html/2603.15594v1/x1.png)

*图1：OpenSeeker 的核心方法流程——左侧为事实驱动的 QA 合成，右侧为去噪轨迹合成*

---

## 🧠 核心方法一：事实驱动的 QA 合成

OpenSeeker 延续了基于知识图谱递归构造复杂 QA 数据的方法，从真实的网页链接结构出发，反向生成需要遍历这些链接才能回答的问题。

### 具体流程

1. **图扩展**：从种子页面出发，沿着出边遍历，形成局部子图。
2. **实体提取与子图重组**：去除文本噪声，保留关键实体及其关系，构建实体子图。
3. **基于子图生成问题**：利用 LLM 根据实体子图生成需要多跳推理的问题。
4. **实体混淆**：将问题中的具体实体替换为模糊描述，迫使 Agent 在搜索过程中进行实体消歧。
5. **双重验证**：
   - **难度验证**：确保基础模型在无搜索情况下无法回答。
   - **可解性验证**：确保提供完整子图内容时模型能够正确回答。

![图2：QA 合成方法详解](https://arxiv.org/html/2603.15594v1/x2.png)

*图2：事实驱动的 QA 合成流程——从网页图的拓扑扩展到实体混淆，再到双重验证*

---

## 🔧 核心方法二：去噪轨迹合成

真实网页包含大量无关信息（如导航栏、广告）。OpenSeeker 提出在合成阶段去噪，在训练阶段还原噪声的策略。

- **合成阶段**：教师模型（如 GPT-4o）在生成轨迹时，历史步骤仅保留关键信息的摘要，仅最近一步保留完整网页内容。
- **训练阶段**：学生模型训练时，将历史步骤的摘要替换回原始的完整网页内容。

**批判性分析：数据噪声的必要性**
在标准的 Agent 数据合成流程中，教师模型和学生模型通常使用相同的工具和交互环境，不存在交互环境偏差。强行在训练时引入额外的噪声，其实际必要性存疑。同期的 Qwen WebAgent 系列论文对数据合成有细致探讨，但并未采用此类加噪操作。

![图3：去噪轨迹合成](https://arxiv.org/html/2603.15594v1/x3.png)

*图3：去噪轨迹合成的核心思想——教师模型在压缩的历史上下文中生成动作，学生模型在完整的噪声上下文中学习*

---

## 🧪 实验分析

### 实验设置

- **基础模型**：Qwen3-30B-A3B-Thinking-2507
- **训练数据**：11.7k 合成样本
- **训练方法**：单次 SFT

### 基准测试表现

在 BrowseComp-ZH（中文版）上，OpenSeeker 取得了 48.4% 的准确率，高于通义 DeepResearch 的 46.7%。在同参数量、仅 SFT 的模型对比中，OpenSeeker 优于使用 147k 样本的 MiroThinker（25.8%）。

**批判性分析：评估基准的合理性**
BrowseComp-ZH 的难度与原版英文 BrowseComp 存在显著差异。OpenSeeker 的数据合成针对原版 BrowseComp 的高难度多跳推理设计，使用此类高难度合成数据在难度较低的 BC-ZH 上测试，取得高分符合预期。在原版英文 BrowseComp 上，其成绩处于行业平均水平。因此，仅凭 BC-ZH 的分数得出单次 SFT 优于 RL 的结论论据不足。

![图4：实验结果对比](https://arxiv.org/html/2603.15594v1/figs/compare_ZH.png)

*图4：BrowseComp-ZH 上的性能对比*

![图5：工具调用对比](https://arxiv.org/html/2603.15594v1/figs/compare_tool_calls.png)

*图5：数据复杂度对比——OpenSeeker 的合成数据在工具调用次数等维度上较高*

---

## 💡 客观评价与局限性分析

1. **工程整合与微创新**：OpenSeeker 的核心技术栈（基于图的 QA 合成、轨迹过滤等）在阿里 Qwen WebAgent 和 WebSailor 系列工作中已有体现。其主要差异在于引入了加噪操作以及完全开源的策略。在评估该工作时，应客观认识其在底层创新与工程复现上的定位。
2. **数据质量优于数量**：实验表明，基于图拓扑定制的高难度多跳搜索数据（11.7k），其训练效果优于简单聚合现有数据集的大规模样本（如 MiroThinker 的 147k）。
3. **种子节点选择偏差**：论文未详细说明图扩展时种子页面的选择策略。种子覆盖面的偏差可能导致生成的 QA 分布存在局限性。
4. **实体混淆的边界**：实体混淆程度难以精确控制，过度混淆可能导致问题存在多个合理答案，影响数据质量。

---

## 🔗 与其他工作的对比

| 工作 | 团队性质 | 模型开源 | 数据开源 | 训练方式 | BrowseComp-ZH |
|------|---------|---------|---------|---------|---------------|
| OpenAI Deep Research | 工业 | ❌ | ❌ | 未知 | - |
| 通义 DeepResearch | 工业 | ✅ | ❌ | CPT+SFT+RL | 46.7% |
| REDSearcher | 工业+学术 | ✅ | 部分 | Mid-training+SFT+RL | 26.8% |
| OpenResearcher | 学术 | ✅ | ✅ | SFT | - |
| **OpenSeeker** | **学术** | ✅ | ✅ | **仅 SFT** | **48.4%** |

---

## 📝 总结

OpenSeeker 提供了一套可复现的高质量数据生产线，并开源了前沿搜索 Agent 的训练数据和模型。该工作证明了基于图拓扑定制的高难度多跳搜索数据，在单次 SFT 下即可达到良好的基准测试表现，为学术界提供了有价值的参考实现。

---

## 📚 参考文献

1. OpenSeeker GitHub: https://github.com/rui-ye/OpenSeeker
2. OpenSeeker 数据集: https://huggingface.co/datasets/OpenSeeker/OpenSeeker-v1-Data
3. BrowseComp 基准: https://openai.com/research/browsecomp

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我的微信公众号：机器懂语言*
