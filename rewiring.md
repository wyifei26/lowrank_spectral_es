# 初探 RL 激发大模型推理的底层机制

## 核心观点

RL 并非在真空中创造知识，其底层机制是谱空间重链接（Spectral Space Rewiring）。
通过在基座模型的奇异值分解（SVD）空间中对特征向量进行重链接，RL 对基座模型的 latent skills 进行组合泛化，从而涌现出高阶推理能力。

## 大纲

1. 回顾以往研究对于 RL 机制的分析

   * **能力本质之辩**：关于 RL 能否让大语言模型学到新能力的争论，从三个角度出发：

     1. pass@K 的比较
     2. 新的评价指标：CoT-pass@k
     3. Toy data 上展现组合泛化能力
   * **物理特性观察**：大模型在 RL 后参数更新的特征：低秩性（1% Rank）、稀疏性、避开主权重区域
   * **解释框架失效**：传统 KL 约束解释框架无法解释当前 SOTA recipe
2. 我们的框架：RL 是对基座模型在 SVD 空间中的 latent skills 进行重链接

   * Formulation 和算法
   * 验证结果：SVD 空间的参数更新保留所有推理能力
   * 模型融合：code+math 模型在 SVD 空间融合
   * 讨论：与以往研究的关系

---

## 一、争论：RL 到底教会了模型什么？

### 维度 1：Pass@k 争端

* **效率观**：有些工作认为 RLVR 仅是“概率重排”，在 k 足够大时 pass@k 可被基座模型追平。
* **反例**：QuestA、maxRL 在 k 较大时依然能显著提升 pass@k，说明 RL 确实扩张了能力边界。
* 局限性：无法让 k = ∞，无法定义“能力边界”。

### 维度 2：评价维度的转向（推理质量与连贯性）

* **从结果到过程**：单纯 pass@k 可能掩盖模型思维质量变化。
* **CoT-Pass@k 提升**：RL 后模型在 CoT-Pass@k（思维链正确性）上表现更佳，即使最终答案相同，RL 增强了推理链条的逻辑严密性、自我反思能力及整体推理质量。

### 维度 3：Toy Task 研究

* **组合泛化能力**：LLMs Learn New Skills in RL by Composing Old Ones
  如果模型具备 A(x) 和 B(x) 技能，但未在 SFT 阶段见过 B(A(x)) 二阶任务，RL 可引导模型学会这种组合，并可泛化到更高阶组合。
* **学习新游戏**：在 OOD 游戏中，RL 可将 pass@k 从 0 提升到 100%。
* **局限性**：仅限于 toy task。

---

## 二、碎片化观察：RL 更新的物理特性

### 1. RL 是低秩

* 仅用
  $$
  \Delta W = W_{rl} - W_{base}
  $$
  的 1% rank（SVD 分解后的 rank）即可恢复推理性能。

### 2. RL 是稀疏的，并避开 principal weight

* RL 的参数更新仅出现在参数矩阵的部分位置（稀疏）
* 参数更新的位置（update mask）避开 principal weight（基座模型 SVD 分解后，数值大的 top-k 子向量位置）。

### 3. 解释框架失效

* 以往 KL 散度约束被用来解释 RL 更新稀疏且遗忘更少，但许多 SOTA 实践（如 dapo、olmo3）没有使用 KL 约束，因此解释框架不适用。

---

## 三、我们的工作：RL incentivizes LLM reasoning via spectrum space rewiring

核心 insight：RL 激发 base model 已有的 reasoning 能力，RL 后模型参数应与 base model 参数处于同一几何空间。

### 伪代码：将 RL 模型投影到 base model 的 SVD 空间

1. Base model SVD 分解：
   $$
   W_0 = U \Sigma V^T = \sum_{i=1}^r \sigma_i u_i v_i^T
   $$
   定义投影算子：
   $$
   P_U = UU^T, \quad P_V = VV^T
   $$

2. 更新参数的分解：
   $$
   \Delta W = W_1 - W_0 \xrightarrow{SVD} \sum_{i=1}^r \sigma_i' u_i'(v_i')^T
   $$

3. 左右边投影：
   $$
   u_i^* = P_U u_i', \quad v_i^* = P_V v_i'
   $$

4. 投影后奇异向量重组：
   $$
   \Delta W^* = \sum_{i=1}^k \sigma_i' u_i^* (v_i^*)^T = UMV^T
   $$

5. Plug-in 到 base model：
   $$
   W_1^* = W_0 + \Delta W^* = U(\Sigma + M)V^T
   $$

结果：投影后模型拥有完整 reasoning 能力。

---

### 实验结果表：SOTA 训练 recipe

| Model size                               | AIME 24 AVG 32 | AIME 24 Pass 32 | AIME 25 AVG 32 | AIME 25 Pass 32 |
| ---------------------------------------- | -------------- | --------------- | -------------- | --------------- |
| 1.5B Base: Deepseek-distill-qwen2.5-1.5B | 31.67%         | 80.00%          | 23.96%         | 53.33%          |
| 1.5B RL: DeepscaleR                      | 40.31%         | 76.67%          | 29.58%         | 60.00%          |
| 1.5B Projected: 1 percent                | **40.21%**     | 76.67%          | **28.96%**     | 60.00%          |
| 4B Base: Qwen3 4B                        | 72.50%         | 90.00%          | 63.44%         | 86.67%          |
| 4B RL: Polaris                           | 78.75%         | 93.33%          | 75.21%         | 90.00%          |
| 4B Projected: 10 percent                 | **78.02%**     | 93.33%          | **72.21%**     | 93.33%          |
| 32B Base: Olmo3-32B-think-dpo            | 74.48%         | 93.33%          | 70.83%         | 90.00%          |
| 32B RL: Olmo-3.1-32B-Think               | 80.62%         | 93.33%          | 76.16%         | 90.00%          |
| 32B Projected: 1 percent                 | **79.27%**     | 93.33%          | **74.88%**     | 93.33%          |

---

### 参数效率表

| Model name         | Total Params | Projected Rank | Parameter Update | Rewiring Matrix (M) | Compression Ratio (M/Total) |
| ------------------ | ------------ | -------------- | ---------------- | ------------------- | --------------------------- |
| DeepscaleR         | 1.54 B       | 1%             | **16.1 M**       | 9.0 M               | ~0.58%                      |
| POLARIS            | 4.00 B       | 10%            | **552.0 M**      | 310.0 M             | ~7.75%                      |
| Olmo-3.1-32B-Think | 32.00 B      | 1%             | **311.0 M**      | 204.0 M             | ~0.64%                      |

---

### 点对点到多对一的逻辑（公式）

Base model：
$$
y_{base} = W_0 x = \sum_{i=1}^r \sigma_i (v_i^T x) u_i
$$

Projected-RL model：
$$
y_{rewired} = U(\Sigma + M)V^T x = \sum_i \left[ (\sigma_i + M_{ii})(v_i^T x) + \sum_{j \ne i} M_{ij} (v_j^T x) \right] u_i
$$

---

### 模型融合实验

将投影后的更新参数 $\Delta W^*$ 加到 coding 模型 ($W_2$) 上，可同时提升 math 和 code 能力。

#### 实验表（部分）

| Size | Model name                                | Math AIME24 Avg32 | Math Pass32 | Math Length | Code AIME25 Avg8 | Code Pass32 | Livecodebench Avg8 |
| ---- | ----------------------------------------- | ----------------- | ----------- | ----------- | ---------------- | ----------- | ------------------ |
| 1.5B | Base: deepseek-distill-qwen 1.5b          | 31.67%            | 76.67%      | 0.23        | -                | -           | 0.23               |
| 1.5B | Math RL: DeepscaleR                       | **40.31%**        | 76.67%      | 9496.82     | 29.58%           | 60.00%      | 0.272              |
| 1.5B | Code RL: Archer-code                      | 39.48%            | 76.67%      | 8126.16     | 31.19%           | 56.67%      | **0.318**          |
| 1.5B | Merge With Projection: 1 percent          | **43.44%**        | 76.67%      | 7860.47     | 31.04%           | 60.00%      | **0.3225**         |
| 1.5B | 1 percent from math to code No Projection | 36.25%            | 63.33%      | 8345.35     | 28.65%           | 53.33%      | 0.3225             |

---

## 四、讨论

1. RL 激发模型已有能力，本质是对 base model 学到的 latent skills 进行组合泛化。
2. RL 的低秩性：1% rank 就可恢复 SOTA 性能。
3. RL 的稀疏性：更新避开主权重（principal weights）。
4. RL 遗忘更少：对特征向量进行低强度重链接，减少对知识和记忆的破坏。
