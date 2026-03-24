> 目前这个仓库中的算法研究与 `/GenSIvePFS/users/yfwang` 工作区中的 ProtLLM 主线研究完全独立。处理本目录任务时，不需要参考 ProtLLM 相关代码与文档。

# Low-Rank Spectral ES RL 实现说明

## 1. 当前实现范围

当前仓库只保留一条实际可运行的主线：

- 算法：`spectral_es`
- 推理后端：`vLLM`
- 并行方式：`single-node multi-GPU mutant-parallel`
- 更新规则：`pairwise directional`、`gaussian_mean` 或 `per_layer_cma_es`
- 约束：`standard alpha-over-sigma ES step + per-layer trust region`
- 任务：`GSM8K`、`math_data`、`MMLU-Pro`
- reward：
  - `GSM8K` / `math_data`：boxed answer exact match
  - `MMLU-Pro`：boxed multiple-choice letter exact match

训练入口是 `train.py`，当前只接受：

- `algorithm.name = spectral_es`
- `execution.backend = vllm`
- `execution.distributed_mode = mutant_parallel`

也就是说，这份文档描述的是当前代码已经接通的那条实现，而不是一个泛化的“未来可能支持”的算法族。

## 2. 记号与参数化

设目标层集合为

$$
\mathcal{L} = \{\ell_1, \ell_2, \dots, \ell_m\}.
$$

对每个目标线性层 $\ell \in \mathcal{L}$，其基座权重记为

$$
W_\ell \in \mathbb{R}^{d_{\text{out},\ell} \times d_{\text{in},\ell}}.
$$

### 2.1 截断谱子空间

对 $W_\ell$ 做 SVD：

$$
W_\ell = U_\ell \Sigma_\ell V_\ell^\top.
$$

从奇异方向中选出一个大小为 $r$ 的索引集合 $I_\ell$。当前代码支持三种选取策略：

- `top-band`
- `middle-band`
- `mixed-band`

然后定义

$$
U_{\ell,r} = U_\ell[:, I_\ell] \in \mathbb{R}^{d_{\text{out},\ell} \times r}, \qquad
V_{\ell,r}^\top = V_\ell^\top[I_\ell, :] \in \mathbb{R}^{r \times d_{\text{in},\ell}}.
$$

当前可训练状态不是直接修改全矩阵 $W_\ell$，而是在这个低维谱子空间中维护一个小矩阵

$$
M_\ell \in \mathbb{R}^{r \times r}.
$$

于是该层的增量写成

$$
\Delta W_\ell(M_\ell) = U_{\ell,r} M_\ell V_{\ell,r}^\top.
$$

整模型状态记为

$$
M = \{M_\ell\}_{\ell \in \mathcal{L}}.
$$

对应的模型权重为

$$
W_\ell^{\text{eff}} = W_\ell + \Delta W_\ell(M_\ell).
$$

### 2.2 当前实现的一个关键细节

SVD cache 中会保存

$$
U_{\ell,r}, \quad \Sigma_{\ell,r}, \quad V_{\ell,r}^\top,
$$

但当前训练参数化只使用 $U_{\ell,r}$ 和 $V_{\ell,r}^\top$，并没有把 $\Sigma_{\ell,r}$ 显式并入状态参数化。也就是说，当前实际搜索的是

$$
\Delta W_\ell = U_{\ell,r} M_\ell V_{\ell,r}^\top,
$$

而不是例如

$$
\Delta W_\ell = U_{\ell,r} \Sigma_{\ell,r}^{1/2} M_\ell \Sigma_{\ell,r}^{1/2} V_{\ell,r}^\top.
$$

因此当前方法更准确地说，是“在选定的谱左右基张成的低维子空间中搜索”，而不是“按原始奇异值度量加权的自然谱更新”。

## 3. 训练目标与 batch reward

对一个样本 $(x, y)$，模型生成文本输出记为

$$
\hat{z} \sim \pi_M(\cdot \mid x).
$$

当前 GSM8K reward 只使用 boxed 数值 exact match。把 reward 写成

$$
R(\hat{z}, y) =
\begin{cases}
1, & \text{若最终 boxed 数值答案与标准答案完全一致}, \\
0, & \text{否则}.
\end{cases}
$$

当前实现会先从输出中提取最后一个 `\box{...}` 或 `\boxed{...}`，再做数值归一化比较。

设训练 batch 为

$$
\mathcal{B} = \{(x_b, y_b)\}_{b=1}^{B}.
$$

则 batch 平均 reward 为

$$
\widehat{J}_{\mathcal{B}}(M)
= \frac{1}{B} \sum_{b=1}^B R(\hat{z}_b, y_b).
$$

由于当前解码默认是 greedy，并且 `temperature=0.0`，所以在当前主线配置下，随机性主要来自参数扰动而不是采样解码。

## 4. 当前实现的三种采样 / 更新机制

当前代码里真正接通训练主线的采样 / 更新机制有三种：

1. `pairwise_directional`
2. `gaussian_mean`
3. `per_layer_cma_es`

它们共享同一个低秩谱状态参数化，但对“如何采样 mutant、如何把 reward 变成更新方向”的处理不同。下面分开写。

### 4.1 `pairwise_directional`：antithetic pairwise 采样

设噪声状态为

$$
\varepsilon = \{\varepsilon_\ell\}_{\ell \in \mathcal{L}}, \qquad
\varepsilon_\ell \in \mathbb{R}^{r \times r}, \qquad
\varepsilon_\ell \sim \mathcal{N}(0, I).
$$

其中各层、各元素独立采样。

给定探索半径 $\sigma > 0$，定义高斯平滑目标

$$
\widetilde{J}_\sigma(M)
= \mathbb{E}_{\varepsilon}\left[\widehat{J}_{\mathcal{B}}(M + \sigma \varepsilon)\right].
$$

由高斯平滑的标准恒等式，有

$$
\nabla_M \widetilde{J}_\sigma(M)
= \frac{1}{\sigma}
\mathbb{E}_{\varepsilon}
\left[
\widehat{J}_{\mathcal{B}}(M + \sigma \varepsilon)\,\varepsilon
\right].
$$

利用高斯分布关于原点的对称性，也可以写成 antithetic 形式：

$$
\nabla_M \widetilde{J}_\sigma(M)
= \frac{1}{2\sigma}
\mathbb{E}_{\varepsilon}
\left[
\left(
\widehat{J}_{\mathcal{B}}(M + \sigma \varepsilon)
- \widehat{J}_{\mathcal{B}}(M - \sigma \varepsilon)
\right)\varepsilon
\right].
$$

### 4.1.1 有限对采样估计

设总 mutant 数为 $K$，并要求 $K$ 为偶数。定义 antithetic pair 数

$$
H = \frac{K}{2}.
$$

对每个 $i \in \{1, \dots, H\}$，采样一个噪声方向 $\varepsilon^{(i)}$，并构造一对 mutant：

$$
M^{(i,+)} = M_t + \sigma \varepsilon^{(i)}, \qquad
M^{(i,-)} = M_t - \sigma \varepsilon^{(i)}.
$$

记对应 batch 平均 reward 为

$$
r_i^+ = \widehat{J}_{\mathcal{B}}(M^{(i,+)}), \qquad
r_i^- = \widehat{J}_{\mathcal{B}}(M^{(i,-)}).
$$

那么 antithetic 差分估计器为

$$
\widehat{g}_t
= \frac{1}{2\sigma H}
\sum_{i=1}^{H}
\left(r_i^+ - r_i^-\right)\varepsilon^{(i)}.
$$

它满足

$$
\mathbb{E}[\widehat{g}_t]
= \nabla_M \widetilde{J}_\sigma(M_t).
$$

### 4.1.2 当前代码实际实现的方向量

定义每个 antithetic pair 的 pairwise utility / advantage 为

$$
A_i = \frac{r_i^+ - r_i^-}{2}.
$$

则标准 mirrored-ES 写法可以写成

$$
\widehat{g}_t
= \frac{1}{H\sigma}
\sum_{i=1}^{H}
A_i \varepsilon^{(i)}
= \frac{1}{2H\sigma}
\sum_{i=1}^{H}
\left(r_i^+ - r_i^-\right)\varepsilon^{(i)}.
$$

当前代码在 `es/spectral_update.py` 中先计算方向缓存

$$
\widehat{d}_t
= \frac{1}{H}
\sum_{i=1}^{H}
A_i \varepsilon^{(i)}.
$$

因此有

$$
\mathbb{E}[\widehat{d}_t]
= \sigma \nabla_M \widetilde{J}_\sigma(M_t),
$$

而最终更新会显式再除以 $\sigma$。因此：

- `sigma` 主要决定探索半径与 reward 差分的信号质量
- `alpha` 决定真正落地到状态上的步长尺度

`pairwise_directional` 的特点是：

- 每一步要求 `K` 为偶数；
- 每个方向都成对出现，`(+epsilon, -epsilon)`；
- 更新只看 pair 内 reward 差分，因此通常比普通 Gaussian ES 方差更小；
- 当前实现里它与 `es.antithetic=true` 绑定。

### 4.2 `gaussian_mean`：标准 Gaussian ES 采样

如果不做 antithetic 成对构造，而是直接采样

$$
\varepsilon^{(i)} \sim \mathcal{N}(0, I), \qquad i = 1, \dots, K,
$$

并令

$$
M^{(i)} = M_t + \sigma \varepsilon^{(i)}, \qquad
r_i = \widehat{J}_{\mathcal{B}}(M^{(i)}),
$$

那么标准 Gaussian ES 的估计器写成

$$
\widehat{g}_t^{\text{gauss}}
= \frac{1}{K\sigma}\sum_{i=1}^{K} r_i \varepsilon^{(i)}.
$$

当前代码在 `es/spectral_update.py` 中对应的缓存方向是

$$
\widehat{d}_t^{\text{gauss}}
= \frac{1}{K}\sum_{i=1}^{K} r_i \varepsilon^{(i)},
$$

然后同样在真正写回参数时再乘上 `alpha / sigma`。

`gaussian_mean` 的特点是：

- 不要求 antithetic pair；
- 实现最直接，和经典 NES / Gaussian ES 写法最接近；
- 在 reward 很稀疏或很 noisy 时，方向估计方差通常比 `pairwise_directional` 更大；
- 但它不依赖“前半 / 后半必须成对”的结构，因此在一些自定义采样策略下更灵活。

### 4.3 `per_layer_cma_es`：每层独立维护协方差的 CMA-ES

第三条路径不再把所有 layer 都视为共享同一个各向同性高斯，而是对每个目标层单独维护一个搜索分布。

对任意目标层 $\ell$，先把状态矩阵展平：

$$
x_\ell = \mathrm{vec}(M_\ell) \in \mathbb{R}^{r^2}.
$$

当前实现会为每个 layer 单独维护

$$
\sigma_\ell, \qquad
C_\ell \in \mathbb{R}^{r^2 \times r^2}, \qquad
L_\ell L_\ell^\top = C_\ell,
$$

其中 $L_\ell$ 是协方差的 Cholesky 因子。

采样时，先取标准正态方向

$$
z_\ell^{(i)} \sim \mathcal{N}(0, I),
$$

再做线性变换

$$
y_\ell^{(i)} = L_\ell z_\ell^{(i)},
$$

最后构造 mutant：

$$
x_\ell^{(i)} = x_{\ell,t} + \sigma_\ell y_\ell^{(i)}.
$$

如果配置 `es.antithetic=true`，当前实现仍会先采半数 $z$，再拼接成 `(+z, -z)`，于是 layer 内的 CMA 采样也保持 antithetic 对称。

拿到所有 mutant reward 后，代码会对 reward 排序，选取前 $\mu$ 个样本，用标准 CMA-ES 的重组权重构造每层加权方向：

$$
y_{\ell,w} = \sum_{j=1}^{\mu} w_j y_{\ell,(j)}, \qquad
z_{\ell,w} = \sum_{j=1}^{\mu} w_j z_{\ell,(j)}.
$$

然后：

- 用 $\sigma_\ell y_{\ell,w}$ 形成该层的均值步长；
- 更新每层的 evolution paths `p_sigma` 与 `p_c`；
- 用 rank-one + rank-$\mu$ 的形式更新 $C_\ell$；
- 重新分解协方差，并自适应更新每层的 $\sigma_\ell$。

因此 `per_layer_cma_es` 的本质是：

- 不再固定使用各向同性高斯；
- 每个 layer 自己学习“该往哪些方向探索、探索半径该变大还是变小”；
- 代价是每层都要维护一个 `r^2 x r^2` 协方差矩阵，时间和显存开销都明显高于前两条标准 ES 路径。

## 5. 更新规则与 trust region

### 5.1 标准 ES 更新

设标准 ES 步长超参数为 $\alpha > 0$，当前代码记作 `es.alpha.m`。令

$$
N = H = \frac{K}{2}
$$

表示 antithetic pair 数，则更新写成

$$
M_{t+1} = M_t + \frac{\alpha}{N\sigma}\sum_{i=1}^{N} A_i \varepsilon^{(i)}.
$$

等价地，如果先把方向缓存记为

$$
\widehat{d}_t = \frac{1}{N}\sum_{i=1}^{N} A_i \varepsilon^{(i)},
$$

则原始更新直接写成

$$
\Delta M_t^{\text{raw}}
= \frac{\alpha}{\sigma} \widehat{d}_t.
$$

于是每一层的原始步长为

$$
\Delta M_{t,\ell}^{\text{raw}}
= \frac{\alpha}{\sigma} \widehat{d}_{t,\ell}.
$$

如果把所有层的方向矩阵视为一个整体状态，其全局范数

$$
\|\widehat{d}_t\|_{\text{global}}
= \left(
\sum_{\ell \in \mathcal{L}}
\|\widehat{d}_{t,\ell}\|_F^2
\right)^{1/2}
$$

在当前实现里只作为诊断量记录，不再参与步长归一化。

### 5.2 每层步长裁剪

设每层单步最大范数为 $\tau_{\text{step}} > 0$，当前配置名为

$$
\tau_{\text{step}} = \texttt{es.trust\_region.max\_layer\_step\_norm.m}.
$$

则对每个目标层，执行

$$
\Delta M_{t,\ell}
=
\begin{cases}
\Delta M_{t,\ell}^{\text{raw}}, &
\|\Delta M_{t,\ell}^{\text{raw}}\|_F \le \tau_{\text{step}}, \\
\dfrac{\tau_{\text{step}}}{\|\Delta M_{t,\ell}^{\text{raw}}\|_F}\,
\Delta M_{t,\ell}^{\text{raw}}, &
\|\Delta M_{t,\ell}^{\text{raw}}\|_F > \tau_{\text{step}}.
\end{cases}
$$

### 5.3 每层状态范数裁剪

先做加法更新：

$$
M_{t+1,\ell}^{\text{preclip}} = M_{t,\ell} + \Delta M_{t,\ell}.
$$

再施加状态上界 $\tau_{\text{state}} > 0$，当前配置名为

$$
\tau_{\text{state}} = \texttt{es.trust\_region.max\_state\_norm.m}.
$$

当前代码的实际行为是“对每个 layer 的 `m_state` 分别裁剪”，因此

$$
M_{t+1,\ell}
=
\begin{cases}
M_{t+1,\ell}^{\text{preclip}}, &
\|M_{t+1,\ell}^{\text{preclip}}\|_F \le \tau_{\text{state}}, \\
\dfrac{\tau_{\text{state}}}{\|M_{t+1,\ell}^{\text{preclip}}\|_F}\,
M_{t+1,\ell}^{\text{preclip}}, &
\|M_{t+1,\ell}^{\text{preclip}}\|_F > \tau_{\text{state}}.
\end{cases}
$$

注意：尽管配置名叫 `max_state_norm`，但当前实现不是对整个拼接后的全局状态做一次总裁剪，而是对每个目标层分别裁剪。

## 6. 从谱状态到 vLLM LoRA adapter

### 6.1 为什么需要导出成 LoRA

当前训练后端是 vLLM。vLLM 对“冻结 base model + 动态挂载多个 LoRA adapter”的支持非常成熟，因此当前实现没有直接在 vLLM 内部改写基础权重，而是把每个 mutant 的谱增量

$$
\Delta W_\ell = U_{\ell,r} M_\ell V_{\ell,r}^\top
$$

导出为一个等价的 LoRA 形式

$$
\Delta W_\ell = B_\ell A_\ell,
$$

其中

$$
A_\ell \in \mathbb{R}^{r \times d_{\text{in},\ell}}, \qquad
B_\ell \in \mathbb{R}^{d_{\text{out},\ell} \times r}.
$$

### 6.2 严格构造

先对谱状态矩阵 $M_\ell$ 做 SVD：

$$
M_\ell = \widetilde{U}_\ell \widetilde{\Sigma}_\ell \widetilde{V}_\ell^\top.
$$

定义

$$
L_\ell = \widetilde{U}_\ell \widetilde{\Sigma}_\ell^{1/2}, \qquad
R_\ell = \widetilde{\Sigma}_\ell^{1/2}\widetilde{V}_\ell^\top.
$$

于是

$$
M_\ell = L_\ell R_\ell.
$$

再令

$$
B_\ell = U_{\ell,r} L_\ell, \qquad
A_\ell = R_\ell V_{\ell,r}^\top.
$$

则有

$$
B_\ell A_\ell
= U_{\ell,r} L_\ell R_\ell V_{\ell,r}^\top
= U_{\ell,r} M_\ell V_{\ell,r}^\top
= \Delta W_\ell.
$$

因此导出的 LoRA adapter 与当前谱增量严格等价。仓库中的 `tests/test_spectral_vllm_export.py` 就是在验证

$$
B_\ell A_\ell = U_{\ell,r} M_\ell V_{\ell,r}^\top.
$$

### 6.3 当前实现流程

当前训练器中的实现顺序是：

1. 在 CPU 上加载冻结 base model。
2. 选择 `layers.target_blocks` 和 `layers.target_modules` 指定的线性层。
3. 为这些层创建或读取 SVD cache。
4. 在 CPU 内存中维护每层的小状态矩阵 `m_state`。
5. 对当前中心状态或某个 mutant 状态，导出一份 LoRA adapter 目录。
6. 用 `LoRARequest` 把该 adapter 动态挂载到 vLLM 推理引擎上。

这条路径的关键点是：

- 大模型主权重始终冻结。
- ES 更新只作用在一组很小的 $r \times r$ 状态矩阵上。
- vLLM 只负责高吞吐 forward/generate。

## 7. 基于 vLLM 的高效并行训练是怎么做的

### 7.1 并行粒度：按 mutant 分片，而不是按模型切分

当前实现不是 tensor parallel，也不是 pipeline parallel。每个 rank 都启动一个独立的 vLLM 引擎，并设置：

$$
\texttt{tensor\_parallel\_size} = 1.
$$

多卡扩展靠的是 mutant-parallel：

- 总 mutant 数为 $K$
- world size 为 $P$
- 每个 rank 固定负责

$$
K_{\text{local}} = \frac{K}{P}
$$

个 mutant

当前代码要求严格满足

$$
K = P \cdot \texttt{mutants\_per\_worker}.
$$

因此 rank $p$ 负责一个连续 shard：

$$
\mathcal{S}_p = \{pK_{\text{local}}, \dots, (p+1)K_{\text{local}} - 1\}.
$$

这样每个 GPU 只做自己那一段 mutant 的 rollout，天然避免重复计算。

### 7.2 单 rank 内部：mutant chunk 和 question micro-batch 两级 batching

单个 rank 上还会做两层打包：

#### 第一层：按 mutant chunk

把本地 shard 再切成大小为 $C$ 的 chunk：

$$
C = \texttt{execution.mutant\_chunk\_size}.
$$

对一个 chunk 内的每个 mutant，先各自导出一份 LoRA adapter，并创建对应的 `LoRARequest`。

#### 第二层：按 question micro-batch

把训练样本 batch 再切成大小为 $B_q$ 的 question micro-batch：

$$
B_q = \texttt{train.micro\_batch}
$$

训练时，或者

$$
B_q = \texttt{eval.micro\_batch}
$$

评估时。

对于一个 `(mutant chunk, question micro-batch)` 组合，代码会把请求展平成

$$
C \times B_q
$$

条 prompt，并在一次 `llm.generate(...)` 调用中同时送入 vLLM。每条 prompt 都带上对应 mutant 的 `LoRARequest`。

这意味着单次 vLLM 调用不是“一个模型打一批问题”，而是“多个 LoRA mutant 各打一批问题”。

### 7.3 为什么这条路径高效

#### 1. 不做反向传播

整个训练过程只有：

- 采样 mutant
- forward/generate
- reward 聚合
- 低维状态更新

没有：

- 反向传播
- optimizer state
- gradient all-reduce

因此显存主开销来自：

- base model
- KV cache
- 当前活跃的 LoRA slot

而不是训练态梯度与优化器状态。

#### 2. 通信量与参数维度解耦

每一步跨 rank 通信的主要对象只有：

- local reward 向量
- local exact-match 向量
- profiler / GPU 监控统计
- 从主进程广播的噪声 payload
- 从主进程广播的 step payload

其中真正和“参数维度”相关的，是每层一个 $r \times r$ 小矩阵，而不是全模型梯度。因此每步通信复杂度更接近

$$
\mathcal{O}(|\mathcal{L}|\,r^2 + K),
$$

而不是

$$
\mathcal{O}(\#\text{full-model-params}).
$$

并且这些 payload 当前都是低维状态张量，不需要同步 base model 权重。

#### 3. SVD cache 避免重复预处理

层选择和 SVD 只需要做一次，之后直接从 cache 读取：

- `subspace.rank`
- `subspace.band_strategy`
- 目标层集合

共同决定 cache key。

这使得 sweep 不需要反复对 base model 做全量 SVD。

#### 4. LoRA slot 复用 vLLM 的现成高吞吐路径

当前实现显式设置：

- `enable_lora=True`
- `max_loras`
- `max_cpu_loras`

这样一个 rank 可以在一次或少量 generate 调用中高效调度多个 mutant adapter，而不用为每个 mutant 单独重启模型。

### 7.4 一次训练 step 的实际执行顺序

设当前 step 为 $t$，其执行顺序可以写成：

1. 主进程采样一个训练 batch $\mathcal{B}_t$，并广播给所有 rank。
2. 主进程采样 antithetic 噪声 $\{\varepsilon^{(i)}\}_{i=1}^{H}$，并广播给所有 rank。
3. 每个 rank 在本地激活自己的 mutant shard。
4. 每个 rank 用 vLLM 对本地 shard 做 rollout，得到本地 reward。
5. 所有 rank `all_gather_object` 回主进程。
6. 主进程拼接出全局 reward 向量 $\{r_i^+\}_{i=1}^{H} \cup \{r_i^-\}_{i=1}^{H}$。
7. 主进程计算方向 $\widehat{d}_t$、步长裁剪和状态裁剪。
8. 主进程把更新后的 step payload 广播给所有 rank。
9. 所有 rank 同步应用该 step，得到新的中心状态 $M_{t+1}$。
10. 按 `eval.eval_every_steps` 做验证，并保存 `last` / `best` checkpoint。

这里的关键是：全局协调只发生在“reward 聚合”和“低维状态同步”两个阶段，而昂贵的 rollout 是天然并行的。

## 8. 当前可调超参数

下面只列“当前代码真正会读取并生效”的配置项。为了更贴近实验使用场景，这里不只是罗列字段名，而是按“这组参数控制什么”来说明。

### 8.1 子空间定义：决定我们在什么低维空间里搜索

这一组参数决定可训练空间的表达能力。它们回答的是：哪些层参与训练、每层允许用多少个谱方向来表示更新。

| 配置项 | 作用 | 当前默认值 | 调参时通常看什么 |
| --- | --- | --- | --- |
| `layers.target_blocks` | 指定哪些 transformer block 参与训练 | 全部 `28` 个 block (`0..27`) | block 越多，可调参数越多，表达能力更强，但 rollout 开销也更大 |
| `layers.target_modules` | 指定每个 block 里哪些线性层参与训练 | 全部目标线性层 `[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]` | 从 attention-only 扩到 MLP 后，通常容量更强，但也更容易过拟合或更耗显存 |
| `subspace.rank` | 每层谱子空间 rank，即每层状态矩阵 $M_\ell \in \mathbb{R}^{r\times r}$ 的边长 | `32` | `rank` 越大，搜索空间越大；但 LoRA rank、通信量和状态更新量也会随之增加 |
| `subspace.band_strategy` | 从奇异方向中抽取哪一段来构造谱子空间 | `top-band` | 当前主线默认用前几大奇异方向；如果后续扩展策略，这里决定子空间偏向哪类方向 |
| `subspace.cache_dir` | SVD cache 保存目录 | `artifacts/svd_cache` | 主要影响复现实验和 sweep 时的复用效率，不改变算法本身 |

### 8.2 ES 更新：决定每一步探索多远、更新多大

当前实现支持三类更新规则：

- `pairwise_directional`：标准 mirrored ES，用 antithetic pair 的 reward 差分做方向估计；
- `gaussian_mean`：标准 Gaussian ES，用全部 mutant reward 的均值加权噪声方向；
- `per_layer_cma_es`：对每个 layer 的 `vec(M_l)` 分别维护一个高斯协方差，并按 layer 独立做 CMA-ES 风格的均值 / 协方差 / `sigma` 自适应。

其中前两者遵循标准 ES 更新式

$$
w \leftarrow w + \frac{\alpha}{N\sigma}\sum_{i=1}^{N} A_i\,\varepsilon_i.
$$

因此这一组参数里：

- `sigma` 控制扰动半径，也就是每个 mutant 离当前中心点有多远；
- `alpha` 控制最终写回中心状态时的更新幅度；
- `num_mutants` 控制每一步用多少个样本来估计方向；
- trust region 是可选稳定化项，默认关闭；只有显式配置时才会限制单步更新和累计状态。

| 配置项 | 作用 | 当前默认值 | 调参时通常看什么 |
| --- | --- | --- | --- |
| `es.num_mutants` | 每步总 mutant 数，必须为偶数 | `64` | 越大，方向估计方差越小，但每步 rollout 成本越高 |
| `es.update_rule` | 选择 `pairwise_directional`、`gaussian_mean` 或 `per_layer_cma_es` | `pairwise_directional` | 前两者更轻，`per_layer_cma_es` 会额外学习每层搜索分布 |
| `es.sigma.m` | 扰动半径 $\sigma$ | `0.01` | 太小则 reward 差分信号弱，太大则局部线性近似变差 |
| `es.alpha.m` | 标准 ES 更新中的步长系数 $\alpha$ | `0.005` | 直接控制更新强度；如果 loss/reward 波动很大，通常优先先降它 |
| `es.trust_region.max_layer_step_norm.m` | 每层单步最大 Frobenius 范数 | 默认关闭 | 显式配置后可防止单次更新把某一层推得过猛 |
| `es.trust_region.max_state_norm.m` | 每层状态最大 Frobenius 范数 | 默认关闭 | 显式配置后可限制累计状态半径，避免长期漂移到过远区域 |

当 `es.update_rule=per_layer_cma_es` 时，`es.alpha.m` 不再决定均值更新，真正起作用的是 `es.sigma.m` 和 `es.cma.*`：

| 配置项 | 作用 | 当前默认值 | 调参时通常看什么 |
| --- | --- | --- | --- |
| `es.cma.selection_ratio` | 每层 CMA-ES 重组时选取前多少比例的 mutant | `0.5` | 越小越偏 exploitation，越大越平滑 |
| `es.cma.mean_step_scale` | 均值方向写回 `M_l` 时的额外缩放 | `1.0` | 可视为 CMA 均值步长的直接控制杆 |
| `es.cma.min_sigma` | 每层 `sigma_l` 的最小值 | `1e-6` | 防止探索半径塌缩到 0 |
| `es.cma.max_sigma` | 每层 `sigma_l` 的最大值 | `0.1` | 防止搜索分布膨胀过快 |
| `es.cma.min_eigenvalue` | 每层协方差最小特征值下界 | `1e-8` | 用于数值稳定和 SPD 修复 |
| `es.cma.jitter` | 每层协方差分解前的 jitter | `1e-10` | Cholesky / Eigh 数值稳定化 |

实践上，标准 ES 更看重 `es.alpha.m` 和 `es.sigma.m`；而 `per_layer_cma_es` 更看重 `es.sigma.m`、`es.cma.selection_ratio` 和 `es.cma.mean_step_scale`。

### 8.3 训练 batch：决定每一步看多少题、每次 rollout 切多大

这一组参数控制训练吞吐与 reward 估计噪声。

| 配置项 | 作用 | 当前默认值 | 调参时通常看什么 |
| --- | --- | --- | --- |
| `train.train_steps` | 总训练步数 | `500` | 控制总优化时长 |
| `train.effective_question_batch` | 每步抽取的问题数 | `32` | 越大，reward 均值更稳定，但每步计算更贵 |
| `train.micro_batch` | 训练时每次 rollout 的 question micro-batch 大小 | `32` | 主要是系统吞吐参数；增大它通常更高效，但会更吃显存和 KV cache |

这里要区分两种“batch”：

- `effective_question_batch` 是算法意义上的训练 batch，大了会降低 reward 方差；
- `micro_batch` 是执行意义上的切分粒度，主要影响单次 generate 的装载效率和显存压力。

### 8.4 评估：决定多久评一次、一次评多少

| 配置项 | 作用 | 当前默认值 | 调参时通常看什么 |
| --- | --- | --- | --- |
| `eval.micro_batch` | 验证/测试时的 question micro-batch | `187` | 主要看在不 OOM 的前提下把评估吞吐打满 |
| `eval.eval_every_steps` | 每多少步做一次验证 | `5` | 更频繁能更快看到趋势，但会增加训练中断开销 |
| `eval.skip_initial_validation` | 是否跳过 step 0 验证 | `true` | 主要影响日志习惯，不影响训练算法 |

### 8.5 并行执行：决定 mutant 如何在多卡上切分

这一组参数控制 mutant-parallel 的并行拓扑，而不是算法更新公式本身。

| 配置项 | 作用 | 当前默认值 | 调参时通常看什么 |
| --- | --- | --- | --- |
| `execution.world_size` | 总 rank 数，通常等于总 GPU 数 | 自动读取 `WORLD_SIZE` | 一般不手填，由启动器按实际环境提供 |
| `execution.gpus_per_node` | 单节点 GPU 数 | 自动读取 `LOCAL_WORLD_SIZE` | 主要用于分布式启动和日志记录 |
| `execution.mutants_per_worker` | 每个 rank 负责的 mutant 数 | 默认自动推导为 `es.num_mutants / execution.world_size` | 如果手动覆盖，必须与总 mutant 数严格匹配 |
| `execution.mutant_chunk_size` | 单 rank 内一次同时激活的 LoRA chunk 大小 | `16` | 更大通常吞吐更高，但会占用更多 LoRA slot 和显存 |

这些量必须满足

$$
\texttt{es.num\_mutants}
=
\texttt{execution.world\_size}
\times
\texttt{execution.mutants\_per\_worker}.
$$

可以把这组参数理解为“两层切分”：

- 第一层是跨 rank 切分，由 `mutants_per_worker` 决定每张卡负责多少个 mutant；
- 第二层是 rank 内分 chunk 执行，由 `mutant_chunk_size` 决定一次塞进 vLLM 多少个 LoRA。

### 8.6 vLLM 系统参数：决定 rollout 吞吐和显存占用

这一组参数主要服务于高效推理，不直接改变 ES 公式，但会显著影响训练速度、OOM 风险和并发能力。

| 配置项 | 作用 | 当前默认值 | 调参时通常看什么 |
| --- | --- | --- | --- |
| `model.model_path` | base model 路径 | `/GenSIvePFS/users/model/Qwen/Qwen3-0.6B-Base` | 决定底座模型 |
| `model.dtype` | vLLM 推理 dtype | `bfloat16` | 影响推理速度与显存占用 |
| `model.svd_dtype` | 计算 SVD 时的 dtype | `float32` | 主要影响预处理数值稳定性 |
| `generation.max_new_tokens` | 每次生成最大长度 | `1024` | 越大越慢、KV cache 越大，但能容纳更长推理链 |
| `generation.temperature` | 采样温度 | `0.0` | 当前主线默认 greedy，尽量把随机性集中在参数扰动上 |
| `vllm.max_loras` | GPU 上同时活跃的 LoRA 数 | 默认等于 `execution.mutant_chunk_size` | 决定一次最多并发多少个 mutant adapter |
| `vllm.max_cpu_loras` | CPU 侧缓存的 LoRA 数 | `16` | 影响 adapter 复用和换入换出频率 |
| `vllm.gpu_memory_utilization` | vLLM 显存占用上限 | `0.85` | 调高可提升吞吐，调太高更容易 OOM |
| `vllm.max_model_len` | vLLM 最大上下文长度 | `4096` | 要覆盖 prompt 长度加生成长度 |
| `vllm.enforce_eager` | 是否强制 eager 模式 | `true` | 与当前实现路径兼容性有关，通常不建议随意改 |

如果训练慢但没 OOM，通常优先看的不是 ES 参数，而是 `train.micro_batch`、`execution.mutant_chunk_size`、`vllm.max_loras` 和 `vllm.gpu_memory_utilization` 这四个吞吐参数。

### 8.7 数据与 reward：决定训练信号来自哪里

| 配置项 | 作用 | 当前默认值 | 调参时通常看什么 |
| --- | --- | --- | --- |
| `seed` | 全局随机种子 | `42` | 影响可复现性 |
| `data.cache_dir` | 原始 GSM8K cache 路径 | `/GenSIvePFS/users/yfwang/data/gsm8k/hf_main_full` | 数据来源位置 |
| `data.processed_dir` | 处理后数据集路径 | `artifacts/gsm8k_processed` | 影响数据复用 |
| `data.split_seed` | train/val 划分随机种子 | `42` | 影响验证集切分 |
| `data.val_size` | validation 集大小 | `748` | 验证更稳定，但也更慢 |
| `data.train_max_examples` | 训练样本裁剪上限，`0` 表示全量 | `0` | 适合快速 smoke test |
| `data.val_max_examples` | 验证样本裁剪上限，`0` 表示全量 | `0` | 可用于快速比较不同超参数 |
| `data.test_max_examples` | 测试样本裁剪上限，`0` 表示全量 | `0` | 同上 |
| `reward.exact_match` | exact match reward 权重 | `1.0` | 当前 reward 基本就是 GSM8K 的答案是否完全匹配 |

### 8.8 输出与日志：决定实验如何落盘和追踪

| 配置项 | 作用 | 当前默认值 | 调参时通常看什么 |
| --- | --- | --- | --- |
| `output.root_dir` | 训练输出根目录 | `output/lowrank_spectral_es_rl` | 结果保存位置 |
| `output.run_id` | 本次实验 ID | `spectral_es_vllm_mutant_parallel_allblocks_allmodules_r32_k64_q32_mb32` | 建议编码进关键超参数，便于比较 |
| `wandb.enabled` | 是否启用 wandb | `true` | 是否做在线实验追踪 |
| `wandb.project` | wandb project | `lowrank-spectral-es-rl` | 实验归档空间 |
| `wandb.entity` | wandb entity | `null` | 团队或个人空间 |
| `wandb.group` | wandb group | `spectral_es_vllm_mutant_parallel` | 适合同一组 sweep |
| `wandb.job_type` | wandb job type | `train` | 日志分类 |
| `wandb.name` | wandb run name | `null` | 可手动覆盖更清晰的实验名 |
| `wandb.dir` | wandb 本地目录 | `output/wandb` | 本地缓存位置 |
| `wandb.mode` | `online/offline` | `online` | 集群无外网时可切 `offline` |
| `wandb.tags` | run tags | `[gsm8k,qwen3_0p6b,spectral_es,vllm,mutant_parallel]` | 用于筛选实验 |
| `baseline.output_dir` | baseline eval 输出目录 | `output/gsm8k_eval` | baseline 结果保存位置 |

## 9. 当前存在但没有真正接入训练逻辑的字段

下面这些字段虽然出现在 YAML 里，但当前主训练逻辑并不会根据它们切分分支：

| 配置项 | 当前状态 |
| --- | --- |
| `prompt.template_name` | 目前未接入；实际 prompt 模板由 `data.source` 在 `data/gsm8k.py` 中选择 |
| `prompt.require_box_answer` | 目前未接入；boxed 格式要求同样在 `data/gsm8k.py` 中硬编码 |
| `es.antithetic` | 控制是否使用 antithetic 噪声采样 |

同理，下面这些字段虽然会被读取，但当前只能取固定值，否则训练入口会直接报错：

| 配置项 | 当前可接受值 |
| --- | --- |
| `algorithm.name` | `spectral_es` |
| `execution.backend` | `vllm` |
| `execution.distributed_mode` | `mutant_parallel` |

## 10. 实验上最值得优先调的参数

如果只看当前主线，最值得优先 sweep 的是：

1. `subspace.rank`
2. `subspace.band_strategy`
3. `layers.target_blocks`
4. `layers.target_modules`
5. `es.num_mutants`
6. `es.sigma.m`
7. `es.alpha.m`
8. `es.trust_region.max_layer_step_norm.m`
9. `es.trust_region.max_state_norm.m`
10. `train.effective_question_batch`
11. `train.micro_batch`
12. `execution.mutant_chunk_size`
13. `vllm.max_loras`
14. `vllm.gpu_memory_utilization`
15. `generation.max_new_tokens`

其中：

- `rank / band_strategy / target layers` 决定“搜索空间长什么样”
- `num_mutants / sigma / alpha` 决定“探索和更新有多强”
- `micro_batch / mutant_chunk_size / max_loras` 决定“单机吞吐和显存占用”

这三类参数分别对应算法表达能力、优化稳定性和系统效率，是当前最核心的调参轴。
