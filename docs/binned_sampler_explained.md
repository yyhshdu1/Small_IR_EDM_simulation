# 短 IR 蒙特卡洛：快速分箱采样器原理

> 独立文档，不依赖代码库即可阅读。对应代码：`short_ir/fast_shot.py` 中的 `simulate_shot_binned`（快速版）与 `simulate_shot_streaming`（逐分子参考版）。

## 0. 背景

CeNTREX 短相互作用区（short IR）EDM 信号模拟：TlF 分子束穿过 Emma 的 COMSOL RF 场做 Ramsey 自旋翻转，经三开关（失谐 d / 校准相位 p / CP c）方案读出 CP-odd 频率 $f_{CP}$。

每一"发"（shot）要模拟 $10^7$ 个分子；一次完整运行是 8 个开关态 × $N_{\rm blocks}$ 个 block，扫描下来是上百亿条分子历史。

## 1. 要解决的问题

**逐分子做法**（主 IR 的 centrex_asymmetry 用的 `streaming` 后端）：对每个分子

1. 抽速度 $v$
2. 算翻转概率 $p(v)$
3. 抛硬币决定自旋
4. 算到达时间
5. 归到到达时间箱

代价 $O(N)$。主 IR 用 16 个进程跑 89.6 亿条历史，投影阶段要 115 秒。

**分箱采样器**把代价降到 $O(n_v \times n_t)$（速度箱数 × 到达箱数），**与分子数无关**。

## 2. 核心：联合分布可以精确因子化

单个分子的生成过程：

$$
v\sim\mathcal N(\bar v_{\rm shot},\sigma_v),\quad
t_s\sim\mathcal N(0,\sigma_{ts}),\quad
\text{spin}\sim\text{Bernoulli}\big(p(v)\big),\quad
t_{\rm arr}=t_s+\frac{L}{v}
$$

参数：$\bar v_{\rm shot}\sim\mathcal N(184,4)$ m/s（逐发平均速度），$\sigma_v=16$ m/s，$\sigma_{ts}=1$ ms，$L=5.2$ m。

三个结构性事实：

1. **$p$ 只依赖 $v$** → 给定 $v$ 时，自旋和到达时间**条件独立**
2. **起始时间 $t_s$ 与 $v$ 独立**
3. 分子之间独立同分布（i.i.d.）

因此，在（速度箱 $j$，到达箱 $k$，自旋）上的**计数**，其联合分布严格分解为：

$$
\begin{aligned}
n_j &\sim \text{Multinomial}(N,\ q_j) && \text{速度箱 } j \text{ 的分子数}\\
N_{1j} &\sim \text{Binomial}(n_j,\ \bar p_j) && \text{其中处于 state 1 的数目}\\
\{c_{1jk}\}_k &\sim \text{Multinomial}(N_{1j},\ w_{jk}) && \text{把 state 1 分到各到达箱}\\
\{c_{2jk}\}_k &\sim \text{Multinomial}(N_{2j},\ w_{jk}) && \text{state 2 独立地再抽一次}
\end{aligned}
$$

- $q_j$：速度箱 $j$ 的概率质量
- $\bar p_j$：该箱内的平均翻转概率
- $w_{jk}$：给定速度箱 $j$ 时落入到达箱 $k$ 的概率

**只要 $p$ 和 $L/v$ 在每个速度箱内恒定，这就不是近似**，得到的分布和逐分子抽样完全相同。

直观理解：把"抽 $N$ 个分子"换成"直接抽一张计数表"。

## 3. 两个受控近似

实际上 $p(v)$ 和 $L/v$ 在一个速度箱内并不完全恒定，分别这样处理：

### (a) 用箱内平均概率，不用箱中心值

$$
\bar p_j=\frac{\int_{\rm bin}p(v)\rho(v)\,dv}{\int_{\rm bin}\rho(v)\,dv}
$$

用箱中心值 $p(v_j)$ 会留下一阶残差；按速度密度加权取箱平均后，一阶项被积掉，**误差降到箱宽的二阶**。所以 200 个速度箱就够了。实现上用 9 点梯形积分。

### (b) 箱内飞行时间展宽折进到达权重

箱内速度跨度 $\Delta v$ 对应飞行时间跨度 $L\Delta v/v^2$，近似平顶分布。平顶分布的方差是 $w^2/12$，把它加进起始时间的方差：

$$
\sigma_{t,\rm eff}=\sqrt{\sigma_{ts}^2+\frac{1}{12}\Big(\frac{L\,\Delta v}{\bar v^2}\Big)^2}
$$

这个展宽是**折进去**的，没有忽略。唯一的近似是把平顶分布当成高斯。代码里有检查：一旦展宽超过 $\sigma_{ts}$ 就发警告，提示增加速度箱数。

按当前参数，$\Delta v = 0.96$ m/s 对应 147 µs 的展宽，远小于 1 ms 的起始时间抖动，卷积后的形状由起始抖动决定，这个近似的误差可以忽略。

## 4. 箱质量用误差函数精确积分

$q_j$ 和 $w_{jk}$ 都不是"箱中心的密度 × 箱宽"，而是高斯分布在每个箱上的精确积分：

$$
q_j=\tfrac12\left[\operatorname{erf}\!\Big(\tfrac{v_{j+1}-\bar v}{\sigma_v\sqrt2}\Big)-\operatorname{erf}\!\Big(\tfrac{v_j-\bar v}{\sigma_v\sqrt2}\Big)\right]
$$

所以分箱这一步本身不带来离散化误差。

## 5. 为什么 $N_1$ 和 $N_2$ 必须分别分配到到达箱

给定 $v$，两个自旋通道的到达时间服从**同一个分布**，但各自的**抽样实现相互独立**，所以要分两次独立的多项式抽样。

**常见的错误写法**：先抽总数在各到达箱的分布，再在每个箱里按 $\bar p$ 劈成两份。这样会让两个通道的到达时间涨落完全相关，**低估**非对称量 $A=(N_1-N_2)/n$ 的方差。

## 6. 性能

| 每发分子数 | 逐分子 | 分箱 | 加速比 |
|---|---|---|---|
| $2\times10^7$ | 4375 ms | 46 ms | **95×** |
| $7\times10^7$ | 约 15 s | 约 46 ms | **约 330×** |

分箱方法的耗时几乎不随分子数 $N$ 增长，多项式抽样的开销只取决于箱数。单核就能完成主 IR 需要 16 个进程的工作量。

## 7. 验证

1. **期望值**：$\mathbb E[A]$ 与逐分子结果相差 $5.8\times10^{-6}$，而散粒噪声是 $2.2\times10^{-4}$
2. **单次实现**：两种方法的结果在散粒噪声范围内一致
3. **到达时间分布**：差异约为散粒噪声的 $\sqrt2$ 倍，正好是两个独立实现之间应有的差异

## 8. 适用范围

这个方法依赖"$p$ 只依赖 $v$"。遇到**依赖单个分子的效应**就不适用了，例如：

- 横向轨迹：分子在不同径向位置感受到的场不同
- 单个分子层面的态制备涨落
- 依赖单个分子历史的探测效应

这些情况需要改回逐分子模式（`simulate_shot_streaming` 一直保留着，用作参考实现）。

**磁场漂移不影响这个方法**：它是逐发变化的，只改变 $p(v)$ 这张表本身，不破坏因子化。

## 9. 与荧光（LIF）读出的衔接

光子循环模型 `centrex_beam.photon_cycling.simulate_alternating_cycling` 的输入是**每个 5 µs 激光窗口里到达的分子数**，而不是分子列表，分箱采样器的输出正好是这个格式。实测荧光读出每发 92 ms，投影每发 60 ms。

这也是分箱方法的一个额外好处：光子循环本身就是按窗口递推的聚合过程（对每批分子里仍处于亮态的数目做二项抽样），不需要知道每个分子是谁。逐分子模拟做到这一步反而要先聚合回计数。

## 10. 代码骨架（伪代码）

```python
# --- 速度箱 ---
v_edges = linspace(v̄ - 6σ_v, v̄ + 6σ_v, n_v + 1)
q  = gauss_bin_mass(v_edges, v̄, σ_v);  q /= q.sum()     # erf 精确积分
p̄  = prob_fn.bin_average(v_edges, pdf)                  # 箱平均，不是箱中心

# --- 箱内飞行时间展宽折进 ---
Δv      = v_edges[1] - v_edges[0]
σ_t_eff = hypot(σ_ts, L*Δv / v̄**2 / sqrt(12))

# --- 计数抽样 ---
n_v  = rng.multinomial(N, q)            # 各速度箱分子数
n1_v = rng.binomial(n_v, p̄)             # 箱内自旋
n2_v = n_v - n1_v
for j in range(n_v_bins):
    w = gauss_bin_mass(t_edges, L / v_centres[j], σ_t_eff);  w /= w.sum()
    c1 = rng.multinomial(n1_v[j], w)    # 两次独立抽样
    c2 = rng.multinomial(n2_v[j], w)
    one += c1;  tot += c1 + c2
```

## 一句话总结

把"抽 $N$ 个分子"换成"抽一张计数表"。给定速度后，自旋和到达时间条件独立，所以这个替换在分布上是精确的；箱内 $p(v)$ 和飞行时间的变化分别用箱平均和方差折入处理，误差是二阶的。
