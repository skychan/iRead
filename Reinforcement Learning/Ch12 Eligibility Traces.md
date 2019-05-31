# Ch12 Eligibility Traces

资格迹是 RL 中的基本机制，TD 的许多方法（Q-learning, Sarsa） 可以与资格迹进行兼容，成为更为一般的方法，说不定可以学习得更为高效。

资格迹将 TD 和 MC 方法进行了整合，使得 MC 等价于 TD(1)，单步 TD 等价于 TD(0)。

资格迹可以提供更为优雅的算法机制，其计算效率很高。

核心：短期的记忆向量，成为 eligibility trace

$$
\bm{\mathrm z}_t \in  \mathbb R^d
$$

将长期的权重进行向量化。

相比于 n-step 方法，资格迹可以连续统一地及时计算，而非需延时或者等到每个 episode 的末尾才进行计算。

此外，学习过程可以在进入某个状态后马上生效。

向前看 n-step 的方法，称之为 forward views.
然而，forward views 总是非常的复杂以至于难以应用。这是因为更新需要依赖之后的东西，在当前时刻是无法得到的。

解决的想法：使用当前的 TD 误差，使用资格迹向后看。可以作 n-step 的一些近似。（backward views）

<!-- TOC -->

- [1. The $\lambda$-return](#1-the-\lambda-return)

<!-- /TOC -->

## 1. The $\lambda$-return

