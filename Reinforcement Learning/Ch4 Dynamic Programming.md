# Ch 4. Dynamic Programming

动态规划，计算 MDP 的最优策略的一组方法。

回顾一下 DP 中需要用到的 Bellman 方程

$$
\begin{aligned}
v_*(s) &= \max_a\mathbb E [R_{t+1} + \gamma v_*(S_{t+1})|S_t=s,A_t=a]\\
&=\max_a \sum_{s',r} p(s',r|s,a)[r+\gamma v_*(s')]
\end{aligned}
$$

$$
\begin{aligned}
q_*(s,a) &= \mathbb E[R_{t+1} + \gamma\max_{a'} q_*(S_{t+1}, a')|S_t=s,A_t=a]\\
&=\sum_{s',r}p(s',r|s,a)[r+\gamma\max_{a'} q_*(s',a')]
\end{aligned}
$$
<!-- TOC -->

- [1. Policy Evaluation (Prediction)](#1-policy-evaluation-prediction)
- [2. Policy Improvement](#2-policy-improvement)
- [3. Value Iteration](#3-value-iteration)

<!-- /TOC -->
## 1. Policy Evaluation (Prediction)

策略评估：计算状态-值函数 $v_\pi$，我们将之称为预测问题

迭代解法（迭代策略固定）：

$$
\begin{aligned}
v_{k+1}(s) &\doteq \mathbb E [R_{t+1} + \gamma v_{k}(S_{t+1})|S_t=s,A_t=a]\\
&=\sum_{s',r} p(s',r|s,a)[r+\gamma v_{k}(s')]
\end{aligned}
$$

这是一种确切更新，因为：他们是基于下个所有状态的期望，而不是下个状态的采样。

## 2. Policy Improvement

当选择一个固定的策略 $\pi$ 之后，我们是否要改变这个策略？

一个办法是：
在该策略下，$s$ 状态中，选择行为 $a$ 并计算

$$
\begin{aligned}
q_\pi (s,a) &\doteq \mathbb E[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s, A_t=a]\\
&= \sum_{s',r} p(s',r|s,a)[r+\gamma v_\pi(s')]
\end{aligned}
$$

然后判断，这个值是否比 $v_\pi(s)$ 大，若是，那么说明当前的 $\pi$ 不是最好的，需要更新为，在状态 $s$ 下选择 $a$。

也就是说如果

 $$
 q_\pi(s,\pi'(s)) \ge v_\pi(s)
 $$

 那么策略 $\pi'$ 要不差于 $\pi$


## 3. Value Iteration

$$
\begin{aligned}
v_{k+1}(s) &\doteq \max_a\mathbb E [R_{t+1} + \gamma v_k(S_{t+1})|S_t=s,A_t=a]\\
&=\max_a \sum_{s',r} p(s',r|s,a)[r+\gamma v_k(s')]
\end{aligned}
$$

