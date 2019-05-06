# Ch 6. Temporal-Difference Learning

TD learning is a combination of Monte Carlo ideas and DP ideas.

首先考察评估或者预测问题，
然后考察控制问题：找到最优策略。

强化学习中，最为关键和创新的地方：temporal-difference learning

TD Learning: 结合了 Monte Carlo 思想和 Dynmaic Programming 的思想。

特点：直接从原始经验中进行学习，而不需要动态环境的模型。

<!-- TOC -->

- [1. TD Prediction](#1-td-prediction)
- [2. Advantages of TD Prediction Methods](#2-advantages-of-td-prediction-methods)
- [3. Optimality of TD(0)](#3-optimality-of-td0)
- [4. Sarsa: On-policy TD Control](#4-sarsa-on-policy-td-control)
- [Q-learning: Off-policy TD Control](#q-learning-off-policy-td-control)
- [Expected Sarsa](#expected-sarsa)
- [Maximization Bias and Double Learning](#maximization-bias-and-double-learning)
- [Games, Afterstates, and Other Special Cases](#games-afterstates-and-other-special-cases)

<!-- /TOC -->

## 1. TD Prediction

Monte Carlo 方法：每次要等到访问的状态已知后才输出。

Constant-$\alpha$ MC:
$$
V(S_t) \gets V(S_t) + \alpha\left[G_t -V(S_t) \right]
$$

但是，Mento Carlo 方法需要等到episode 的结尾（$G_t$才能知道）才能决定增量。

然而，TD 只需要等到下一个时间增量即可。

TD(0)（单步TD）:
$$
V(S_t) \gets V(S_t) + \alpha \left[R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]
$$

$R_{t+1}, V(S_{t+1})$ 的值是立马可以获得的。

$$
\begin{aligned}
v_\pi (s) & \doteq \mathbb E_\pi \left[G_t|S_t=s \right] \\
&= \mathbb E_\pi \left[R_{t+1} + \gamma G_{t+1} | S_t = s\right] \\
& = \mathbb E_\pi \left[R_{t+1} + \gamma v_\pi(S_{t+1})\right]
\end{aligned}
$$

第一行是 MC，第三行是 DP，由于 $v_\pi(S_{t+1})$ 也是未知，因此 TD 中用 $V(S_{t+1})$ 来代替。

因此，TD 是结合了这两者的优势来进行的。

TD 误差：

$$
\delta_t \doteq R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$

与 MC 误差的关系：

$$
\begin{aligned}
G_t - V(S_t) & = R_{t+1} + \gamma G_{t+1} - V(S_t) + \gamma V(S_{t+1}) - \gamma V(S_{t+1})\\
& = \delta_t + \gamma (G_{t+1} - V(S_{t+1}))  \\
& \cdots \\
&= \sum_{k=t} ^{T-1} \gamma^{k-t}\delta_k
\end{aligned}
$$

这个只在 TD 部分更新的时候成立


## 2. Advantages of TD Prediction Methods

相比于 MC：
1. 不需要环境的模型，回报的模型，和下一状态可能性分布
2. 内在的在线特性，完全增量的样式
3. MC 必须忽略或者打折实验性质的行为选择，会大大降低学习效率，而 TD 方法受之影响较小，因为他们学习的是转移，而不管之后的行为是否被执行。

虽然理论上尚未证明，但实践中，处理随机的任务时，TD 通常会比 constant-$\alpha$ MC 收敛更快。

## 3. Optimality of TD(0)

MC 和 TD 最终都会收敛，但是收敛的值不相同。

MC 是根据真实经历过的经验数据，而 TD 更和预测相关，因此 TD 总好于 MC。

- Batch MC 方法总会找到训练集的最小化 mean-squared 误差
- Batch TD(0) 方法总会估计出模型的 马尔科夫过程最大似然。是一种绝对等价估计（certainty-equivalence estimate）

在状态空间较大的问题上，TD 方法或许是唯一的可行近似绝对等价方法。

## 4. Sarsa: On-policy TD Control

使用 state-action 对作为单位

$$
Q(S_t,A_t) \gets Q(S_t,A_t) + \alpha\left[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t,A_t)\right]
$$

仍然用当前的策略 $q_\pi$ 来找到下一个 action，然后得到 $Q(S_{t+1}, A_{t+1})$

若 $S_{t}$ 是最终状态，那么 $Q(S_t,A_t) =0$。

五元组事件组成的算法名为 Sarsa （$(S_t,A_t,R_{t+1},S_{t+1},A_{t+1})$）

Sarsa 版本的 TD 误差：

$$
\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)
$$

## Q-learning: Off-policy TD Control

早期的一项重大突破： Q-learning

$$
Q(S_t,A_t) \gets Q(S_t,A_t) + \alpha\left[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t,A_t)\right]
$$

学习到的 Q 函数，直接用来估计 $q_*$，和当前的策略无关。

Q-learning 会学习到比较好，但是更危险的策略，而 Sarsa 则相对安全。

但是如果 $\varepsilon$ 足够小的话，两者将接近。

## Expected Sarsa

$$
\begin{aligned}
Q(S_t,A_t) &\gets Q(S_t,A_t) + \alpha\left[R_{t+1} + \gamma \mathbb E_\pi[Q(S_{t+1}, A_{t+1})|S_{t+1}] - Q(S_t,A_t)\right]\\
&\gets Q(S_t,A_t) + \alpha\left[R_{t+1} + \gamma \sum_a \pi(a|S_{t+1})Q(S_{t+1},a) - Q(S_t,A_t)\right]
\end{aligned}
$$

是一种 可on 可off-policy，除了一些微小的额外计算代价，Expected Sarsa 可能是比其他 TD control 算法都要好的算法。

## Maximization Bias and Double Learning

正数偏倚：若一个值的最优是0，但是由于估计的时候有分布，因此，会有正有负，但是最大化估计会选择那些正数，导致最终结果称为正数。

办法：分成两个玩家，单独学习。

- $Q_1(a)$：学习 $A^*=\argmax_a Q_1(a)$
- $Q_2(a)$：学习期望 $Q_2(A^*) = Q_2(\argmax_a Q_1(a))$

我们甚至可以交换他们的角色。$Q_1(\argmax_a Q_2(a))$

这个称之为：Double Learning。

应用在 Double Q-learning：投一枚硬币，若正面：

$$
Q_1(S_t,A_t) \gets Q_1(S_t,A_t) +  \alpha\left[R_{t+1} + \gamma Q_2(S_{t+1},\argmax_a Q_1(S_{t+1}, a)) - Q_1(S_t,A_t)\right]
$$

若反面，则交换 1 和 2。

Double Sarsa:

$$
Q_1(S_t,A_t) \gets Q_1(S_t,A_t) +  \alpha\left[R_{t+1} + \gamma Q_2(S_{t+1},A_{t+1}) - Q_1(S_t,A_t)\right]
$$

Double Expected Sarsa (On-policy)

$$
Q_1(S_t,A_t) \gets Q_1(S_t,A_t) + \alpha\left[R_{t+1} + \gamma \sum_a \pi(a|S_{t+1})Q_2(S_{t+1},a) - Q_1(S_t,A_t)\right]
$$

## Games, Afterstates, and Other Special Cases

Afterstates，在轮流的博弈中，更为有效，可以合并一些空间。

off-policy view of Expected Sara (General Q-learning) 

von Hasselt