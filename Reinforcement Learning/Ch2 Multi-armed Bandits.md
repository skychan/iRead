# Ch 2. Multi-armed Bandits

强化学习的一个非常重要的特征是，它使用训练信息，评估计算行为，而不是通过给定正确的行为来指导。

Nonassociative setting: one that does not involve learning to act in more than one situation.

<!-- TOC -->

- [1. A $k$-armed Bandit Problem](#1-a-k-armed-bandit-problem)
- [2. Action-value Methods](#2-action-value-methods)
- [3. The 10-armed Testbed](#3-the-10-armed-testbed)
- [4. Incremental Implementation](#4-incremental-implementation)
- [5. Tracking a Nonstationary Problem](#5-tracking-a-nonstationary-problem)
- [6. Optimistic Initial Values](#6-optimistic-initial-values)
- [7. Upper-Confidence-Bound(UCB) Action Selection](#7-upper-confidence-bounducb-action-selection)
- [8. Gradient Bandit Algorithm](#8-gradient-bandit-algorithm)
- [9. Associative Search (Contextual Bandits)](#9-associative-search-contextual-bandits)

<!-- /TOC -->
## 1. A $k$-armed Bandit Problem

重复地在 $k$ 个不同的选项或者行为中进行选择，每次选择后，从一个固定的概率分布中收获一个数字的奖励。

你的目标是，最大化一定时段内所有收益和的期望。

We deonte action selected on time step $t$ as $A_t$， and the corresponding reward as $R\_{t}$, the value of an arbitrary action $a$ denoted $q\_*(a)$ as 

$$
q_*(a) \doteq \mathbb{E}[R_t|A_t = a]
$$

我们假设，你不知道每一个行为的确切价值，但你可能有个估计，记为　$Q_t(a)$。

- Exploiting: 当贪心地根据当前知识选择最大值的行为。可能在当前的１步中可以最大化
- Exploring: 选择非贪心的行为，可能在长期的收益总和中获得最大值

在单次的选择中，不可能同时进行explore和exploit。
因此，平衡这两个操作是非常关键的。

## 2. Action-value Methods

A natural way to estimate the value is by averaging the rewards actually received:

$$
Q_t(a) \doteq \frac{\sum_{i=1}^{t-1}R_i\cdot\mathbb{1}_{A_i=a}}{\sum_{i=1}^{t-1}\mathbb{1}_{A_i=a}}
$$

采样平均（sample-average）估算方法：随着上式分母增大，根据大数定理，$Q_t(a) \to q_*(a)$ 。

采用采样平均方法来进行行为选择：最简单的是greedy方法，选择有着最大期望值的行为：

$$
A_t \doteq \arg_a\max Q_t(a)
$$

贪婪行为选择总是会用尽当前的知识以最大化立得收益。

$\epsilon$-greedy 方法：大多时间选择 greedy 方法，以 $\epsilon$ 的概率选择探索。

## 3. The 10-armed Testbed

可以看到，greedy 方法比其他的方法要提升地略快一些，在最一开始的时候，但很快会停止更新。会陷入次优的行为。

$\epsilon$ 较小的方法会最终得到比较好的效果。

然而，$\epsilon$-greedy 的优势要根据任务而定。

收益的变动越大，使用 $\epsilon$-greedy 的效果越好。反之，使用 greedy 效果好。

Reinforcement learning requires a balance between exploration and exploitation.

## 4. Incremental Implementation

Agent 怎样才能高效计算观测到的收益的平均值。

Let $Q_n$ denote the estimate of its action value after it has been selected $n-1$ times, which we can now write simply as:

$$
Q_n \doteq \frac{R_1+R_2+\cdots R_{n-1}}{n-1}
$$

可以推导：

$$
Q_{n+1} = Q_n + \frac 1 n [R_n - Q_n]
$$

更一般地，增量法的思想贯穿全书：

$$
NewEstimate \gets OldEstimate + StepSize [Target - OldEstimate]
$$

## 5. Tracking a Nonstationary Problem

平均方法对稳定bandit问题是比较适合的，但是对于非稳定的问题，如果给近期的收益更大一点的权重比较合理。

$$
Q_{n+1} \doteq Q_n + \alpha[R_n-Q_n]
$$

分解可得：

$$
Q_{n+1} = (1-\alpha)^nQ_1+\sum_{i=1}^n\alpha(1-\alpha)^{n-i}R_i
$$

这个式子也称为指数较新加权平均（exponential recency-weighted average）。
如果　$\alpha=1$ 那么，这个权重会给最后一个收益。

可以证明：

$$
(1-\alpha)^n + \sum_{i=1}^n \alpha(1-\alpha)^{n-i} = 1
$$
变动的步长参数需要满足：

$$
\sum_{n=1}^\infty \alpha_n(a) = \infty \text{ and } \sum_{n=1}^\infty \alpha_n^2(a) < \infty 
$$

前一个是为了克服初值条件或者随机扰动，后者是为了确保收敛。

## 6. Optimistic Initial Values

目前所讨论的方法都再某些程度上依赖初值估计，也就是统计学上说的*初始估计偏倚*。

对于采样均值方法，若所有的行为都至少被选过依次，那么这个偏倚会消失。但是对于固定的步长参数$\alpha$，这个偏倚将是永久存在的。

可以将之视为先验知识。同样，初值的设置也可以鼓励　exploration。

然而，任何专注于初始情况的方法对于更为一般的非稳定问题都没有多大帮助。

**一种解决常步长参数偏倚的方法**：

令步长为

$$
\beta_n \doteq \frac \alpha {\bar o_n}
$$

其中，$\bar o_n$ 是初始为$0$的迹：

$$
\bar o_n = \begin{cases}
\bar o_{n-1} + \alpha (1-\bar o_{n-1}) & n>0 \\
0 & n=0
\end{cases}
$$

## 7. Upper-Confidence-Bound(UCB) Action Selection

虽然 $\epsilon$-greedy 方法可以尝试exploration，但是对于各个行为的选择没有偏好。比较理性的方法是尝试更有潜力的行为。

一种有效的非贪婪行为选择方法是：

$$
A_y \doteq \arg_a\max\left[Q_(a)+c\sqrt{\frac {\ln t} {N_t(a)}} \right]
$$

其中，$N_t(a)$ 是行为 $a$ 的计数器。置信系数$c>0$　控制exploration的程度。若　$N_t(a)=0$，那么就选最大值的行为。

平方根用于控制不确定的程度。

但是，这个方法比$\epsilon$-greedy 更为困难：
1. 处理非稳定问题时，这个处理使得问题更为复杂
2. 处理状态空间比较大的问题时，使用函数逼近的方法也更为困难。

因此，在更先进的方法中，这个UCB的想法常常会不切实际。

## 8. Gradient Bandit Algorithm

之前讲到的都是基于对行为价值进行估计的方法，但这不是唯一的选择，比如:**考虑基于量化偏好的行为选择。**

偏好值记为 $H_t(a)$，和收益相关的解释无关。只是行为的相对偏好。

$$
Pr\{A_t=a\} \doteq \frac{e^{H_t(a)}}{\sum_{b=1}^k e^{H_t(b)}} \doteq \pi_t(a) \quad \forall a
$$

其中，$\forall a, H_1(a)=0$

$H_t(a)$ 的更新：

$$
H_{t+1}(a) \doteq H_t(a) + \alpha (R_t - \bar R_t)(\mathbb{1}_{a=A_t}-\pi_t(a))
$$

可以看出，如果在$t$步选择了的行为$A_t$ 的收益高于基线，那么后期会增加被选的可能，并且削弱其他的非$A_t$ 行为。

$\bar R_t$ 是到当前步的收益平均值，可以作为基线。不使用基线的情况可令 $\bar R_t = 0$，但这样做效果会变得很差。

可以证明：gradient bandit algorithm 的期望更新梯度等于收益梯度。并且，这是一个随机梯度上升。并且，baseline 的选择并不影响随机梯度的特性。

选择平均值作为基线不一定能获得最佳的效果，但是非常简单，别且实践中的效果也不错。

> 关于 Bandit Gradient Algorithm 的证明

再完却的梯度提升方法中，量化偏好的更新是

$$
H_{t+1}(a) = H_t(a) + \alpha \frac{\partial\mathbb{E}[R_t]}{\partial H_t(a)}
$$

其中，

$$
\mathbb{E}[R_t] = \sum_x \pi_t(x)q_*(x)
$$

那么有

$$
\begin{aligned}
\frac{\partial\mathbb{E}[R_t]}{\partial H_t(a)} &= \frac{\partial}{\partial H_t(a)}\left[\sum_x \pi_t(x)q_*(x) \right] \\
&= \sum_x q_*(x) \frac{\partial \pi_t(x)}{\partial H_t(a)} \\
&= \sum_x \left(q_*(x) - B_t\right) \frac{\partial \pi_t(x)}{\partial H_t(a)}
\end{aligned}
$$

其中，$B_t$ 是与 $x$ 无关的基线值。由于各个方向上的梯度值之和为 $0$，即 $\sum_x \frac{\partial \pi_t(x)}{\partial H_t(a)} = 0$，因此基线值的插入并不影响等式。

$$
\begin{aligned}
\frac{\partial\mathbb{E}[R_t]}{\partial H_t(a)} &=\sum_x \pi_t(x)\left(q_*(x) - B_t\right) \frac{\partial \pi_t(x)}{\partial H_t(a)}/\pi_t(x) \\
&= \mathbb{E}\left[(q_*(A_t) - B_t) \frac{\partial \pi_t(A_t)}{\partial H_t(A_t)}/\pi_t(A_t) \right] \\
&= \mathbb{E}\left[(R_t - \bar R_t) \frac{\partial \pi_t(A_t)}{\partial H_t(a)}/\pi_t(A_t) \right] \\
\end{aligned}
$$

其中，基线值的选取比较随机，并且由于 $\mathbb{E}[R_t|A_t] = q_*(A_t)$，替代进入也成立。此外，根据除法倒数定理：

$$
\begin{aligned}
\frac{\partial \pi_t(x)}{\partial H_t(a)} &= \frac{\partial}{\partial H_t(a)} \left[ \frac{e^{H_t(x)}}{\sum_{y=1}^k e^{H_t(y)}}  \right] \\
&= \frac{ \frac{\partial e^{H_t(x)}}{\partial e^{H_t(a)}}\sum_{y=1}^k e^{H_t(y)} - e^{H_t(x)} \frac{\partial \sum_{y=1}^{k} e^{H_t(y)}}{\partial H_t(a)} }{ \left( \sum_{y=1}^k e^{H_t(y)} \right)^2 } \\
&= \frac{ \mathbb{1}_{a=x} e^{H_t(x)}\sum_{y=1}^k e^{H_t(y)} - e^{H_t(x)}  e^{H_t(a)}} { \left( \sum_{y=1}^k e^{H_t(y)} \right)^2 } \\
&= \frac{ \mathbb{1}_{a=x} e^{H_t(x)} } {  \sum_{y=1}^k e^{H_t(y)}} - \frac{e^{H_t(x)}  e^{H_t(a)}} { \left( \sum_{y=1}^k e^{H_t(y)} \right)^2 }\\
&= \mathbb{1}_{a=x}\pi_t(x) - \pi_t(x)\pi_t(a) \\
&= \pi_t(x)\left( \mathbb{1}_{a=x} - \pi_t(a) \right)
\end{aligned}
$$

代入上式期望可得

$$
\begin{aligned}
\frac{\partial\mathbb{E}[R_t]}{\partial H_t(a)} &= \mathbb{E}\left[(R_t-\bar R_t) \pi_t(A_t) (\mathbb{1}_{a=A_t} - \pi_t(a)) / \pi_t(A_t)\right] \\
&= \mathbb{E}\left[(R_t-\bar R_t) (\mathbb{1}_{a=A_t} - \pi_t(a)) \right] \\
&= (R_t-\bar R_t) (\mathbb{1}_{a=A_t} - \pi_t(a))
\end{aligned}
$$

由此得证：

$$
H_{t+1}(a) = H_t(a) + \alpha (R_t \bar R_t) (\mathbb{1}_{a=A_t} - \pi_t(a))
$$

## 9. Associative Search (Contextual Bandits)

目前，所涉及的问题仅仅是非关联（nonassociative）的任务，即，不同的行为不需要关联不同的环境情形。

比如对于 $k$-armed bandit 问题，每一步的概率都会变。

现在假设，你可以知道某些暗示，比如颜色和胜率的关系，那么你可以将之关联。

关联搜索任务：包含
1. trial-and-error 学习以搜索最佳的行为
2. 关联这些行为与最适合的执行情形。

