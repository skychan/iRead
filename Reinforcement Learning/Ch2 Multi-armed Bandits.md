# Ch 2. Multi-armed Bandits

强化学习的一个非常重要的特征是，它使用训练信息，评估计算行为，而不是通过给定正确的行为来指导。

Nonassociative setting: one that does not involve learning to act in more than one situation.

<!-- TOC -->

- [1. A $k$-armed Bandit Problem](#1-a-k-armed-bandit-problem)
- [Action-value Methods](#action-value-methods)

<!-- /TOC -->
## 1. A $k$-armed Bandit Problem

重复地在 $k$ 个不同的选项或者行为中进行选择，每次选择后，从一个固定的概率分布中收获一个数字的奖励。

你的目标是，最大化一定时段内所有收益和的期望。

We deonte action selected on time step $t$ as $A_t$， and the corresponding reward as $R_t$, the value of an arbitrary action $a$ denoted $q_*(a)$ as 

$$
q_*(a) \doteq \mathbb{E}[R_t|A_t = a]
$$

我们假设，你不知道每一个行为的确切价值，但你可能有个估计，记为　$Q_t(a)$。

- Exploiting: 当贪心地根据当前知识选择最大值的行为。可能在当前的１步中可以最大化
- Exploring: 选择非贪心的行为，可能在长期的收益总和中获得最大值

在单次的选择中，不可能同时进行explore和exploit。
因此，平衡这两个操作是非常关键的。

## Action-value Methods

A natural way to estimate the value is by averaging the rewards actually received:

$$
Q_t(a) \doteq \frac{\sum_{i=1}^{t-1}R_i\cdot\mathbb{1}_{A_i=a}}{\sum_{i=1}^{t-1}\mathbb{1}_{A_i=a}}
$$

