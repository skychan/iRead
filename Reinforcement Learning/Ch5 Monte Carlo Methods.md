# Ch 5. Monte Carlo Methods

估算价值函数，并发现最优策略。

并不假设环境知识完全可得。

Monte Carlo 方法仅仅需要经验——采样，通过仿真交互获得收益即可。

## Monte Carlo Prediction

在给定策略的情况下，学习 状态-价值 函数：
平均化进入过的状态观测返回值。

随着观测值的增多，这个平均值会收敛到期望值。

每一次进入状态 $s$ 的 episode 称为一个 $s$ 的访问。若在同个 episode 中，多次访问 $s$，那么称第一次为 到 $s$ 的第一次访问。

- first-visit MC
- every-visit MC

Monte Carlo 方法并不是简历在之前状态的预测值，而是相互独立的。非 bootstrap

## Monte Carlo Estimation of Action Values

若模型不可得，那么使用行为值估计比状态值估计更为有效。

使用 (s,a) 对来做 MC

唯一的复杂是，可能有非常多的 (s,a) 对从来没有被访问过

**Exploring starts**: Specifying that the episodes start in a state-action pair, every pair has a non-zero probability of being selected as the start. This guarantees that all state-action pairs will be visited an infinite number of times in the limit of an infinite number of episodes.

## Monte Carlo Control

所谓的蒙特卡洛控制：
用于近似最优策略。

两个假设：
1. episodes 有着 exploring starts
2. 策略评估可以在有限的 episodes 内完成

然而，为了得到实用的算法，我们需要将这两个假设去除。

## Monte Carlo Control without Exploring Starts

如何去除　Exploring starts？

唯一的通用办法是，保证所有行为被无限选择，并是被agent连续选择。

有两个方式
1. on-policy 方法：尝试去评估或提升作选择的行为。
2. off-policy 方法：评估或提升使用不同于生成数据的行为。

## Off-policy Monte Carlo Control

两个功能分开：
1. Value of a policy
2. Control

- 用于生成行为的策略：behavior policy
- 与之无关的用于评估与提升的：target policy

分开的好处：target 可能是确定的（greedy），behavior 可以继续采样所有可能的行为

They follow the behavior policy while learning about and improving the target policy.

前提是，behavior policy 会被非零概率选择到。（Coverage）

然而，一个潜在的问题是，这个方法仅仅从实验的 episodes 中学习，当所有剩下的其他 episode 是贪婪的。

## Discounting-aware Importance Sampling

