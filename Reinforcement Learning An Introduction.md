# iRead _Reinforcement Learning - An Introduction_

To evaluate a policy, an evolutionary method must hold it fixed and play many games against the opponent, or simulate many games using a model of the opponent. The frequency of wins gives an unbiased estimate of the probability of winning with that policy, and can be used to direct the next policy selection.

Each policy change is made only after many games, and only the final outcome of each game is used.

What happens during the games is ignored.

But value function methods, allow individual states to be evaluted. 

In the end, both evolutionary and value function methods search the space of policies, but learning a value function takes advantage of information available during the course of play.

<<<<<<< HEAD
## Finite Markov Decision Processes

包括 evaluative feedback, associative aspect -- choosing different actions in different situations.

是一个典型的形式化的序列化决策制定。

Actions influene not just immediate rewards, but also subsequent situations, or states, and through those future rewards.

MDPs involve delayed reward and the need to tradeoff immediate and delayed reward.

Estimate the value $q_*(s,a)$ of each action $a$ in each state $s$

Or estimate the value $v_*(s)$ of each state given optimal action selections.

These state-dependent quantities are essential to accurately assigning credit for long-run consequences to individual action selections.


- **Agent**: the learner and decision maker is called agent.(controller)
- **Environment**: The thing it interacts with, comprising everything outside the agent. (controlled system or plant)
- **Action**:(control signal)

## Goals and Rewards

在强化学习中，agent 的目的是将特殊情况公式化

The use of a reward signal to formalize the idea of a goal is one of the moet distinctive features of reinforcement learning.

Goal: maximize the cumulative reward it received after time step $t$ is denoted $R_{t+1}, R_{t+2}, \dots$.

Episodes are sometimes called "trials" in the literature.

Tasks with episodes of this kind are called episodic tasks. In episodic tasks we sometimes need to distinguish the set of all nonterminal states.

## 动态规划

经典的 DP 算法在强化学习中的应用非常有限，原因是：

1. 关于模型的假设需要非常完美
2. 计算代价大 

假设在有限MDP中（状态、决策、收益都是有限集）

通常，获得近似解的方法是量化连续的状态和行为，转化为使用有限状态的动态规划方法。

DP/RL 的关键想法：使用价值函数来组织搜索好策略的结构。

### Policy Evaluation (Prediction)

对于任意计算 state-value 函数（$v_{\pi}$）

迭代式的求解方法更为适用。

DP 的最大问题是，包含了需要需要整个状态集合调整的操作，也就是说，每次移动需要重新计算整个空间。

**异步动态规划**

是一种原地迭代的动态规划算法。并不是按照系统的状态清除来组织的。可以按照任意的方法来更新状态的价值。

=======

## Associative search

将试错法获得的行为决策和当前的情景结合起来，（Contextual bandits）

比如，每次老虎机的概率是变化的，但是你可能从尝试中得知，某些颜色的概率会大一些，那么你在做决策的时候，行为会受颜色的影响。

更像完全体的 RL：行为不光会影响下一个环境（next situation），还会影响下一次的收益（next reward）

If actions are allowed to a↵ect the next situation as well as the reward, then we have the full reinforcement learning problem.
>>>>>>> 63cf003f3f1dee802cf39bb3154a34bae4a8b260
