# Ch 1. Introduction
<!-- TOC -->

- [1. Reinforcement Learning](#1-reinforcement-learning)
- [2. Elements of Reinforcement Learning](#2-elements-of-reinforcement-learning)
- [3. Limitations and Scope](#3-limitations-and-scope)
- [4. An Extended Example: Tic-Tac-Toe](#4-an-extended-example-tic-tac-toe)
- [5. Summary](#5-summary)
- [6. Early History of Reinforcement Learning](#6-early-history-of-reinforcement-learning)

<!-- /TOC -->
## 1. Reinforcement Learning

**RL**: is simultaneously a problem, a class of solution methods that work well on the problem, and the field that studies this problem and its solution metods.

RL is different from both of _supervised learning_ and _unsupervised learning_ because:

1. Supervised Learning is learning from a training set of labeled examples provided by a knowledgeable external supervisor.
2. Unsupervised Learning is typicall about finding structure hidden in collections of unlabeled data. 
3. Reinforcement Learning is to maximizing a reward signal.

RL agent has to *exploit* what is has already experienced in order to obtain reward, but it also has to explore in order to make better action selections in the future.

强化学习的另一个关键特征：考虑整个问题，使用目标引导的agent 与不确定的环境进行交互。


## 2. Elements of Reinforcement Learning

- **policy**: may be stochastic, spcifying probabilities for each action
- **reward signal**: the goal
- **value function**: specifies what is good in the long run.
- _model of the environment_: methods use models and planning called model-based methods. Explicitly trial-and-error learners is model-free methods.

planning: any way of deciding on a course of action by considering possible future situations before they are actually experienced.

> The most important component of almost all RL algorithm is a method for efficiently estimating values.

> Modern RL spans the spectrum from low-level, trial-and-error learning to high-level, deliberative planning.

## 3. Limitations and Scope

- RL relies heavily on the concept of state--as input to the policy and value function, and as both imput to and output from the model.


State: as a signal convying to the agent some sense of "how the environment is" at a particular time. **Whatever information is available to the agent about its environment**.

Evolutionary methods have advantages on problems in which the learning agent cannot sense the complete state of its environment.

## 4. An Extended Example: Tic-Tac-Toe

If we let $S_t$ denote the state before the greedy move, and $S_{t+1}$ the state after the move, then the update to the estimated value of $S_t$, denoted $V(S_t)$ can be written as

$$
V(S_t) \gets V(S_t) + \alpha \left[V(S_{t+1}) - V(S_t) \right]
$$

where $\alpha$ is a small positive fraction called the _step-size parameter_, which influences the rate of learning.

Key features of RL:
- there is the emphasis on learning while interacting with an environment, in this case with an opponent player
- there is a clear goal, and correct behavior requires planning or foresight that takes into account delayed effects of one's choices.

the simple RL player would learn to set up multi-move traps for a shortsighted opponent.

## 5. Summary

RL is a computational approach to understanding and automating goal-directed learning and decision making.

> RL is the first field to seriously address the computational issues that arise when learning from interaction with an environment in order to achieve long-term goals.

RL uses the formal framework of MDP to define the interaction between a learning agent and its environment in terms of states, actions and rewards.

## 6. Early History of Reinforcement Learning

RL 的研究历史主要包括两条主线：
1. 关于　trial-and-error
2. 关于最优控制和使用值函数的解，以及动态规划
3. 是temporal-difference 方法将这两者有机结合到了一起。

