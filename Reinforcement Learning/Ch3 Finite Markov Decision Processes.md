# Ch 3. Finite Markov Decision Processes

Finite MDP 问题包括
1. 量化反馈
2. 关联层面：不同的情形下选择不同的行为

MDP 包含：
1. 延迟收益
2. 需要权衡即时收益和延时收益。

- 使用 $q_*(s,a)$ 来评估在每个状态 $s$ 下行为 $a$ 的值，
- 或者使用 $v_*(s)$ 来评估最优选择下状态的值。

---

<!-- TOC -->

- [1. The Agent-Environment Interface](#1-the-agent-environment-interface)
- [2. Goals and Rewards](#2-goals-and-rewards)
- [3. Returns and Episodes](#3-returns-and-episodes)
- [4. Unified Notation for Episodic and Contiuning Tasks](#4-unified-notation-for-episodic-and-contiuning-tasks)
- [5. Policies and Value Functions](#5-policies-and-value-functions)
- [6. Optimal Policies and Optimal Value Functions](#6-optimal-policies-and-optimal-value-functions)
- [7. Optimality and Approximation](#7-optimality-and-approximation)

<!-- /TOC -->

## 1. The Agent-Environment Interface

MDP 是一个从交互过程中求解的直观框架

- **Agent**(Controller): The learner and the decision maker.
- **Environment**(Controlled system, plant): The thing agent interacts with, comprising everything outside the agent.
- **Action**(Control signal)

MDP and agent together thereby give rise to a sequence or *trajectory* like:

```math
S_0,A_0,R_1,S_1,A_1,R_2,\dots
```

> 大写字母都是随机变量，相应的小写字母是对应的确切值

MDP 的动态性（这只是个定义）：

$$
p(s',r|s,a)\doteq Pr\{S_t=s',R_t=r|S_{t-1}=s,A_{t-1}=a\}
$$

根据概率的特性，我们有

$$
\sum_{s'\in\mathcal{S}}\sum_{r\in\mathcal{R}} p(s',r|s,a) = 1 \quad \forall s\in\mathcal{S}, a\in\mathcal{A}(s)
$$

**马尔科夫性**：$p$ 可以完整地描述环境的动态性，并且，每个状态和行为的值只取决于紧前状态和行为，将问题限制在状态，而决策过程。

根据上式，我们可以计算其他的内容：

1. 状态转移概率

$$
p(s'|s,a) \doteq Pr\{S_t=s'|S_{t-1}=s,A_{t-1}=a\} = \sum_{r\in\mathcal{R}} p(s',r|s,a)
$$

2. "状态-行为"的期望收益

$$
r(s,a) \doteq \mathbb{E}[R_t|S_{t-1}=s, A_{t-1}=a] = \sum_{r\in\mathcal{R}}r\sum_{s'\in\mathcal{S}}p(s',r|s,a)
$$

3. "状态-行为-下个状态" 的期望收益

$$
r(s,a,s')\doteq \mathbb{E}[R_t|S_{t-1}=s, A_{t-1}=a, S_t=s'] = \sum_{r\in\mathcal{R}}r \frac{p(s',r|s,a)}{p(s'|s,a)}
$$

---

time-step 可以用任意连续的阶段(arbitrary successive stages)来表示。

agent 和 environment 之间的界限常常和机器人或者动物的物理界限不同。通常，这个界限更加靠近 agent。比如，机器人的铰链和传感硬件常常被认为是环境，而非agent。

类似的，收益常常被自然物理对象或者人工学习系统计算，但却被认为是agent 的外部。

> Anything that cannot be changed arbitrarily by the agent is considered to be outside of it and thus part of its environment.

Agent-Environment 的界限表达是agent 完全的控制能力，而非其知识。

信号：
1. choices made by the agent(actions)
2. basis on which the choices are made (states)
3. agent's goal (rewards)

如何表达这些信号对性能的影响非常巨大，然而，在RL中，这些表达的选择更像是艺术而非科学。

## 2. Goals and Rewards

收益假设：
> That all of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal(called reward).

Agent 总是会学习最大化它的收益。因此，若我们想让它做什么，我们就要提供相应的收益。

若给次级目标肉以，那么agent 就很可能会找到一个方法完成它，而非我们所想的目标。

> The reward signal is your way of communicating to the robot what you want it to achiev, not how you want it achieved.

## 3. Returns and Episodes

通常来说，我们寻找最大化的期望收获

$$
G_t \doteq R_{t+1} + R_{t+2} + \cdots R_T
$$

Episode: 有些文献里也称为 trial，可以是玩游戏的次数，迷宫的路线，或者任何在重复的交互。

每个 episode 会在终止状态结束，然后重新开始新的一个 episode。并且重新开始的 episode 和上一次无关。

- Episodic tasks: Tasks with episodes.
- Continuing tasks: an on-going process-control task, or an application to a robot with a long life span.


- 非结束状态： $\mathcal S$
- 结束状态： $\mathcal S^+$

对于连续的任务，我们通常需要使用折扣来避免无穷大的收益。

$$
G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} \cdots = \sum_{k=0}^\infty \gamma^k R_{t+k+1} \quad 0 \le \gamma \le 1 
$$

如果 $\gamma=0$ 那么我们称之为目光短浅的agent。

$$
G_t \doteq R_{t+1} + \gamma G_{t+1}
$$


## 4. Unified Notation for Episodic and Contiuning Tasks

$S_{t,i}$ 表示 episode $i$ 中在 $t$ 时的状态。

常常，为了防止下标被滥用，我们会使用 $S_t$ 来指代 $S_{t,i}$

同样地，我们可以改写：

$$
G_t \doteq \sum_{k=t+1}^{T} \gamma^{k-t-1} R_k
$$

## 5. Policies and Value Functions

- **价值函数**：是状态或者状态-行为对的函数，用以估计当前agent 所处的状态有多好，或者所处状态下使用某个行为有多好。
- **策略**：是状态到行为选择概率的映射。表示为 $\pi(a|s)$

状态$s$使用策略$\pi$的状态-价值函数：

$$
v_\pi(s) \doteq \mathbb E_\pi [G_t|S_t=s] = \mathbb E_\pi \left[\sum_{k=0}^\infty \gamma^kR_{t+k+1}\middle| S_t=s\right] \quad \forall s\in\mathcal S
$$

类似地，我们定义行为-价值函数：

$$
q_\pi(s,a) \doteq \mathbb E_\pi [G_t|S_t=s, A_t=a] = \mathbb E_\pi \left[\sum_{k=0}^\infty \gamma^k R_{t+k+1}\middle| S_t=s, A_t=a\right]
$$

可以有：

$$
v_\pi(s) = \sum_a q_\pi(s,a)
$$

若采用蒙特卡洛方法进行值的记录和平均，那么遇到大问题会消耗很大的资源。
可以换一种形式来计算：

$$
\begin{aligned}
v_\pi(s) &\doteq \mathbb E_\pi[G_t|S_t=s] \\
&= \mathbb E_\pi[R_{t+1}+\gamma G_{t+1}|S_t=s] \\
&= \sum_a\pi(a|s)\sum_{s'}\sum_{r}p(s',r|s,a)\left(r+\gamma\mathbb E_\pi[G_{t+1}|S_{t+1}=s']\right) \\
&= \sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)\left(r+\gamma v_\pi(s')\right] \quad  \forall s\in\mathcal S
\end{aligned}
$$

这就是 $v_\pi$ 的 Bellman 等式。

## 6. Optimal Policies and Optimal Value Functions

最优的价值函数分别是

$$
v_*(s)\doteq \max_\pi v_\pi(s)
$$

以及

$$
\begin{aligned}
q_*(s,a) &\doteq \max_\pi q_\pi(s,a)\\
&= \mathbb E[R_{t+1}+\gamma v_*(S_{t+1})| S_t=s, A_t=a]
\end{aligned}
$$

根据 Bellman 最优性等式，最佳的状态值函数一定是最大的最佳状态-行为值函数

$$
\begin{aligned}
v_*(s) &= \max_{a\in\mathcal A(s)}q_{\pi_*}(s,a)\\
&=\max_a\mathbb E_{\pi_*}[G_t|S_t=s,A_t=a] \\
&= \max_a\mathbb E_{\pi_*}[R_{t+1} + \gamma G_{t+1}|S_t=s,A_t=a] \\
&= \max_a\mathbb E[R_{t+1}+\gamma v_*(S_{t+1})| S_t=s,A_t=a] \\
&= \max_a\sum_{s',r}p(s',r|s,a)(r+\gamma v_*(s'))
\end{aligned}
$$

同样的，可以得到

$$
\begin{aligned}
q_*(s,a) &= \mathbb E[R_{t+1}+\gamma\max_{a'}q_*(S_{t+1,a})| S_t=s,A_t=a] \\
&= \sum_{s',r}p(s',r|s,a)(r+\gamma\max_{a'}q_*(s',a'))
\end{aligned}
$$

对于有限 MDP，Bellman 最优性等式有唯一解，并且和策略无关。

Bellman 最优性方程是系统方程组，也就是一个状态一个。

直接根据线性方程组求解是一个办法，但是要满足：
1. 准确知道环境的动态性
2. 有足够的计算资源
3. Markov 性

因此，通常采用近似求解方法。

## 7. Optimality and Approximation

由于问题的状态空间可能会非常大，因此，内存很关键。