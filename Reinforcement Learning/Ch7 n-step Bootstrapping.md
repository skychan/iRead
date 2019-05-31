# Ch 7 $n$-step Boostrapping

统一 MC 和单步 TD 方法，由于这两个方法不会总是够好。

n-step 方法是介于两者之间，最佳的方法一般也会介于这两个极端之间。

n-step 方法通常用于介绍资格迹（eligibility traces）算法的思想，这是一个可以同时 bootstrap 多个事件间隔的方法。延迟资格迹的收益。

<!-- TOC -->

- [1. $n$-step TD Prediction](#1-n-step-td-prediction)
- [2. $n$-step Sarsa](#2-n-step-sarsa)
- [3. $n$-step Off-policy Learning](#3-n-step-off-policy-learning)
- [4. Off-policy Learning without Importance Sampling: The $n$-step Tree Backup Algorithm](#4-off-policy-learning-without-importance-sampling-the-n-step-tree-backup-algorithm)

<!-- /TOC -->

## 1. $n$-step TD Prediction

temporal difference 扩展了 $n$ 步称为 $n$-step TD 方法。

若我们忽略 actions，那么对 $S_t$ 的估计需要连续的 状态-收益 序列：

$$
S_t, R_{t+1}, S_{t+1}, R_{t+2},\dots,R_T,S_T
$$

MC 中：

$$
G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{T-t-1} R_T
$$

$T$：目标的更新数量。

one-step 更新方法：

$$
G_{t:t+1}\doteq R_{t+1} + \gamma V_t(S_{t+1})
$$

n-step TD:

$$
G_{t:t+n}\doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V_{t+n-1}(S_{t+n})
$$


需要等 $R_{t+n}$ 和 $V_{t+n-1}$ 都遇到后，才能进行估计。因此需要一个估计的方法。

一个自然的更新的方式：

$$
V_{t+n}(S_t) \doteq V_{t+n-1}(S_t) + \alpha \left[ G_{t:t+n} - V_{t+n-1}(S_t) \right], \quad 0 \le t < T
$$

其他状态的值不更新：

$$
V_{t+n} (s) = V_{t+n-1} (s), \quad s \ne S_t
$$

## 2. $n$-step Sarsa

$$
G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n Q_{t+n-1}(S_{t+n}, A_{t+n}), \quad n\ge 1, 0\le t < T-n
$$

相应的估计求解算法：

$$
Q_{t+n}(S_t, A_t) \doteq Q_{t+n-1}(S_t, A_t) + \alpha\left[ G_{t:t+n} - Q_{t+n-1}(S_t,A_t) \right]
$$

对于其他的状态或者行为，保持原样。

n-step Expected Sarsa 的版本只需要将 $Q_{t+n-1}(S_{t+n})$ 修改为 $\bar V_{t+n-1}(S_{t+n})$ 即可。

## 3. $n$-step Off-policy Learning

Off-policy learning is learning the value function for one policy $\pi$, while following another policy $b$. 
Often, $\pi$ is the greedy policy for the current action-value-function estimate, and $b$ is a more exploratory policy, perhaps $\varepsilon$-greedy.

比如 n-step TD 版本：

$$
V_{t+n}(S_t) \doteq V_{t+n-1}(S_t) + \alpha\rho_{t:t+n-1} \left[ G_{t:t+n} - V_{t+n-1}(S_t) \right], \quad 0 \le t < T
$$

其中，$\rho$ 是重要性采样比例。

$$
\rho_{t:h} \doteq \prod_{k=t}^{\min (h,T-1)} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}
$$

类似的，off-policy 的 Sarsa 可以表示：

$$
Q_{t+n}(S_t, A_t) \doteq Q_{t+n-1}(S_t, A_t) + \alpha \rho_{t+1:t+n}\left[ G_{t:t+n} - Q_{t+n-1}(S_t,A_t) \right]
$$

## 4. Off-policy Learning without Importance Sampling: The $n$-step Tree Backup Algorithm

one-step tree-backup 就是 Expected Sarsa

$$
G_{t:t+1} \doteq R_{t+1} + \gamma \sum_a\pi(a|S_{t+1})Q_t(S_{t+1},a)
$$

扩展到 n

$$
G_{t:t+n}\doteq R_{t+1} \gamma \sum_{a\ne A_{t+1}} \pi(a|S_{t+1}) Q_{t+n-1}(S_{t+1,a}) + \gamma \pi (A_{t+1},S_{t+1}) G_{t+1:t+n}
$$

n-step Sarsa:

$$
Q_{t+n}(S_t,A_t)\doteq Q_{t+n-1}(S_t,A_t) + \alpha \left[ G_{t:t+n} - Q_{t+n-1} (S_t, A_t) \right]
$$

