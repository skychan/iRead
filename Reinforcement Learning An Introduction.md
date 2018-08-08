# iRead _Reinforcement Learning - An Introduction_

To evaluate a policy, an evolutionary method must hold it fixed and play many games against the opponent, or simulate many games using a model of the opponent. The frequency of wins gives an unbiased estimate of the probability of winning with that policy, and can be used to direct the next policy selection.

Each policy change is made only after many games, and only the final outcome of each game is used.

What happens during the games is ignored.

But value function methods, allow individual states to be evaluted. 

In the end, both evolutionary and value function methods search the space of policies, but learning a value function takes advantage of information available during the course of play.


## Associative search

将试错法获得的行为决策和当前的情景结合起来，（Contextual bandits）

比如，每次老虎机的概率是变化的，但是你可能从尝试中得知，某些颜色的概率会大一些，那么你在做决策的时候，行为会受颜色的影响。

更像完全体的 RL：行为不光会影响下一个环境（next situation），还会影响下一次的收益（next reward）

If actions are allowed to a↵ect the next situation as well as the reward, then we have the full reinforcement learning problem.