# iRead _Reinforcement Learning - An Introduction_

To evaluate a policy, an evolutionary method must hold it fixed and play many games against the opponent, or simulate many games using a model of the opponent. The frequency of wins gives an unbiased estimate of the probability of winning with that policy, and can be used to direct the next policy selection.

Each policy change is made only after many games, and only the final outcome of each game is used.

What happens during the games is ignored.

But value function methods, allow individual states to be evaluted. 

In the end, both evolutionary and value function methods search the space of policies, but learning a value function takes advantage of information available during the course of play.