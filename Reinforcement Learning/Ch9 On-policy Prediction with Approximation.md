# Ch9 On-policy Prediction with Approximation

研究强化学习中的函数近似，用于估计 状态-值 函数，使用 on-policy 数据：从已知策略 $\pi$ 来生成经验以估计 $v_\pi$。 

和前面章节不同的是，这里的近似值函数不是以表的形式呈现的，而是使用参数化的函数形式，伴随着权重向量 $\bm{\mathrm w}\in\mathbb R^d$。

$$
\hat v(s,\mathrm w) \approx v_\pi(s)
$$

其中，$d \ll |\mathcal S|$，函数可以是各种形式的，甚至是决策树的分割点和叶值。

