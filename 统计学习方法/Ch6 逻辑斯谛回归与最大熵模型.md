# Ch6 逻辑斯谛回归与最大熵模型

Logistic Regression：统计学中经典分类方法

最大熵：概率模型学习的一个准则，将分类问题推广到最大熵模型

都是对数线性模型。

## 逻辑斯谛回归模型

**Logistic 分布：**

$$
F(x) = P(X\le x) = \frac 1 {1 + e^{-(x-\mu)/\gamma}}
$$

$\mu,\gamma$ 分别是位置参数和状态参数

该曲线以点 $(\mu, \frac 1 2)$ 为中心对称，即：

$$
F(-x+\mu) - \frac 1 2 = -F(x -\mu) + \frac 1 2
$$

在中心点附近增长速度快，两端较慢。

**二项逻辑斯谛回归模型：**

由条件概率分布 $P(Y|X)$ 表示，是参数化的逻辑斯谛分布。$X\in\mathbb R, y\in\{0,1\}$

$$
\begin{aligned}
P(Y=1|x) = \frac{\exp(w\cdot x + b)}{1 + \exp(w\cdot x + b)}\\
P(Y=0|X) = \frac{1}{1+\exp(w\cdot x + b)}
\end{aligned}
$$

事件的对数几率：

$$
logit (p) = \log \frac p {1-p} = w\cdot x + b
$$

就是说，在逻辑斯谛回归模型中，输出 $Y=1$ 的对数几率是输入 $x$ 的线性函数。


**模型参数估计**

可以使用极大似然估计法来估计模型参数。

似然函数：

$$
\prod_{i=1}^N [\pi(x_i)]^{y_i}[1-\pi(x_i)]^{1-y_i}
$$