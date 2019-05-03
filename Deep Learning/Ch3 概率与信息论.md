# Ch3 概率与信息论

概率论用途：
1. 概率法则告诉我们 AI 系统如何推理，据此设计一些算法来计算或估计由概率论推导出的表达式
2. 用概率和统计从理论上分析我们提出的 AI 系统的行为

不确定性来源：
1. 被建模系统内在的随机性
2. 不完全观测
3. 不完全建模

<!-- TOC -->

- [1. 分布的混合](#1-分布的混合)
- [2. 常见函数](#2-常见函数)
- [3. 连续变量的细节](#3-连续变量的细节)
- [4. 信息论](#4-信息论)

<!-- /TOC -->

## 1. 分布的混合

通过组合一些简单的概率分布来定义的新的概率分布。一种通用的组合方法是构造混合分布。每次实验，样本是由那个组件分布产生的取决于一个 Multinoulli 分布中采样的结果：

$$
P(\mathrm x) = \sum_iP(\mathrm c=i)P(\mathrm x|\mathrm c=i)
$$

混合模型使我们能够一瞥以后 会用到的一个非常重要的概念——潜变量

潜变量是我们不能够直接观测到的随机变量。混合模型的组件标识变量 $c$ 就是其中一个例子。

一个非常强大且常见的混合模型是**高斯混合模型**，它的组件 $p(\mathrm x|\mathrm c=i)$ 是高斯分布

除了均值和协方差以外，高斯混合模型有时会限制每个组件的协方差矩阵为对角的或者各向同性的。

高斯混合模型的参数指明了给么个组件 $i$ 的先验概率 $\alpha_i=P(\mathrm c=i)$。

高斯混合模型是概率密度的万能近似器，任何平滑的概率密度都可以用足够多的高斯混合模型以任意精度来逼近。

## 2. 常见函数

1. Logistic sigmoid 函数

$$
\sigma(x) = \frac 1 {1 + \exp(-x)}
$$

2. softplus 函数

$$
\zeta(x) = \log(1+\exp(x))
$$

形如 ReLU

$$
x^+ = \max (0,x)
$$

一些性质：

$$
\sigma(x)=\frac {\exp(x)} {\exp(x)+\exp(0)} \\
\frac d {dx}\sigma(x) = \sigma(x)(1 - \sigma(x)) \\
1 - \sigma(x) = \sigma(-x) \\
\log(\sigma(x)) = - \zeta(-x) \\
\frac d{dx}\zeta(x) = \sigma(x) \\
\sigma^{-1}(x) = \log\left(\frac 1 {1-x}\right) \quad \forall x\in(0,1)\\
\zeta^{-1}(x) = \log(\exp(x) -1) \quad \forall x > 0\\
\zeta(x) = \int_{-\infty}^x\sigma(y)dy \\
\zeta(x) - \zeta(-x) = x
$$

## 3. 连续变量的细节

若有 $\bm y = g(\bm x)$, 那么

$$
p_x (x) = p_y(g(x)) \left|\frac {\partial g(x)}{\partial x}\right|
$$

于是

$$
p_x(\bm x) = p_y(g(\bm x))\left|\det\left( \frac{\partial g(\bm x)}{\partial \bm x} \right)\right|
$$

## 4. 信息论

基本想法：一个不太可能的事情居然发生了，要比一个非常可能发生的事情发生，能提供更多的信息。

一个事件 $\mathrm x = x$ 的自信息为

$$
I(x) = -\log P(x)
$$

单位是奈特（nats），是以 $\frac 1 e$ 为概率观测到一个事件时获得的信息量。

如果以 2 为对数，单位是比特（bit） 或者香农。

自信息只处理单个的输出，对于整个概率分布中的不确定性总量，我们可以用香农熵。

$$
H(\mathrm x) = \mathbb E_{\mathrm x \sim P}[I(x)]
$$

也记为 $H(P)$