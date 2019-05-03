# Ch4 数值计算

## 上溢和下溢

舍入误差导致的一系列问题，如果在设计时没有考虑最小化舍入误差的累积，在实践时算法可能会失效。

- 毁灭性的舍入误差**下溢**：比如被除为0
- 极具破坏力的误差形式**上溢**：大量级的数被近似为 $\infty$ 或 $-\infty$

比如 softmax 函数

$$
\mathrm{softmax}(\bm x_i) = \frac {\exp (x_i)}{\sum_{j=1}^{n} \exp(x_j)}
$$

可以通过 

$$
\bm z = \bm x - \max_i x_i
$$

来解决。

## 病态条件

条件数指的是函数相对于输入的微小变化二变化的快慢程度。输入被轻微扰动而迅速改变的函数对于科学计算来说，可能是有问题的。

比如，对于矩阵特征值分解

$$
\max_{i,j} \left|\frac {\lambda_i} {\lambda_j}\right|
$$

是最大和最小特征值的模之比。当该数很大时，矩阵求逆对输入的误差特别敏感。

这种敏感是矩阵本身的固有特性，而不是矩阵求逆期间舍入误差的结果。

## 基于梯度的优化

- 最大化或最小化：目标函数、准则
- 最小化：代价函数、损失函数、误差函数

梯度：相对一个向量求导的导数，包含所有偏导数的向量。

- Jacobian 矩阵：包含所有的偏导数矩阵

$$
J_{i,j} = \frac \partial {\partial x_j} f(\bm x)_i
$$

- Hessian 矩阵：包含所有的二阶导数矩阵

$$
H(f)(\bm x)_{i,j} = \frac {\partial^2}{\partial x_i \partial x_j} f(\bm x)
$$

Hessian 等价于梯度的 Jacobian 矩阵

特定方向 $\bm d$ 上的二阶导数可以写成

$$
\bm d^T \bm H \bm d
$$

当 $\bm d$ 是 $\bm H$ 的一个特征向量时，这个方向的二阶导数就是对应的特征值。

> 最大特征值确定最大二阶导数，最小特征值确定最小二阶导数。

## 约束优化

Karush-Kuhn-Tucker 方法 是针对约束优化非常通用的解决方案。

约束为
- $m$ 个不等式 $g^{(i)}$
- $n$ 个等式 $h^{(j)}$

为每个约束引入变量，称为 KKT 乘子，广义 Lagrangian 可以定义为

$$
L(\bm x, \bm \lambda, \bm \alpha) = f(\bm x) + \sum_i \lambda_i g^{(i)}(\bm x) + \sum_j \alpha_j h^{(j)}(\bm x)
$$

对于最小化问题，等价于

$$
\min_{\bm x}\max_{\bm \lambda}\max_{\bm\alpha,\bm\alpha\ge 0} L(\bm x,\bm\lambda,\bm\alpha)
$$

KKT 条件：

- 必要条件：

1. 广义 Largangian 的梯度为0

$$
\nabla f(\bm x^*) + \sum_i \lambda_i g^{(i)}(\bm x^*) + \sum_j\alpha_jh^{(j)}(\bm x^*) = 0 \\
\lambda_i g^{(i)}(\bm x^*) = 0
$$

2. 所有 KKT 乘子的约束都满足
3. 不等式约束的互补松弛性

