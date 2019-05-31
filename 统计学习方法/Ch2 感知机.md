# Ch2 感知机

感知机是二分类的线性分类模型。

- 输入：实例的特征向量。
- 输出：实例的类型（二值）。

感知机学习旨在求出将训练数据进行线性划分的分离超平面，属于判别模型。

<!-- TOC -->

- [1. 感知机模型](#1-感知机模型)
- [2. 感知机学习策略](#2-感知机学习策略)
- [3. 感知机学习算法](#3-感知机学习算法)

<!-- /TOC -->

## 1. 感知机模型

$$
f(x) = sign(w\cdot x + b)
$$


$w,b$ 为感知机模型参数。

## 2. 感知机学习策略

损失函数

$$
L(w,b) = - \sum_{x_i\in\mathcal M} y_i (w\cdot x_i + b)
$$

是感知机的经验风险函数。其中，$\mathcal M$ 是误分类的点的集合。

## 3. 感知机学习算法

损失函数的梯度：

$$
\nabla_w L(w,b) = - \sum_{x_i\in\mathcal M} y_i x_i
$$

$$
\nabla_b L(w,b) = -\sum_{x_i\in\mathcal M}y_i
$$

迭代更新：

$$
\begin{aligned}
w & \gets w + \alpha y_ix_i \\
b & \gets b + \alpha y_i
\end{aligned}
$$

