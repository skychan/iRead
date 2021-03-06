# Ch6 深度前馈网络

## 架构设计

架构指的是网络的整体结构，它应具有多少单元，以及这些单元应该如何连接。

### 万能的近似性质和深度

通过矩阵乘法将特征映射到输出，仅能表示线性函数。

万能近似定理表明，一个前馈神经网络如果具有线性输出层和至少一层具有任何一种“挤压”性质的激活函数的隐藏层，只要给予网络足够数量的隐藏单元，它可以以任意精度来近似任何从一个有限空间到另一个有限维度空间的 Borel 可测函数。

万能近似定理意味着无论我们试图学习什么函数，我们知道一个大的 MLP 一定能够表示这个函数。

然而，即使 MLP 能够表示该函数，学习也可能因两个不同的原因而失败。
1. 用于训练的优化算法可能找不到用于期望函数的参数值。
2. 训练算法可能由于过拟合而选择了错误的函数。

并且，万能近似定理也没说明这个网络有多大。
