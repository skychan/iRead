# Ch2 线性代数

广播：$\bm C = \bm A + \bm b$

$$
C_{i,j} = A_{i,j} + b_j
$$

范数是将向量映射到非负值的函数，衡量原点到 $\bm x$ 的距离。

满足以下性质：

- $f(\bm x)=0 \Rightarrow \bm x = \bm 0$
- $f(\bm x + \bm y) \le f(\bm x) + f(\bm y)$ 三角不等式
- $\forall \alpha\in\mathbb{R}, f(\alpha\bm x) = |\alpha|f(\bm x)$

$L^p$ **范数**：

$$
||\bm x||_p = \left(\sum_i |x_i|^p\right)^{\frac 1 p}
$$

衡量矩阵的大小，使用 Frobenius 范数：

$$
||\bm A||_F = \sqrt{\sum_{i,j}A^2_{i,j}}
$$

**特征分解**:

$$
\bm A = \bm Q \bm\Lambda \bm Q^T
$$

$\bm Q$ 是 $\bm A$ 的特征向量组成的正交矩阵。

**奇异值分解（SVD）**：

$$
\bm A = \bm U \bm D \bm V^T
$$

$\bm U, \bm V$ 是方阵，包含左右奇异向量（列）

**Moore-Penrose 伪逆：**

$$
\bm A ^+ = \bm V \bm D^+ \bm U^T
$$

**迹运算**

$$
Tr(\bm A) = \sum_i A_{i,i}
$$

