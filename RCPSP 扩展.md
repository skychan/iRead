# A survey of variants and extension of the resource-constrainted project scheduling problem (S. Hartmann)

## 从活动概念推广（generalized activity concepts）

### 可中断调度（Preemptive scheduling）

活动可以被打断，然后之后再接着做

(D. Debels, M. Vanhoucke, The impact of various activity assumptions on the
lead time and resource utilization of resource-constrained projects) 更进一步假设这分片可以被并行处理，而不需要重新来过。(需要看看这篇论文的研究背景及应用场景，为什么可以并行)

### 资源的需求随时变化（Resource requests varying with time）

有需求是时间的函数，或者给出需求的最小值和最大值，给个区间。
可以通过一定技术把它转化为时不变的需求。

### 准备时间（setup time）
有多种形式
- 只和当前activity 有关
- 和前续任务有关
- 和连续任务的相似程度有关

还有断后时间（removal times）也会有所考虑

### 多种加工模式（multiple modes）

(类似于FJSP 之于 JSP)

### 权衡问题（trade-off problem）

1. Discrete time-resource tradeoff problem: 

$$
p_j r_{j,k} \ge w_j
$$

$$
(p_j - 1) r_{j,k} < w_j \quad and \quad p_j (r_{j,k}-1) < w_j
$$

2. Discrete time-cost tradeoff problem:


这两种都是Multi-mode 的特例

##  推广临时约束（Generalized temporal constraints）

- 可以规定，前后活动的最小时间间隔（time lag），包括最小，最大。
- 时间切换约束，把任务描述成工作和休息的循环。
- 增加逻辑节点，普通的有后续的工作的节点是AND，也可以是OR，XOR等

## 推广资源约束(Generalized resource constraints)

### 非更新资源和双重资源约束

双重约束也就是同一时间同时要考虑可再生和非可再生资源

### 部分可再生资源

### 累加资源

### 连续资源

### 专用资源

### 时变资源

# The impact of various activity assumptions on the lead time and resource utilization of resource-constrained projects (Mario Vanhoucke)

Work content （工作总量？$p \times r$）

这个扩展总的来说，从 **时间** 上切分任务(activity-splitting)，然后可以并行化处理(fast tracking)。
The within-activity fast tracking option is inspired by the idea that
activities are often executed by groups of resources (with a fixed availability), but the total work can often be done
by multiple groups (in parallel).

# Project scheduling with resource capacities and requests varying with time a case study

任何一个活动 $j$ 必须在时间窗 $[ES_j, LF_j]$ 内完成。

需求的work content 不是一个矩形。

# Time-varying resource requirements and capacitites

主要看Application，也就是案例

一个医药研究项目。

主要是时间上的不确定。