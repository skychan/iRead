# iRead 《Hadoop Illuminated》

这本书是开源的在线版，网址是[Hadoop Illuminated](http://hadoopilluminated.com)

## What is Big Date?

大数据是非常巨量，松散结构的数据集，传统的存储方法以及不够用了。

在Hadoop 出现之前，机器产生的数据往往被忽视，没有去捕捉，这是由于和存储打交道是不可能的也是低效的。

大数据的主要来源：
- Web
- Click stream
- sensor
- Connected Devices

挑战：
- 巨量
- 非结构化和半结构化
- 不能处理的数据是没有保存的意义的，向大集群转移数据非常耗时。解决方法：原地处理

## Hadoop and Big Data

Hadoop 如何处理大数据问题：在集群上运行。

Hadoop 被设计成运行在集群上的。
存储的同时也可以计算。

商业案例：
- 以合理的价格提供存储
- 可以捕捉新的更多的信息
- 更长期存储数据
- 可伸缩分析（MapReduce）

## Soft Introduction to Haddoop

$Hadoop = HDFS + MapReduce$

简单来说：存储 + 处理

HDFS (Hadoop Distributed File Systems)
- 水平拓展性
- 兼容硬件
- 容灾

HBase 高性能，没有大小限制的数据库，运行在Hadoop的顶层。
- key-value 存储数据库，非关系数据库（不存Null值）可以存更为稀疏的数据。

Pig - SQL 类似的
- 移动HDFS cages, 消化大量信息。

Hive - 数据仓库设计，并且支持ETL (抽取，转化，加载)能力。
- SQL 管理HDFS cages (QL)

ZooKeeper:

中心维护服务，配置信息、命名、提供分布式同步，提供组服务。

## HDFS 介绍

one of the core components of Hadoop

**问题1**: 在单机中无法存储大量数据。

Hadoop 方案：存到多机环境中。

**问题2**: 高级终端机都很贵

Hadoop 方案：在商业硬件上运行

**问题3**: 商业硬件可能会失效。

通常，一些公司会使用高质量的配件，实时做备份来应对偶尔的失效。但是会增加成本。

Hadoo 方案：软件智能来应对硬件失效

**问题4**：硬件的失效导致数据丢失

Hadoop 方案：replicate(duplicate) 数据，副本数据

**问题5**:分布式节点是如何协作的？

Hadoop 方案：Master 节点来协调所有的worker 节点

- Master : Name Node
- Slave : Data Node

HDFS 是弹性的，通过在节点中的副本数据，应对节点的失效。

在HDFS 中，文件只能写一次，可以追加（append），这是由于在机械硬盘中找数据很慢？Hadoop 试图去最小化磁盘寻找(disk seeks)

## MapReduce 介绍

IO以及网络瓶颈，读数据。

1. MapReduce 有master 和workers节点，但是他们不仅仅是push or pull，他们之间会协作
2. master 会分配工作量给附近的worker，所以没有工作量会剩下或者未完成。
3. worker 定期汇报(send periodic heartbeats)，如果worker 有一段时间沉默了，那么master 会分配任务给其他worker。
4. 所有的数据都在HDFS 中，所以就避免了中心服务的概念。这样就可以实现并行访问。MapReduce 不会更新数据，而是写到新的输出中。有点类似函数式编程，避免更新查找。
5. MapReduce 是网络和存储架件，优化网络流量。

Master 在 masters 的配置文件里，Slave 在 slaves 配置文件里。Master 有 "Job Tracker"进程， Slave 有 "Task Tracker" 进程。

Master 并不是一开始就霸任务分割好的，而是有一个分配下一个任务量的算法。所以不需要预热时间，

在分布式系统中，网络带宽是最为稀缺的资源，需要小心用（be used with care）。

优化前需要调正和稳定，并不是一个容易的事情。

通过IP 地址，获取数据和Task Tracker的信息。

## Business Intelligence Tools for Hadoop and Big Data

BI 工具提供非常丰富、友善的环境，可以分割、恢复数据，而且大部分有GUI环境。

- Data Validataion：验证数据，确保有一定的限制，可以清洗、去重数据
- Share with Others：在组织内部或者外部共享结果
- Local Rendering：加速专用的数据拓展。
