三：hadoop调度器 Hadoop Job Scheduler
Hadoop默认的调度器是基于队列的FIFO调度器：
    所有用户的作业都被提交到一个队列中，然后由JobTracker先按照作业的优先级高低，再按照作业提交时间     的先后顺序选择将被执行的作业。
    优点: 调度算法简单明了，JobTracker工作负担轻。
    缺点: 忽略了不同作业的需求差异。
Fair Scheduler(公平调度器)：
    1：多个Pool，Job需要被提交到某个Pool中；
    2：每个pool可以设置最小 task slot（猜测最小的job数），称为miniShare
    3：FS会保证Pool的公平，Pool内部支持Priority（优先级）设置，支持资源抢占（优先级）


