# PyTorch 入门
<!-- TOC -->

- [1. 多层全连接神经网络](#1-多层全连接神经网络)
    - [1.1. PyTorch 基础](#11-pytorch-基础)
        - [1.1.1. `Variable` 变量](#111-variable-变量)
        - [1.1.2. `Dataset` 数据集](#112-dataset-数据集)
        - [1.1.3. `nn.Module` 模组](#113-nnmodule-模组)
        - [1.1.4. `torch.optim` 优化](#114-torchoptim-优化)
        - [1.1.5. 模型的保存和加载](#115-模型的保存和加载)
    - [1.2. 线性模型](#12-线性模型)
        - [1.2.1. 线性回归](#121-线性回归)
        - [1.2.2. 多项式回归](#122-多项式回归)
    - [1.3. 分类问题](#13-分类问题)
        - [1.3.1. 二分类的 Logistic 回归](#131-二分类的-logistic-回归)
    - [1.4. 简单的多层全连接前向网络](#14-简单的多层全连接前向网络)
        - [1.4.1. 激活函数](#141-激活函数)
        - [1.4.2. 模型的表示能力和容量](#142-模型的表示能力和容量)
- [2. 卷积神经网络](#2-卷积神经网络)
    - [2.1. 卷积模块](#21-卷积模块)
    - [2.2. 一些操作](#22-一些操作)
    - [2.3. 经典案例](#23-经典案例)
    - [2.4. 图像增强的方法](#24-图像增强的方法)
- [3. 循环神经网络](#3-循环神经网络)
    - [3.1. LSTM](#31-lstm)
    - [3.2. GRU](#32-gru)
    - [3.3. 收敛性问题](#33-收敛性问题)
    - [3.4. PyTorch 的循环网络相关模块](#34-pytorch-的循环网络相关模块)
    - [3.5. 应用场景](#35-应用场景)
        - [3.5.1. 图片分类](#351-图片分类)
        - [3.5.2. 自然语言处理](#352-自然语言处理)
- [生成对抗网络](#生成对抗网络)
    - [生成模型（Generative Model）](#生成模型generative-model)
    - [生成对抗网络](#生成对抗网络-1)

<!-- /TOC -->
## 1. 多层全连接神经网络

### 1.1. PyTorch 基础

Tensor 和 numpy.ndarray 之间的转换：

```python
numpy_b = b.numpy()

torch_e = torch.from_numpy(e)
```

1. 数据类型
- `torch.FloatTensor`
- `torch.IntTensor`
- `torch.rand/torch.randn`
- `torch.arange`
- `torch.zeros`

2. 计算
- `torch.abs`
- `torch.add`
- `torch.clamp`：对输入参数按照自定义的范围裁剪，最后将结果输出。
- `torch.div`
- `torch.mul`
- `torch.pow`
- `torch.mm`：矩阵相乘

#### 1.1.1. `Variable` 变量

`Variable` 提供了自动求导的功能。本质上和 `Tensor` 没有区别，不过 `Variable` 会被放入一个计算图中，然后进行前向传播，反向传播，自动求导。

`Variable` 由3个重要的属性组成：
- `data`：读取tensor数值
- `grad`：得到这个`Variable`的操作，比如加减
- `grad_fn`：反向传播梯度

```python
import torch
from torch.autograd import Variable

# Create Variable
x = Variable(torch.Tensor([1], requires_grad=True))
w = Variable(torch.Tensor([2], requires_grad=True))
b = Variable(torch.Tensor([3], requires_grad=True))

# Build a computational graph
y = w * x + b

# Compute gradients
y.backward() # 等价于 y.backward(torch.FloatTensor([1]))

# 当变量是多维的，则需要传入参数声明，即每个梯度分量的数乘

# Get the gradients
print(x.grad)
print(w.grad)
print(b.grad)
```

#### 1.1.2. `Dataset` 数据集

`torch.utils.data.Dataset` 是一个抽象类，可以自己定义你的数据类继承和重写这个抽象类，只需要定义 `__len__` 和 `__getitem__` 这两个函数即可。


```python
import pandas as pd
from torch.util.data import Dataset

class myDataset(Dataset):
    def __init__(self, csv_file, txt_file, root_dir, other_file):
        self.csv_data = pd.read_csv(csv_file)
        with open(txt_file, 'r') as f:
            data_list = f.readlines()
        self.txt_data = data_list
        self.root_dir = root_dir

    
    def __len__(self):
        return len(self.csv_data)

    
    def __getitem__(self, idx):
        data = (self.csv_data[idx], self.txt_data[idx])
        return data
```

可以进一步，通过`torch.util.data.DataLoader`来定义新的迭代器，使用`batch`，`shuffle`或多线程去读取数据。

```python
from torch.util.data import DataLoader

dataiter = DataLoader(myDataset, batch_size=32, shuffle=True, collate_fn=default_collate)  # collate_fn 定义如何取样本
```

#### 1.1.3. `nn.Module` 模组
PyTorch 中的神经网络中的所有层结构和损失函数都来自 `torch.nn`，所有的模型构建都是从这个基类`nn.Module`继承的。

```python
from torch import nn
class net_name(nn.Module):
    def __init__(self, other_arguments):
        super(net_name, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernal_size)
        # Other network layers
    

    def forward(self, x):
        x = self.conv1(x)
        return x
```

定义完模型后，我们需要通过`nn`这个包来定义损失函数。常见的损失函数都已经定义在了`nn`中，比如均方误差，多分类的交叉熵，等等。用这些已经定义好的损失函数：

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)
```

#### 1.1.4. `torch.optim` 优化
优化算法就是一种调整模型参数更新的策略。

1. 一阶优化算法：梯度下降。
2. 二阶优化算法：二阶导数（Hessian 方法）

调用随机梯度下降或者自适应学习率等，需要优化的参数传入，这些参数都必须是`Variable`。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

#### 1.1.5. 模型的保存和加载

两种保存方式：
- 保存整个模型的结构信息和参数信息，保存的对象是`model`

        torch.save(model, './model.pth')

- 保存模型的参数，保存的对象是`model.state_dict()`
        
        torch.save(model.state_dict(), './model_state.pth')

对应两种模型加载方式：
- 加载完整模型结构和参数

        load_model = torch.load('model.pth')
- 加载模型参数信息，需要先导入模型的结构

        model.load_state_dict(torch.load('model_state.pth'))


### 1.2. 线性模型

#### 1.2.1. 线性回归

- `nn.Linear(x, y)`：定义线性模型的输入和输出维度
- `nn.MSELoss()`：定义均方差损失函数
- `optim.SGD(model.parameters(), lr=1e-3)`：定义梯度下降进行优化

每一步的计算可以这样：
```python
out = model(inputs)             # 定义输出
loss = criterion(out, target)   # 定义损失函数

optimizer.zero_grad()           # 每次反向传播之前需要归零梯度，不然梯度会累加，造成结果不收敛
loss.backward()
optimizer.step()
```

`model.eval()` 将模型变成测试模式，Dropout 和 BatchNormalization 等操作在训练和测试时候是不一样的。

#### 1.2.2. 多项式回归

- `torch.cat()` 拼接 Tensor

定义多项式回归模型，只需要使用`nn.Linear(x, y)` 时，令`x>1`即可。

### 1.3. 分类问题

Logistic 分布的积累分布函数为：

$$
F(x) = P(X\le x) = \frac{1}{1 + e^{-(x-\mu)/\gamma}}
$$

$\mu$ 是中心点的位置，$\gamma$越小，中心点附近的增长越快。

Sigmoid 函数是 $\mu=0, \gamma = 1$ 的特殊情况。


#### 1.3.1. 二分类的 Logistic 回归

通过分类概率$P(Y=1)$与输入变量$x$的直接关系，通过比较概率值来判断类别。

$$
P(Y=0|x) = \frac{1}{1+e^{wx+b}} \\
P(Y=1|x) = \frac{e^{wx+b}}{1+e^{wx+b}}
$$

一个事件的发生几率：$\frac{p}{1-p}$，该事件的对数几率是：

$$
logit(p) = \log\frac{p}{1-p}
$$

对于 Logistic 回归模型

$$
\log\frac{P(Y=1|x)}{1 - P(Y=1|x)} = wx + b
$$

也就是，对输出$Y=1$的对数几率是输入$x$的线性函数。

Logisitc 回归的思路是：先拟合决策边界，再建立这个边界和分类概率的关系，从而得到二分类情况下的概率。

采用极大似然估计：

$$
\Pi_{i=1}^n[\pi(x_i)]^{y_i}[1-\pi(x_i)]^{1-y_i}
$$

取对数后为：

$$
L(w) = \sum_{i=1}^{n}[y_i(w\cdot x_i+b) - \log(1+e^{w\cdot x_i+b})]
$$

求导：

$$
\frac{\partial L(w)}{\partial w} = \sum_{i=1}^n [y_i-logit(w\cdot x_i)]\cdot x_i\\
\frac{\partial L(w)}{\partial b} = \sum_{i=1}^n [y_i-logit(w\cdot x_i)]
$$

可以用迭代的梯度下降来求解。

```python
from torch import nn

class LogisticRegression(nn.Modeule):
    def __init__(self):
        super().__init__()
        self.lr = nn.Linear(2, 1)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x
```

损失函数可以使用 `nn.BCELoss()` （Cross-Entropy）

### 1.4. 简单的多层全连接前向网络

Logistic 回归，只是一个使用了Sigmoid 作为激活函数的一层神经网络。

#### 1.4.1. 激活函数

1. Sigmoid
    - 优点：具有良好的解释性
    - 缺点：
        1. 会造成梯度消失，靠近1和0两端，梯度几乎为0.初始化需要非常小心，如果权重过大，会导致大多数神经元变得饱和，无法根新参数。
        2. 输出不是以0为均值。导致后一层网络的输入不是以0为均值的。更新参数的时候，如果神经元是正的，那么下一层的参数更新全为正。相比梯度消失要好很多。

2. Tanh

$$
tanh(x) = 2\sigma(2x) - 1
$$

将输入数据转化到 $[-1, 1]$。输出成为 0 均值。但是依然存在梯度消失问题。

3. ReLU（Rectified Linear Unit）
    - 优点：
        1. 可以极大地加速随机梯度下降法的收敛速率。（线性，无梯度消失）
        2. 计算简单。
    - 缺点：训练时候很脆弱。很大的梯度经过ReLU，会使得这个神经元不会对任何数据有激活现象。若发生这个现象，那么梯度会永远是0，无法更新参数。不可逆。实际操作中会通过设置比较小的学习速率来避免这个问题。

$$
f(x) = max(0, x)
$$

只是见到保留大于0的部分。

4. Leaky ReLU

$$
f(x) = I(x<0)(\alpha x) + I(x\ge 0)
$$

$\alpha$是一个很小的常数，使得输入小于0的时候也有一个小的梯度。

有些实验说好，有些实验说不好。

5. Maxout

$$
max(w_1\cdot x + b_1, w_2\cdot x + b_2)
$$

ReLU 是 $w_1=b_1=0$的特殊形式，maxout 既有ReLU 的特点，又避免了训练脆弱。但是，它加倍了模型的参数，存储变大。

一般在同一个网络中，实际中都使用同一种类型的激活函数，而不混用。

#### 1.4.2. 模型的表示能力和容量

实际中，一个3层全连接网络比一个两层全连接的网络要好，但是，更深的全连接网络就不一定了。

使用大容量网络去训练模型，同时运用一些方法来控制网络的过拟合。


## 2. 卷积神经网络

### 2.1. 卷积模块

1. 卷积层

`nn.Conv2d()` 是 PyTorch 中的卷积模块，常用参数包括：
- in_channels：输入数据体的深度
- out_channels：输出数据体的深度
- kernel_size：卷积核（滤波器）的大小，可用一个数字，或者一对数字
- stride：滑动步长
- padding：填充
- bias：bool值，使用偏置
- groups：输出数据体深度上和输入数据体的联系，1 表示所有输入和输出关联，2表示输入和输出深度都被分割，两两对应。
- dilation：输入数据体的空间间隔

2. 池化层

常用的包括：
- `nn.MaxPool2d()`
- `nn.AvgPool2d()`
- `nn.LPPool2d()`
- `nn.AdaptiveMaxPool2d()`

主要参数：
- return_indices：是否返回最大值的下标
- ceil_mode：使用一些放歌代替层结构


### 2.2. 一些操作

1. 提取层结构

`nn.Module` 的 `children()` 方法，返回下一级模块的迭代器。用于提取前面两层：

```python
new_model = nn.Sequential(*list(model.children())[:2])
```

提取模型中所有的卷积层：

```python
for layer in model.named_modules():
    if isinstance(layer[1], nn.Conv2d):
        conv_model.add_module(layer[0], layer[1])
```

2. 提取参数及自定义初始化

重做权重初始化：

```python
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        init.normal(m.weight.data)
        init.xavier_normal(m.weight.data)
        init.kaiming_normal(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_()
```

### 2.3. 经典案例

1. LeNet：开山之作，7层
2. AlexNet：2012 冠军
3. VGGNet：2014 亚军，使用了更小的滤波器，增加了网络深度。使用很多小的滤波器的感受野和一个大的相同，并且能减少参数。
4. GoogleNet(InceptionNet)：2014 冠军。使用了创新的 Inception 模块。
    - 设计了局部的网络拓扑结构，然后将这些模块堆叠在一起，形成一个抽象的网络。运用几个并行的滤波器对输入进行卷积和池化。
    - 参数太多，计算复杂
5. ResNet：2015 冠军，微软。残差模型可以训练152层的模型。在不断加深网络的时候，会出现一个 Degradation，也就是准确率会先上升，然后达到饱和，再下降。这个不是由于过拟合引起的。
    - 假设一个比较浅的网络达到了饱和准确率，那么在后面加上几个恒等映射层，误差不会增加。即，直接将前一层输出传到后面。


此外，`torchvision.model`里面已经定义了大部分的网络结构，并且都有预先训练好的参数，不需要重复造轮子。

### 2.4. 图像增强的方法

- 光照太弱
- 姿势不同
- 遮挡

`torchvision.transforms` 图像增强的办法：
- `Scale` 放缩
- `CenterCrop` 正中心给定大小的裁剪
- `RandomCrop` 给定大小的随机裁剪
- `RandomHorizontalFlip` 0.5 几率的水平翻转
- `RandomSizedCrop` 先对图像进行随机尺寸裁剪，然后对裁剪的图片进行一个随机比例的缩放
- `Pad` 边界填充

## 3. 循环神经网络

两种使用最多的变种：LSTM，GRU

### 3.1. LSTM

由3个门控制：
- 输入门
- 输出门
- **遗忘门**：决定之前的哪些记忆将被保留，哪些记忆将被去掉。能够自己学习保留多少以前的记忆，而不需要认为干扰，网络就能自主学习。

### 3.2. GRU

Gated Recurrent Unit

将遗忘门和输入门合成了一个“更新门”，同时，网络不再额外给出记忆状态 $C_t$，而是将输出结果$h_t$作为记忆状态不断向后循环传递，网络的输入和输出变得简单。

### 3.3. 收敛性问题

若写一个简单的 LSTM 网络去训练数据，你会发现 loss 并不会按照想象的方式下降，如果运气好的话，能够得到一直下降的 loss，但是大多数情况下 loss 是乱跳的。这是由于RNN的误差曲面粗糙不平造成的，就算使用较小的学习速率也是不行的。

权重在网络中循环的结构里面会不断地被重复利用，那么梯度的微小变化都会在经过循环结构之后被放大。正是因为网络在训练的过程中梯度的变化范围大，所以设置一个固定的学习速率不能有效收敛。同时梯度的变化并没有规律，所以设置衰减的学习率也不能满足条件。

**解决办法**：梯度裁剪（gradient clipping）,将大的梯度裁掉，这样就能在一定程度上避免收敛不好的问题。现在还有很多别的办法解决这个问题。

### 3.4. PyTorch 的循环网络相关模块

![RNN 结构](./figures/PyTorch/RNN.png)

调用 `nn.RNN()`即可，相关参数：

|参数|默认|含义|
|:--|:--|:--|
|input_size||表示输入 $x_t$ 的特征维度|
|hidden_size||$h_t$ 的特征维度|
|num_layers||网络层数|
|nonlinearity|tanh|非线性激活函数，可以是 relu |
|bias|True|是否偏置|
|batch_first|False|网络输入的维度顺序，（seq, batch, feature） True -> (batch, seq, feature)|
|dropout||会在除了最后一层之外的其他输出层上加dropout层|
|bidirectional|False|是否双向循环神经网络|

网络：
1. 序列输入 $x_t$ （长度，批量，特征维度）和记忆输入 $h_0$ （layers*direction, 批量，输出维度）
2. 输出 output （长度，批量，输出维度*方向）和 记忆单元 $h_n$


一个输入维度20，输出维度50的2层单向网络：

    basic_rnn = nn.RNN(input_size=20, hidden_size=50, num_layers=2)

`nn.LSTM()` 和 `nn.RNN()`类似，只是因为 LSTM 比标准 RNN 多了3个线性变换，并且将多的3个线性变换的权重拼接在一起，所以参数一共是rnn的4倍，偏置也是。换句话说，LSTM 做了4个类似标准RNN的运算，所以参数个数是4倍。

而且，LSTM 的输入也不再只有输入序列和隐藏状态，还有 $C_0$。

`nn.GRU()` 也类似，只不过是3倍。

### 3.5. 应用场景

#### 3.5.1. 图片分类

需要将图片数据转化为一个序列数据，比如手写数字图片大小是 $28\times 28$，那么可以将图片看作是长度为28的序列，序列中的每个元素的特征维度是28，这样，图片就变成了一个序列。同时，图片从左往右输入网络的时候，网络的记忆性可以同时和后面的部分结合来预测结果。

```python
class RNN(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layer, n_class):
        super().__init__()
        self.n_layer = n_layer
        self.hid_dim = hid_dim
        self.lstm = nn.LSTM(in_dim, hid_dim, n_layer, batch_first=True)
        self.classifier = nn.Linear(hid_dim, n_class)
    
    def forward(self, x):
        # h0 = Variable(torch.zeros(self.n_layer, x.size(1), self.hid_dim)).cuda()
        # c0 = Variable(torch.zeros(self.n_layer, x.size(1), self.hid_dim)).cuda()

        out, _ = self.lstm(x)
        out = out[:,-1,:] # 输出是一个序列，用最后一个作为结果
        out = self.classifier(out)
        return out
```

但是，循环神经网络不适合处理图片类型的数据。
1. 图片并没有很强的序列关系。因为图片的观看顺序不影响图片信息的解读。
2. 循环圣经网络传递的时候，必须前面一个数据计算结束才能进行后面一个数据的计算，这对于大图片是很慢的。而CNN可以并行。

#### 3.5.2. 自然语言处理

1. 词嵌入（word embdding）

对每个词，可以使用一个高维向量去表示。

`nn.Embdding(2， 5)`

表示 2 个词，用 5 维向量表示。

```python
hello = 1
embeds = nn.Embedding(2, 5)
hello_idx = Variable(torch.LongTensor([hello]))
hello_embd = embeds(hello_idx)
```

2. N-Gram 模型

在一句话中，由上下文来预测单词的方法。

$$
P(T) = P(w_1)P(w_2|w_1)P(w_3|w_1,w_2)\cdots P(w_n|w_{n-1},w_{n-2},\dots,w_1)
$$

- 缺陷：参数空间过大，预测一个词需要前面所有的词作为条件来计算概率，所以在实际中无法使用。
- 办法：引入马尔科夫假设，有了这个假设，就能在实际中使用 N-Gram 模型了。

3. 词性判断

- **原理**：一句话可以看作是一个序列，序列中的每个元素都是一个高维向量，将这个序列传入LSTM，可以得到与序列等长的输出。
- **本质**：分类问题，根据前面的一些词开对它进行分类，看它属于几种词性中的哪一种
- **字符增强**：一些单词存在着前缀或者后缀，这样就能在字符水平上对词性进行进一步判断，把两种方法集成起来，能够得到更好的效果。

4. Many to one

- 情感分析：将一句话作为序列输入网络，输出只取最后一个，判断这句话的态度
- 关键字提取

5. Many to many

输入和输出都是序列，但是输出的序列比输入的短。用在语言识别中。

6. Seq2Seq

输出输入长度不确定，一般在机器翻译中常见。

机器人聊天问答系统。

## 生成对抗网络

### 生成模型（Generative Model）

属于数据概率统计和机器学习范畴，是指一系列用于随机生成可观测数据的模型。“生成”的样本和“真实”的样本尽可能相似。

主要功能：学习一个概率分布 $P_{model}(X)$ 和生成数据。

1. 自动编码器（AutoEncoder）
    - 跟数据相关度很高，只能压缩和训练数据相似的数据
    - 压缩后的数据是有损的，因为在降维的过程中，不可避免信息丢失

应用：

- 数据去噪
- 可视化降维
- 生成数据

2. 变分自动编码器（Variational AutoEncoder）

在编码过程中，给它增加一些限制，迫使它生成的隐含向量能够粗略地遵循一个标准正太分布。

### 生成对抗网络

第一部分是生成模型，第二部分是对抗模型，即判断真假图片的判别器。

1. 生成模型

不是将图片输入编码器得到隐含向量然后生成图片，而是随机初始化一个隐含向量， 根据变分编码器的特点，初始化一个正太分布的隐含向量，通过类似编码器的过程，将它映射到一个更高的维度，最后生成一个与输入数据相似的数据（假图片）。然后，通过对抗过程来计算损失函数。

2. 对抗模型

是一个判断真假的判别器，相当于一个二分类器，和原图的 label 没有关系，不原图片到底是一个多少类别的图片，它们都统一称为真的图片。

训练时：

1. 先训练判别器，将假的数据和真的数据都给判别器，优化模型希望这个判别器能够正确地判断出真假数据。
2. 然后训练生成器，希望它生成的假数据能够骗过现在这个比较好的判别器。将判别器的参数固定，通过反向传播优化生成器的参数，希望生成器得到的数据在经过判别器之后得到的结果尽量接近真。