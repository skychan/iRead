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

<!-- /TOC -->
## 1. 多层全连接神经网络

### 1.1. PyTorch 基础

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