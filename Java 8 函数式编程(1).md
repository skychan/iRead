# iRead 《Java 8 函数式编程》

## 简介

编写让代码在多核CPU上高效运行的方式，处理批量数据的并行类库，需要：增加 Lambda 表达式。

面向对象编程是对数据的抽象，而函数式编程是对行为的抽象。现实世界中，数据和行为并存，程序也是如此。

核心：在思考问题时，使用不可变值和函数，函数对一个值进行处理，映射成另一个值。

## Lambda 表达式

- Lambda 表达式是一个匿名方法，将行为向数据一样传递
- Lambda 表达式常见结构： ```BinaryOperator<Integer> add = (x,y) -> x+y;```
- 函数接口（functional interface）仅指具有单个抽象方法的接口，用来表示 Lambda 表达式的类型。

不同的形式：
```java
    Runnable noArguments = () -> System.out.println("Hello World");

    ActionListener oneArgument = event -> System.out.println("button clicked");

    Runnalbe multiStatement = () -> {
        System.out.print("Hello");
        System.out.print(" World");
    }

    BinaryOperator<Long> add = (x,y) -> x + y;

    BinaryOperator<Long> addExplict = (Long x, Long y) -> x+y;

```

Lambda 表达式中的参数类型是由编译器推断得的，但有时也最好可以显示声明参数类型，此时就需要使用小括号将参数括起来。

**引用值，而非变量**

要用终态变量（```final```），或者是值，不能用变量。

这也是Lambda 表达式称为闭包的原因。未赋值变量与周围环境隔离起来，进而被绑定到一个特定的值。

### 函数接口

函数接口是只有一个抽象方法的接口，用作Lambda表达式的类型。最开始也就做SAM类型的接口（Single Abstract Method）

接口中单一方法的命名并不重要，只要方法签名和 Lambda 表达式的类型匹配即可。

函数接口可以接受两个参数，并返回一个值，还可以使用泛型，这完全取决于你要干什么。


|Java 8 重要接口|参数|返回类型|
|:---|:---|:---|
|Function<T,R>|T|R|
|Predicate\<T>|T|boolean|
|Consumer\<T>| T | void|
|Supplier\<T>|None|T|
|UnaryOperator\<T>|T|T|
|BinaryOperator\<T>|(T,T)|T|

### 类型推断
Java 7 中程序员可以省略构造函数的泛型类型，Java 8 更进一步，可以省略Lambda 表达式中的所有参数类型。（构造函数直接传递给一个方法在 Java 7 中不能编译通过）

## 流

- 内部迭代将更多控制权交给了集合类。
- Iterator 和 Stream 类似，是一种内部迭代方式。
- 将Lambda 表达式和Stream 上的方法结合，可以完成很多常见的集合操作。

### 从外部迭代到内部迭代

例如，使用集合类时，通常会在集合上进行迭代，然后处理返回的每一个元素。每次迭代集合类时，都需要写很多样板代码。而且将```for```改造成并行运行也很麻烦，需要修改每个for 循环才能实现。

for 循环式一个语法糖，首先调用```iterator``` 方法，产生一个新的```Iterator``` 对象，进而控制整个迭代过程，这就是*外部迭代*。迭代过程通过显示调用```Iterator```对象的```hasNext, next```方法完成迭代。

外部迭代本质上来说是一种串行化操作，使用for 循环会将行为和方法混为一谈。

内部迭代不返回```Iterator```对象，而是返回内部迭代中的相应接口：```Stream```

```java
    long count = allArtists.stream()
    .filter(artist -> artist.isFrom("Londn"))
    .count();
```

内部迭代在应用代码上构建操作（行为），而集合代码管理方法。

**Stream 是用函数式编程方式在集合类上进行复杂操作的工具**

### 实现机制

```filter```这样只描述Stream 而最终不产生新集合的方法叫做*惰性求值方法（Lazy evaluation）*

```count```这样最终会从Stream产生值的方法叫做*及早求值方法（Eager evaluation）*

### 常用流操作

|流操作|解释|接口|
|:----|:--|:--|
|```collect(toList())```|及早求值，由Straeam 里的值生成一个列表|无|
|```map```|转换流的类型|Function|
|```filter```|遍历数据并检查其中元素时，可以尝试|Predicate|
|```flatMap```|可用Stream 替换值，将多个Stream 接成一个|Function|
|```max,min```|要考虑排序指标，要传入Comparator 对象*，返回Optional对象，要用```get()```方法取出|Functional|
|```reduce```|```count,min,max```都是该操作|BinaryOperator|

Java 8 的```Comparator.comparing```可以方便的实现一个比较器。
可以调用空Stream 的```min,max```方法，放回 Optional 对象（代表一个可能存在也可能不存在的值）

通用模式（for 伪代码）
```java
    Object accumulator = initialVale；
    for(Object element : collection){
        accumulator = combine(accumulator, element);
    }
```
accumulator 是累计器，记录当前最新的要求值，如求和、较大、较小、计数等。

reduce 求和：
```java
    int count = Stream.of(1,2,3)
    .reduce(0, (acc, element) -> acc + element);
```

整合操作：首先分解问题，然后找出每一步对应的Stream API就相对容易了。使用List 或 Set 的stream 方法就能得到一个Stream 对象。设计时，对外仅暴露一个Stream 接口，用户在实际操作中无论如何使用，都不会影响内部的List 或Set，能很好地封装内部实现的数据结构。

### 重构遗留代码

- 使用List 或Set 的```stream()```方法，使用Stream 的 ```forEach``` 替换```for```循环
- 转化和替换代码用```map```或更加复杂的```flatMap```
- 例如将遍历的每一个元素整合转化为一个Stream
- 编写的每一步都要进行单元测试，保证代码能正常工作

### 多次调用流（Multiple Stream Calls）

用户也可以选择每一步强制对函数求值，而不是将所有方法调用链接在一起（链式调用，Chained Stream Calls），但是最好不要如此操作。

多次调用的缺点：

- 代码可读性插，隐藏了真正的业务逻辑
- 效率差，都及早求值
- 充斥一堆垃圾变量，它们只用来保存中间结果，除此之外毫无用处
- 难于自动并行化处理。

### 高阶函数（Higher-order function）

高阶函数：接受了另外一个函数作为参数，或返回一个函数的函数。

看签名：如果函数的参数列表里包含函数接口，或者该函数返回一个函数接口，那就是高阶函数。

```map```是一个高阶函数，因为它的```mapper```参数是一个函数。

```Comparator``` 是一个函数接口。

### 正确使用Lambda 表达式

编写的函数最好不能有副作用，也就是改变程序或外界状态。

如向控制台输出信息的*可观测到的*副作用可以。给变量赋值是一种难以察觉的副作用，但是它的确改变了程序的状态。设计者鼓励使用Lambda 表达式获取值而非变量。获取值使用户更容易写出没有副作用的代码。

将Lambda 表达式传给Stream 上的高阶函数，都应尽量避免副作用，唯一的例外是```forEach```方法，它是一个终结方法。
