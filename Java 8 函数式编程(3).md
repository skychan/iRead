# iRead 《Java 8 函数式编程》

## 测试、调试和重构

- 重构遗留代码时，要考虑如何使用Lambda表达式，有一些通用的模式
- 如果想要对复杂一点的Lambda 表达式编写单元测试，将其抽象成一个常规方法。
- ```peek``` 方法能记录中间值，调试时非常有用。

测试驱动开发（Test-driven development, TDD
），持续集成（Continuous integration, CI）。

编写测试单元对Lambda 表达式的日常使用非常重要

### 重构候选项

Lambda 表达式重构也称 Lambdafication，执行重构的程序员也叫 lambdifiers

#### 同样的东西写两遍

DRY(Don't Repeat Yourself) 模式

WET(Write Everything Twice) 模式

比如命令式：

```java
public long countRunningTime() {
    long count = 0;
    for (Album album : albums){
        for (Track track : album.getTrackList()){
            count += track.getLength();
        }
    }
    return count;
}

public long countMusicians() {
    long count = 0;
    for (Album album : albums){
        count += album.getMusicianList().size();
    }
    return count;
}

public long countTracks() {
    long count = 0;
    for (Album album : albums){
        count += album.getTrackList().size();
    }
    return count;
}

```

使用流重构：

```java
public long countRunningTime() {
    return albums.stream()
    .mapToLong(album -> album.getTracks()
        .mapToLong(track -> track.getLength()).sum())
    .sum();
}

public long countMusicians() {
    return albums.stream()
    .mapToLong(album -> album.getMusicians().count())
    .sum();
}

public long countTracks() {
    return albums.stream()
    .mapToLong(album -> album.getTracks().count())
    .sum();
}
```

使用领域方法重构：

```java
public long countFeature(ToLongFunction<Albumn> function){
    return albums.stream().mapToLong(function).sum();
}

public long countRunningTime() {
    return countFeature(album -> album.getTracks().mapToLong(track -> track.getLength()).sum());
}

public long count Musicians() {
    return countFeature(album -> album.getMusicians().count());
}

public long countTracks() {
    return countFeature(album -> album.getTracks().count());
}
```

### Lambda 表达式的单元测试

单元测试是测试一段代码的行为是否符合预期的方式。

Lambda 表达式没有名字，无法直接在测试代码中用。

- 将Lambda 表达式放入一个方法测试。测试方法而不是Lambda 表达式本身
- 不用Lambda表达式，使用方法引用。任何Lambda 表达式都可以改写为普通方法，然后使用方法引用直接引用。单独测试该方法即可。

### 在测试替身时使用Lambda 表达式

编写单元测试的常用方式之一是使用*测试替身*（Test Doubles）描述系统中其他模块的期望行为。测试替身也常被称为*模拟*（Mocks）。测试存根（stubs）和测试模拟都属于测试替身，区别式模拟可以验证代码的行为。

测试代码时，使用Lambda表达式的最简单方式是实现轻量级的测试存根。如果交互的类本身就是一个函数接口，实现这样的存根就非常简单和自然。

```java
@Test
public void canCountFeatures() {
    OrderDomain order = new OrderDomain(asList(
        newAlbum("Exile on Main St."),
        newAlbum("Beggars Banquet"),
        newAlbum("Aftermath"),
        newAlbum("Let it Bleed")
    ));
    assertEquals(8, order.countFeature(album -> 2));
}
```

多数的测试替身都很复杂，使用 Mockito 这样的框架有助于更容易地产生测试替身。

### peek 解决方案

流有一个方法让你能查看每个值，同时能继续操作流。

```java
Set<String> nationalities
    = album.getMusicians()
        .filter(artist -> artist.getName().startsWith("The"))
        .map(artist -> artist.getNationality())
        .peek(nation -> System.out.println("Found nationality: " + nation))
        .collect(Collectors.<String>toSet());
```

可以在peek 方法中加入断点。