# Java高频面试问题总结

---



## Java基础

> 包括JavaSE、多线程、并发、集合和JVM等常见问题

**（1）java中有哪些基本数据类型？**

+ 整数类型：byte short int long

- 浮点数类型：float double 

- 字符型：char

- 布尔型：boolean
- 注意：java还为每一种基本类型提供了相应的包装类，基本类型与包装类型的主要区别在于1、包装类型允许值为空，而基本类型不允许为空；2、包装类型是一个对象，需要消耗更多的内存，也给GC带来了更大的压力，所以在性能上稍差一截。

**（2）如何控制多个线程按一定顺序执行？** 

- 1 使用join()方法，伪代码如下：

```java
Thread thread1 = new Thread(new Runnable(){
    @Override
    public void run(){
        //自定义代码块
    }
})

Thread thread2 = new Thread(new Runnable(){
    @Override
    public void run(){
        threa1.join()  //保证该线程在thread1之后执行
        //自定义代码块
    }
})

```

- 2 使用对象的wait()和notify()方法
- 3 使用线程池的submit()方法，伪代码如下：

```java
ThreadPoolExecutor threadpool = new ThreadPoolExecutor(各个初始参数);
Thread thread1 = new Thread(线程对象参数);
Thread thread2 = new Thread(线程对象参数);
Thread thread3 = new Thread(线程对象参数);
threadpool.submit(thread1);
threadpool.submit(thread2);
threadpool.submit(thread3);
```

- 4 使用ReentrantLock结合condition控制线程顺序执行
- 5 使用CountDownLatch(递减，只能使用一次)或者CyclicBarrier(递增，可重复利用)

- 6 使用Semaphone信号量控制线程顺序执行

**（3）创建线程有哪几种方式？**

- 1 继承Thread类并重写run()方法
- 2 实现Runnable接口并重写run()方法
- 3 实现 Callable接口并重写call()方法
- 4 使用线程池的方式

**（4）实现线程安全的方式有哪些？**

- 1 使用synchronized关键字（包括同步方法和同步代码块）
- 2 使用锁（ReentrantLock）
- 3 使用volatile关键字
- 4 使用原子类

**（5）什么时候会触发GC？**

- 当堆中的新生代中的Eden区没有足够的空间为新的对象分配内存的时候会触发MinorGC
- 当新生代中的S0和S1中相同年龄的对象所占内存超过S0和S1总内存的一半的时候会触发MinorGC
- 当老年代中无法为S1中的对象腾出新的内存的时候会触发fullGC
- 当老年代中无法为新的大对象分配内存的时候会触发fullGC

**（6）同一个类中，普通方法和静态方法在获取锁上有什么区别?**

* 普通方法指向的锁是this，亦即对当前类上锁
* 静态方法指向的锁是this.class，亦即对当前类的实例上锁









## 框架和中间件

> 包括Netty、Redis、kafka、spring、springboot等框架和中间件相关的常见问题













## 数据结构及计算机网络

> 包括数据结构和常用算法、计算机网络基础知识、常用网络协议等相关知识问题









## 数据库

**（1）如何优化数据库性能？**

> 个人认为，优化数据库的性能的目的主要是为了提高SQL响应速度。所以，数据库性能的优化可以从下面的步骤进行：

- 1 首先是找到那些SQL响应慢的语句，然后检查是否可以在程序代码中做优化；
- 2 通过数据库的一些检测工具检查SQL慢的原因，如慢查询日志、show status、show global status、show profiles、explain等工具。然后根据检测结果来一一排查问题
- 3 考虑是否系统中的缓存是否部分失效或挂掉，系统中是否存在外部调用延时从而导致SQL响应时间变慢
- 4 考虑是否可以通过创建合适的索引来提升查询效率
- 5 数据表的结构是否可以做进一步的优化，如数据字段类型优化，垂直分表等
- 6 数据库系统的参数配置是否合理，操作系统的配置是否合理
- 7 服务器的性能是否受到其他程序的影响
- 8 如果是由于数据表数据量过大造成的，那么是否可以考虑水平分表，分库，分布式集群等。

**（2）数据库中常见的索引有哪些？**

- 1 按照索引的数据结构来划分的话有：B-tree索引及其变体、hash索引、R-tree索引、K-D树索引、全文索引和倒排索引等
- 2 对于使用基于B+tree的Innodb引擎的MySQL来说有如下索引：主键索引、唯一索引、联合索引（包含多个列的索引）、前缀索引、后缀索引、覆盖索引及自适应哈希索引。

**（3）聚簇索引和非聚簇索引的区别是什么？**

- 聚簇索引根据索引列的值来有序地紧凑存储数据行记录，这意味着通过聚簇索引列来查询数据是非常快的，因为不需要做回表查询就可以直接拿到整行记录的所有值；
- 非聚簇索引则只存储该索引列的键值，所以如果使用非聚簇索引来查找行记录，那么首先能拿到只有该索引列的键值，然后再回表查询其余列的值，进而得到整行记录的所有值；
- 在MyIsam存储引擎中，是不存在聚簇索引的，即使是主键索引也不属于聚簇索引，主键索引和其他索引没什么区别，只是名字不同罢了。MyIsam存储引擎底层使用B-Tree实现索引，它的每个索引节点都存储了指向对应索引列键值的指针作为节点的值（也就是说，节点只缓存索引，真正的数据需要通过操作系统根据指针调用来找到）所以它需要两次索引才能找到找到需要查找的记录；
- 在Inodb中，支持以主键为索引列的聚簇索引，并且一张表只能有一个聚簇索引（所以一张表只能有一个主键）。建议使用自增的列作为主键，这样在数据表插入数据的时候才能有序插入到内存页从而有序的写入到磁盘，在取数据的时候才能按顺序从磁盘加载到内存中，即将随机IO变为了顺序IO；当然，还建议主键列的字段应该不要太大，因为越小，则一个数据页能存储的数据就越多，从而可以减小IO，进而提高查询效率。