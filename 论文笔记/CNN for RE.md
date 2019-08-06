- # **Relation Classifification via Convolutional Deep Neural Network**

---

**关系抽取简介**：

关系抽取的任务是从给定的带有名词或实体标注的一句话中识别出它们之间的关系，如“*比尔盖茨在美国创建了微软公司“，那么得到的结果应该是这样的——founder：（比尔盖茨，微软公司）

关系抽取可以分为限定域关系抽取和开放域关系抽取。过去和现有的大部分研究都是对于限定域的关系抽取，而限定域的关系抽取更多的是被看作为一种多分类问题

限定域关系抽取的流程大致如下：1）首先是要用后大量的训练数据，这些数据都具有一定的格式和标注，然后还要预先定义一些关系类型  2）从大量的训练数据中学习特征并训练成模型  3）利用模型对需要抽取关系的测试句子进行关系分类过预测。因此限定域关系抽取通常被视为多分类任务，或者也可以看作是对句子的目标名次分配特定的关系。

---

**本paper的主要思想和贡献如下**：

1. 首次使用卷积深度神经网络来抽取句子的词汇级特征和句子级特征，而不需要依赖于现有的NLP工具，从而大大减少了传递误差
2. 提出位置向量来编码目标名次对之间的相对距离，以更精准的为目标名次对分配关系
3. 使用SemEval2010Task8的数据进行了实验，结果显示位置向量、词汇级特征和句子级特征对关系分类效果很好，paper所提出的方法和模型比现有的其他方法效果要好很多

---

**研究现状**：

1. 基于传统机器统计学习的方法：基于特征表示的方法和基于核函数的方法
2. 有监督学习方法（需要大量的人工标注的数据集）、无监督学习方法(使用到了语境特征，主要基于分布式假设理论)
3. 远程监督的方法（数据远程对齐知识库如Freebase）

现有的基于深度学习来学习特征的方法（词嵌入）需要依赖于传统的NLP技术，如句法分析、词性标注和依存分析等，而现有的NLP工具中的这些技术都不是很成熟，效果不是很理想，所以这样就难免会对特征产生传递性的误差从而影响最终的结果。

---

**本文方法**：

**结构图**：首先把两个结构图列出来在说，有图有真相嘛

![1564721355185](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\1564721355185.png)

![1564560246695](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\1564560246695.png)

总的大致流程是这样的：首先是输入一个包含了名次对的句子（paper强调他们的方法对于输入的句子不需要进行任何的复杂的句法分析或语法分析处理，因为这是他们的方法的一个亮点）；然后是利用词嵌入将句子中的每个词用向量表示；然后再将这些向量送到卷积DNN中进行一顿操作从而提取出词汇特征和句子级特征（至于词汇级特征和句子级特征具体是什么后面简述）；然后就是将词汇级特征向量和句子级特征向量结合成为最终的特征向量，最后，为了得到每个相关关系的信任度或分数，再将这个复合向量喂给一个softmax 分类器；分类器的输出是一个还是一个向量，向量的维数是最近的关系的数目，并且每一维上的值就是对应关系类型的信任度。

**word representation**：词表示（词向量化）使用的是前人已经训练好的词向量，因为要自己训练词嵌入的话是分词耗时间的（因为你需要大量的数据来训练，数据少了的话学习到的特征不够不好从而影响词表示的语义效果）

**lexical level feature**:词汇级特征是关系分类的重要线索，这里使用了名次标记和上下文标记的词嵌入，甚至还考虑了名次的上位词特征

![1564562604371](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\1564562604371.png)

**sentence level feature**：将每个词表示成向量还不够，因为一句话可能很长，为了捕捉到长距离词之间的语义特征，所以需要提取句子级特征，这里paper使用最大池化卷积深度神经网络来自动提取句子级特征表示，结构图中的figure2展示了其工作原理。

> word features：词特征的表示，这里举个例子好理解。有一句话：
>
> >  “[start] people have been moving back into downtown [end]"
>
> 下一步就是设定一个窗口大小，比如这里设置为3，那么这一句话的词特征WF表示即为：{ [start,people,have],[people,have,been]......[into,downtown,end] }

> position features：单单只有词特征是远远不够的，因为它只能捕捉局部信息，当我们需要句子的某些结构特征时，比如名词之间的结构信息。这是就需要引入位置特征来为关系分类更好地服务。位置特征PF是当前词w与目标词之间的相对距离，比如对于上面的那句话，moving(w)与people(w1)之间的相对距离为3，moving与downtown(w2)之间的相对距离为-3（可以用左加右减来记忆）。然后PF要被映射到一个为d维的向量中（d是一个超参数），这个向量被随机地初始化，然后可以得到两个向量d1和d2（分别为w到w1的距离向量和w到w2的距离向量），最后得到PF=[d1,d2]，将词特征与位置特征结合得到词的最终表示为[WF,PF]的转置。接下来步骤就交给convolution部分并得到句子级特征向量。

**convolution and output**：

将上面一步得到的所有词的最后特征向量通过CNN处理结合起来，其原理主要是对特征向量进行一个线性矩阵变换，然后再对得到的矩阵的每一行取最大值从而得到一个结合后的向量；得到的新的向量再送入softmax classifier ，最后得到一个关系信任度值向量；

---



**实验数据**：

实验数据中的关系类别有九个明确的关系类型和一个未定义关系类（other），如Entity-Destination、Product-Producer
Entity-Origin、Cause-Effect等，

训练数据和测试数据都是带实体标识的，其格式如：

> The most common <e1>audits</e1> were about <e2>waste</e2> and recycling."
> "The <e1>company</e1> fabricates plastic <e2>chairs</e2>."
> "The school <e1>master</e1> teaches the lesson with a <e2>stick</e2>."

有关系的两个实体或名词之间是存在顺序关系的，比如Cause-Effect（entity1，entity2）和Cause-Effect（entity2，entity1）是不一样的，实验的时候是要区分出来的。

**实验结果**：

paper做了三个实验。第一个实验是关于超参数的选择对关系分类性能的影响；第二个实验是关于paper所提出的CDNN方法提取的特征用于关系分类的效果与其他方法做一个对比；第三个实验是分析每个提取的特征对关系分类所取得的效果。

第一个实验结果：![1564647005213](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\1564647005213.png)

![1564647056492](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\1564647056492.png)

第二个实验结果：

![1564647550224](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\1564647550224.png)

第三个实验：

![1564647584372](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\1564647584372.png)