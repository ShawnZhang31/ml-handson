<!--
 * @Author: shawnzhang
 * @Date: 2020-09-14 21:45:33
 * @LastEditors: shawnzhang
 * @LastEditTime: 2020-09-16 00:23:10
 * @FilePath: /ml_scikit_torch/docs/source/机器学习概览/ml_overview.md
 * @Description:机器学习概览
-->
# 1. 机器学习概览
## 什么是机器学习
机器学习是研究如何让计算机不需要明确的程序也具备学习能力。
<p align="right">---Arthur Samuel, 1959</p>

## 为什么要使用机器学习
- 对于那些现有解决方案需要大量手动调整或者是规则列表超长的问题：通过一个机器学习的算法就可以简单简化代码，并且提升执行表现。
- 对于那些传统技术手段无法解决的负责问题：通过最好的机器学习技术可以找到一个解决方案。
- 对于环境波动：机器学习系统可以适应新的数据。
- 从负责问题和海量数据中获得洞见。

## 机器学习的种类
- 是否在人的监督下训练
    - 监督学习
    - 无监督学习
    - 半监督学习
    - 强化学习
- 是否可以动态地进行增量学习
    - 在线学习
    - 批量学习
- 是简单的将新的数据点和已知的数据点进行匹配，还是像科学家那样，对训练数据进行模式检测，然后建立一个预测模型
    - 基于实例的学习
    - 基于模型的学习

### 监督式/无监督式学习
- 监督式学习
    - K-近邻算法
    - 线性回归（Linear Regression）
    - 逻辑回归（Logistic Regression）
    - 支持向量机（Support Vector Machines, SVM）
    - 决策树与随机森林（Decision Trees and Random Forests）
    - 神经网络（Neural Networks）
- 无监督式学习
    - 聚类算法
        - k-平均算法（k-Means）
        - 分层聚类分析（Hierachical Cluster Analysis, HCA）
        - 最大期望算法（Expectation Maximization）
    - 可视化和降维
        - 主成分分析（PCA）
        - 核主成分分析（Kernel PCA）
        - 局部线性嵌入（LLE）
        - t-分布随机近临嵌入（t-SNE)
    - 关联规则学习
        - Apriori
        - Eclat
- 半监督式学习
    - 深度信念网络（DBN）
        - 深度信念网络是一种互相堆叠组件，这个组件叫做**受限波尔兹曼机（RBN）**
        - 受限波尔兹曼机（RBN）是以无监督的方式进行训练的，然后使用监督式学习对整个系统进行微调
- 强化学习
    - *强化学习*是一个非常与众不同的巨兽。它的学习系统（在其语境中称为智能体）能够观察环境，做出选择，执行操作，并获得回报，或是以负面回报的形式获得惩罚。所以它必须自行学习什么是最好的策略，从而随着时间推移获得最大的回报。
    - 许多机器人是通过强化学习学习如何行走的。
    - DeepMind项目的AlaphGo项目也是一个强化学习的例子。
### 批量学习和在线学习
**这是另外一个给机器学习系统分类的标准，是看系统是否可以从传入的数据流中进行增量学习。**
#### 批量学习
在批量学习中，系统无法进行增量学习。
#### 在线学习
在在线学习中，可以循序渐进的给系统提供训练数据，逐步积累学习成果。
> 在线学习的整个过程是离线完成的（也就是不再live系统上），因此在线学习这个名字容易让人产生误解。我们可以将其视为**增量学习**。

在线学习的一个重要参数是**学习率**，如果设置过大，系统会迅速适应新数据，但同时也会忘记旧数据；如果设置的过小，系统会有惰性。

**在线学习面临一个重大挑战是，如果给系统输入不良数据，系统的性能将会逐渐下降！**

### 基于实例和基于模型的学习
这种分类方法是看系统如何泛化的。而泛化的主要方法有两种：基于实例的学习和基于模型的学习。

#### 基于实例的学习
系统完全死记硬背记住学习实例，然后通过某种相似度度量方式将其泛化到新的实例。

#### 基于模型的学习
从一组数据集中构建出模型，然后使用该模型进行预测，这就是基于模型的学习。

## 机器学习的挑战
- 训练数据的数量不足
- 训练数据不具有代表性
- 质量差的数据
- 无关特征
- 训练数据过度拟合
- 训练数据拟合度不足

## 确保你可以回答以下问题
### 1. 你会如何定义机器学习？
<details>
    <summary>
        <strong style="color:blue">点击显示答案</strong>
    </summary>
    <p style="color:green">
        机器学习是一门能够让系统从数据中学习的计算机科学。
    </p>
</details>

### 2. ML在哪些问题上表现突出，举出四种类型？
<details>
    <summary>
        <strong style="color:blue">点击显示答案</strong>
    </summary>
    <p style="color:green">
        机器学习非常有利于：不错在已知算法解决方案的复杂问题，需要大量手动调整或是规则列表超长的问题，创建可以适应环境波动的系统，以及帮助人类学习（如数据挖掘）
    </p>
</details>

### 3. 什么是被标记的训练数据集？
<details>
    <summary>
        <strong style="color:blue">点击显示答案</strong>
    </summary>
    <p style="color:green">
        被标记的训练集是指包含每个实例所期望的解决方案的训练集
    </p>
</details>

### 4. 最常见的两种监督式学习任务？
<details>
    <summary>
        <strong style="color:blue">点击显示答案</strong>
    </summary>
    <p style="color:green">
        回归和分类
    </p>
</details>

### 5. 列举四种常见的无监督式学习任务？
<details>
    <summary>
        <strong style="color:blue">点击显示答案</strong>
    </summary>
    <p style="color:green">
        聚类、可视化、降维以及关联规则学习
    </p>
</details>

### 6. 要让机器人在各种未知地形行走，你会使用什么类型的ML算法？
<details>
    <summary>
        <strong style="color:blue">点击显示答案</strong>
    </summary>
    <p style="color:green">
        强化学习
    </p>
</details>

### 7. 将客户分成多组，你会使用什么类型的算法？
<details>
    <summary>
        <strong style="color:blue">点击显示答案</strong>
    </summary>
    <p style="color:green">
        如果不知道如何定义，可以使用聚类算法；如果知道想要什么样的群组，可以将每个组的多个示例反馈给算法（监督学习）
    </p>
</details>

### 8. 你会将垃圾邮件分类问题列出监督式学习还是无监督式学习？
<details>
    <summary>
        <strong style="color:blue">点击显示答案</strong>
    </summary>
    <p style="color:green">
        典型的监督学习场景
    </p>
</details>

### 9. 什么是在线学习？
<details>
    <summary>
        <strong style="color:blue">点击显示答案</strong>
    </summary>
    <p style="color:green">
        在线学习系统可以进增量学习，与批量学习系统正好相反。这使得它能够快速适应不断变化的数据和自动化系统，并且可以在大量的数据上进行训练
    </p>
</details>

### 10. 什么是核外学习？
<details>
    <summary>
        <strong style="color:blue">点击显示答案</strong>
    </summary>
    <p style="color:green">
        核外算法可以处理计算机主内存无法应对的大量数据。他将数据分割成小批量，然后使用在线学习技术从这些小批量中学习。
    </p>
</details>

### 11. 什么类型的学习算法依赖相似度来做出预测？
<details>
    <summary>
        <strong style="color:blue">点击显示答案</strong>
    </summary>
    <p style="color:green">
        基于实例的学习系统
    </p>
</details>

### 12. 模型参数和学习算法中的超参数有什么区别？
<details>
    <summary>
        <strong style="color:blue">点击显示答案</strong>
    </summary>
    <p style="color:green">
        模型参数有一个或多个参数，这些参数决定了模型对新的给定的实例会做出怎样的预测。学习算法试图找到这些参数的最优解，使得模型能够很好的泛化至新实例。超参数是学习算法本身的参数，不是模型的参数（比如，要应用的正则化参数）
    </p>
</details>

### 13. 基于模型的学习算法搜索的是什么？它们最常用的策略是什么？它们如何做出预测的？
<details>
    <summary>
        <strong style="color:blue">点击显示答案</strong>
    </summary>
    <p style="color:green">
        基于模型的算法搜索的是使模型泛化最佳的模型参数。通常使用成本(loss)函数最小话进行训练这样的系统，成本函数衡量的是系统对训练数据的预测有多坏，如果模型由正则化，则再加上一个对模型复杂度的惩罚。学习算法最终找到的参数值就是最终得到的预测函数，只需要将实例特征提供给这个预测函数即可进行预测。
    </p>
</details>

### 14. 提出ML中的四个主要挑战？
<details>
    <summary>
        <strong style="color:blue">点击显示答案</strong>
    </summary>
    <p style="color:green">
       数据缺乏、数据质量差、数据不具有代表性、特征不具信息量、模型过于简单对训练数据欠拟合、模型过于复杂对训练数据过拟合
    </p>
</details>

### 15. 如果你的模型在训练集上表现很好，但是应用到实际数据上泛化能力却很差，是怎么回事，提出三种可能解决问题的方案？
<details>
    <summary>
        <strong style="color:blue">点击显示答案</strong>
    </summary>
    <p style="color:green">
       模型在训练数据上过拟合了，可能的解决方案：获取更多的数据、简化模型（选择更简单的算法、减少使用的参数和特征数量、对模型进行正则化），或者是减少数据训练中的噪声、提前结束训练
    </p>
</details>

### 16. 什么是测试集，为什么要使用测试集？
<details>
    <summary>
        <strong style="color:blue">点击显示答案</strong>
    </summary>
    <p style="color:green">
       在模型启动至生产环境之前，使用测试集来估算模型在新的实例上的泛化误差
    </p>
</details>

### 17. 验证集的目的是什么？
<details>
    <summary>
        <strong style="color:blue">点击显示答案</strong>
    </summary>
    <p style="color:green">
       用来比较不同的模型。可以用来选择最佳模型和调整超参数
    </p>
</details>

18. 如果使用测试集调整超参数会出现什么问题？
<details>
    <summary>
        <strong style="color:blue">点击显示答案</strong>
    </summary>
    <p style="color:green">
       使模型在测试集上过拟合，最后测量的泛化误差过于乐观
    </p>
</details>

### 19. 什么是交叉验证，为什么只验证集更好？
<details>
    <summary>
        <strong style="color:blue">点击显示答案</strong>
    </summary>
    <p style="color:green">
       通过使用交叉验证技术，可以不需要单独的验证集实现模型比较，这节省了宝贵的训练数据。
    </p>
</details>



