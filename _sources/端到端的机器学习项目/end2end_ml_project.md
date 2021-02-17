<!--
 * @Author: shawnzhang
 * @Date: 2020-09-16 21:02:00
 * @LastEditors: shawnzhang
 * @LastEditTime: 2020-09-16 21:45:35
 * @FilePath: /ml_scikit_torch/docs/source/端到端的机器学习项目/end2end_ml_project.md
 * @Description: 端到端的机器学习项目
-->
# 2. 端到端的机器学习项目

## 大局观
一个端到端的机器学习项目经历的步骤主要:
1. 观察大局
2. 获得数据
3. 从数据探索和可视化中获得洞见
4. ML算法的数据准备
5. 选择和训练数据
6. 微调模型
1. 展示解决方案
1. 启动、监控和维护数据

**使用真实的数据**

- 流行的开放数据存储库
    - [UC Irvine Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php)
    - [Kaggle datasets](https://www.kaggle.com/datasets)
    - [Amazon's AWS datasets](https://aws.amazon.com/fr/datasets)
- 元门户点
    - http://dataportals.org/
    - http://opendatamonitor.eu/
    - http://quandl.com/
- 其他一些列出许多流行的开放数据存储库的地址
    - [维基百科ML datasets](https://goo.gl/SJHN2k)
    - [Quora.com question](https://goo.gl/zDR78y)
    - [Datasets subreddit](https://www.reddit.com/r/datasets)



**知识点：**
- 一种常见的分布是呈钟形态的分布，称为正态分布（也叫高斯分布），“68-95-99.7”的规则是指：大约68%的值落在$1\sigma$内，95%落在$2\sigma$内，99.7%落在$3\sigma$内
- 当数据有很多离群区域时，可以考虑使用**平均绝对误差-MSE**代替**均方根误差-RMSE**
- 均方根误差-RMSE对应欧几里得范数，也称为$l_2$范数，记作$||.||_2$
- 平均绝对误差-MSE对应$l_1$范数，记作$||.||_1$，有时也被称为**曼哈顿距离**
- 范数指数越高，越关注大的价值，忽视小的价值。这就是为什么RMSE比MAE对异常值更敏感的原因。当异常值非常稀少（例如钟形曲线）时，RMSE的表现优异，通常作为首选。
- 我们在很多设置随机种子的代码中，都会看到`np.random.seed(42)`，其中的42数字并没有特殊属性，只是“关于生命、宇宙和一切终极问题的答案”而已（来自《银河系搭车客指南》）
- 使用[`sklearn.model_selection.StratifiedShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html)进行分层抽样


## 创建测试集
**数据窥探偏误(data snooping bias)**: 大脑是一个非常神奇的模式检测系统，也就出说它很容易过拟合：如果你本人浏览测试数据集，你很可能会跌入某个看似有趣的数据模式，进而选择某个特殊的机器学习模型。然后再使用模型对泛化误差率进行估算的时候，估计结果将会过于乐观。


## 实例代码文件
<!-- [housing jupyter notebook](../../code/chapter1/housing.ipynb) -->
