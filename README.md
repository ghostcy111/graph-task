# graph-task

## task1 节点分类
### 数据集描述
使用Cora, Citeseer数据集

Cora 数据集：
包含 2708 篇科学出版物，有 5429 条边用于表示论文之间的引用关系。
论文主题分为 7 类，分别是基于案例、遗传算法、神经网络、概率方法、强化学习、规则学习和理论。
每个出版物由一个 0/1 值的词向量描述，该词典由 1433 个独特的词组成，即特征维度为 1433 维，1 表示对应词在论文中出现，0 表示未出现。
主要包含 cora.content 和 cora.cites 两个文件。前者包含论文描述，每行是论文编号、1433 维词向量和论文类别；后者包含论文引用关系，每行有两个论文编号，表示引用与被引用关系。
广泛用于文本分类、图神经网络等领域的研究，帮助研究者探索如何利用文本信息和引用网络结构进行论文分类。

Citeseer 数据集：
由 3327 篇学术论文组成，形成了包含 4732 条边的引用网络。
论文分为 6 类，分别是 Agents、AI、DB、IR、ML 和 HCI。
每个节点（论文）有一个 3703 维的二进制特征向量，采用词袋模型表示，值为 1 表示对应词在论文中出现，为 0 表示未出现。
与 Cora 类似，包含 citeseer.content 和 citeseer.cites 两个文件，前者记录论文 ID、特征向量和类别标签，后者记录论文间引用关系。
是图神经网络研究中常用的基准数据集，特别是用于节点分类任务。

### 实验结果
<img width="4800" height="1800" alt="Citeseer_comparison" src="https://github.com/user-attachments/assets/818d48b6-eb4f-4fba-8d5a-bbab60ffbab6" />
<img width="4800" height="1800" alt="Cora_comparison" src="https://github.com/user-attachments/assets/ac2d2147-4a92-4649-a9d9-7243d574e51c" />

