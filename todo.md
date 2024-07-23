- [x] 损失函数
- [x] 数据增强
- [ ] DetectionCL代码
- [ ] 训练主逻辑

# 计算 Wasserstein 距离

## 什么是最优传输问题
**移动概率质量函数**
把离散的概率分布想象成空间中分散的点的质量。我们可以观测这些带质量的点从一个分布移动到另一个分布需要做多少功。
![alt text](image.png)

定义一个度量标准，衡量移动所有点需要做的功。
=>引入耦合矩阵P（Coupling Matrix），表示从p(x)支撑集中的一个点到q(x)支撑集的一个点需要分配多少概率质量。对于均匀分布，规定每个点都有1/4的概率质量。

计算质量分配的过程需要做多少功=>引入距离矩阵
=>距离可以采用欧几里得距离

总成本计算
=>计算把点从一个支撑集移动到另一个支撑集中，使得成本较小的分配方式。

最优传输问题
=>解是所有耦合矩阵上的最低成本


## 生成数据的prompt
为了充分利用有限的训练数据并提高模型在表面缺陷检测任务中的表现，使用一种动态混合数据增强策略，结合CutPaste增强和MixUp增强。
请帮我写一个深度学习的数据加载工具，要求如下：
1. 数据的调用方式实例
"""
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

train_dataset = RandomImageFolderInstance(
        traindir,
        TwoCropsTransform(transforms.Compose(augmentation)))
"""
其中，RandomImageFolderInstance为你需要写的方法，TwoCropsTransform为数据增强方式，已写好。

2. 你需要做的：写RandomImageFolderInstance方法，
   假设训练数据集合为X={x1,x2,..,xn}，数量为n，batch大小为m。在每一批次的训练中，数据样本的构建方式如下：
    对于样本x_i \in x，从剩下的个样本中随机选择两个样本x_a和x_b，应用CutPaste增强和MixUp增强。
    a. CutPaste增强：对于x_a，应用CutPaste增强，首先利用二值分割快速定位前景和背景区域，随后在x_a的前景区域中随机cut一小块区域，paste到x_i的前景区域。
    b. MixUp增强: 对于x_b，应用MixUp增强，在通道维度对x_i和x_b进行线性变换混合。
    c. 混合CutPaste和MixUp：首先应用CutPaste增强，然后将结果与另一张图片进行MixUp增强。公式如下：
    MixUp(CutPaste(x_i, x_a), x_b) = \lambda CutPaste(x_i, x_a) + (1-\lambda)x_b

    RandomImageFolderInstance和下面的ImageFolderInstance输入输出保持一致：
    """
    class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)           
        return sample, index
    """


# 改写训练的prompt
TODO...


# 