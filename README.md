# SVM Introduction
支持向量机 (Support Vector Machines, SVM) 是机器学习中的一个重要算法，本人多次在深圳大学《机器学习》本科课程上对SVM进行入门讲解，
本仓库记录该过程中的相关资料和代码。

## 更新：

## Slides (PDF)
- **SVM_intro_2020.pdf**: 首次在同学面前演讲 SVM 基础理论的最初版本，同时也是比较简陋的版本。内容为基本的 SVM 的大致框架，包括硬间隔、软间隔以及非线性的 SVM。该版本在逻辑上仍有欠缺。
- **SVM_intro_2021.pdf**: 第一次修改的版本，在深圳大学 2020-2021 学年第二学期的《机器学习》本科课程上进行演讲。加入了感知机和 SVM 的比较，去除了不必要的拉格朗日乘数法的过多复习，增加了部分细节，逻辑更加清晰。
- **SVM_intro_2022.pdf**: 修改了2021版中的错误，重新绘制了部分图片.
- **SVM_intro_2023.pdf**: 重新调整了讲解内容（删除感知机，同时更侧重对偶理论的解释）和顺序，添加部分图片.
- **SVM_intro_2025.pdf** (New): 大幅调整内容，加入许多细节讲解，适合新手入门，宝宝巴士版本。

## Video (Bilibili)
- [2023年课程录像](https://www.bilibili.com/video/BV1TP411U7dH)
- [2025年课程录像](https://www.bilibili.com/video/BV1E6doYMEb2)


## Optional Notes (PDF)
- **duality.pdf**: 讲解有关凸优化中对偶理论的原理，需要凸优化方面的基础知识.

## Codes (Matlab)
- **code-matlab**: 该文件夹下包含 Matlab 手动实现 SVM 的实例（二分类、多分类问题），可直接运行。 程序编写环境为 Matlab R2020a。**注意**，这些程序依赖 Matlab 的工具包 Optimization Toolbox，安装后才可以正常运行。

## Codes (Python)
- **code-python**: 该文件夹下包含使用 Python3 的 Numpy 模块和 Cvxopt 模块手动实现 SVM 的实例（二分类问题），可直接运行。**注意**，你需要安装 numpy 和 cvxopt 模块才能正常运行，例如通过 `pip` 安装：
```bash
pip install numpy
pip install cvxopt
```
