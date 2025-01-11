# 基于多视角图像的3D高斯分布重建

本仓库提供了一个完整的流程，用于从多视角图像中重建由3D高斯分布（3DGS）表示的3D场景。该流程包括使用Colmap进行运动恢复结构（SfM）、简化的3D高斯分布实现以及体渲染。该实现基于PyTorch，旨在易于理解和扩展。

---

## 资源

- **论文**: [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- **官方3DGS实现**: [GitHub仓库](https://github.com/graphdeco-inria/gaussian-splatting)
- **Colmap**: [运动恢复结构工具](https://colmap.github.io/index.html)

---

## 需要的python库
import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader

from pathlib import Path

import argparse

import numpy as np

from tqdm import tqdm

from dataclasses import dataclass

import cv2

import os

## 流程概述

该流程包括以下步骤：

1. **运动恢复结构（SfM）**: 使用Colmap从多视角图像中恢复相机姿态和稀疏的3D点。
2. **3D高斯分布重建**: 将稀疏的3D点扩展为3D高斯分布，将其投影到2D空间，并进行体渲染。
3. **与原始3DGS实现的比较**: 将结果与原始3DGS实现进行比较。

---

## 步骤1：运动恢复结构（SfM）

首先，我们使用Colmap从多视角图像中恢复相机姿态和稀疏的3D点。输入图像应放置在`data/chair/images`文件夹中（或您选择的其他文件夹）。

### 运行Colmap进行SfM
```bash
python mvs_with_colmap.py --data_dir data/chair
```

### 调试重建结果
为了可视化重建的3D点和相机姿态，运行：
```bash
python debug_mvs_by_projecting_pts.py --data_dir data/chair
```

---

## 步骤2：简化的3D高斯分布重建

此步骤涉及将稀疏的3D点扩展为3D高斯分布，将其投影到2D空间，并进行体渲染。实现分为以下几个子步骤：

### 2.1 3D高斯分布初始化
将稀疏的3D点转换为3D高斯分布，为每个点定义协方差矩阵。初始高斯分布的中心是3D点本身。协方差矩阵使用缩放矩阵`S`和旋转矩阵`R`定义。此外，为每个高斯分布分配不透明度和颜色属性以进行体渲染。

- **需要实现的代码**: 在[`gaussian_model.py#L103`](gaussian_model.py#L103)中填写函数，以从四元数（旋转）和缩放参数计算3D协方差矩阵。

### 2.2 将3D高斯分布投影到2D空间
使用世界到相机的变换矩阵`_W_`和投影变换的雅可比矩阵`_J_`将3D高斯分布投影到2D空间。

- **需要实现的代码**: 在[`gaussian_renderer.py#L26`](gaussian_renderer.py#L26)中填写函数以计算投影。

### 2.3 计算高斯值
使用2D高斯分布进行体渲染。像素位置`x`处的高斯值使用以下公式计算：

$$
f(\mathbf{x}; \boldsymbol{\mu}\_{i}, \boldsymbol{\Sigma}\_{i}) = \frac{1}{2 \pi \sqrt{ | \boldsymbol{\Sigma}\_{i} |}} \exp \left ( {-\frac{1}{2}} (\mathbf{x} - \boldsymbol{\mu}\_{i})^T \boldsymbol{\Sigma}\_{i}^{-1} (\mathbf{x} - \boldsymbol{\mu}\_{i}) \right )
$$

- **需要实现的代码**: 在[`gaussian_renderer.py#L61`](gaussian_renderer.py#L61)中填写函数以计算高斯值。

### 2.4 体渲染（α混合）
使用alpha混合计算最终渲染结果。2D高斯分布在像素位置`x`处的alpha值由以下公式给出：

$$
\alpha_{(\mathbf{x}, i)} = o_i * f(\mathbf{x}; \boldsymbol{\mu}\_{i}, \boldsymbol{\Sigma}\_{i})
$$

其中`o_i`是高斯分布的不透明度。透射率值计算为：

$$
T_{(\mathbf{x}, i)} = \prod_{j \lt i} (1 - \alpha_{(\mathbf{x}, j)})
$$

- **需要实现的代码**: 在[`gaussian_renderer.py#L83`](gaussian_renderer.py#L83)中填写函数以计算最终渲染结果。

### 训练3DGS模型
完成上述步骤后，使用以下命令训练3DGS模型：
```bash
python train.py --colmap_dir data/chair --checkpoint_dir data/chair/checkpoints
```

---

## 步骤3：与原始3DGS实现的比较

该实现是原始3DGS论文的简化版本。由于使用了纯PyTorch实现，训练速度和GPU内存使用并不理想。此外，未实现一些高级功能，如自适应高斯分布密集化。

要在相同数据集上运行原始3DGS实现以比较结果，请运行：
```bash
# 按照官方3DGS仓库的说明操作：
# https://github.com/graphdeco-inria/gaussian-splatting
```

---

## 结果

简化的3DGS实现的结果可以在`data/chair/checkpoints`文件夹中查看。将这些结果与原始3DGS实现的输出进行比较，以评估重建的性能和质量。

我们训练200个epoch，训练学习率定为0.01.得到结果如下，由于时间关系，只训练了chair的数据集

![result](f14baad14264de7a76652e6923e1a4f5.mp4)
---

## 未来工作

- **提高训练效率**: 优化PyTorch实现以提高训练速度并减少GPU内存使用。
- **实现自适应密集化**: 添加自适应高斯分布密集化支持以提高重建质量。
- **扩展到其他数据集**: 在更复杂场景的数据集上测试该流程。

---



---

## 致谢

- 感谢3D高斯分布论文及其作者的原始实现。
- 感谢Colmap提供了优秀的运动恢复结构工具。

---

如有任何问题或建议，请在GitHub上提交问题或联系维护者。
