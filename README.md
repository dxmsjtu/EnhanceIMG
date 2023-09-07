# EnhanceIMG

[TOC]

此代码库用于图像增强算法的探索，主要包括：低光增强、图像修复、超分辨率重建 …… 
## 实验室部署流程
```
pip install -r requirements.txt
```
编译好基础环境后，


右键run demo.py执行demo脚本，其中大致添加了代码注释。
如下图所示



![noises](https://cdn.nlark.com/yuque/0/2023/png/12831330/1691393677419-542d3be9-a481-4bb9-a01b-0e787c8aa26f.png
)

安装成功后的终端显示结果如下：
![noises](https://cdn.nlark.com/yuque/0/2023/png/12831330/1691393690662-3cfde386-1509-43c8-9f55-92d365f21b42.png
)

点击run demo.py，就会出现不同方法的运行结果



![noises](https://cdn.nlark.com/yuque/0/2023/png/12831330/1691394027657-b96fd442-bf55-4b48-9f41-8b1cda87c1ea.png?x-oss-process=image%2Fresize%2Cw_1500%2Climit_0
)

## Retinex部分解析

具体论文文献已经在9.7号上传到仓库中,Retinex原论文.pdf。

通过对Retinex算法的学习，进行简要的阶段性总结。

Retinex就是一种用于图像增强的经典算法，旨在改善图像的亮度和对比度，特别是在低光照条件下。

* * *

Retinex算法的主要思路包括以下几个关键步骤：

-   图像分解：首先，将原始图像分解为反射成分和光照成分的乘积。这一分解过程通常使用对数域进行，因为在对数域中，图像的光照和反射成分更容易分离。
-   估计光照成分：Retinex算法通过对原始图像进行模糊处理来估计光照成分。模糊处理有助于捕捉图像的全局亮度信息，即光照成分。
-   计算反射成分：将原始图像除以估计得到的光照成分，得到反射成分。反射成分反映了图像中的细节和纹理，通常包含有关物体的信息。
-   增强反射成分：对反射成分进行增强，以提高图像的对比度和清晰度。这一步可以采用各种增强方法，例如直方图均衡化、对比度增强等。
-   重构增强后的图像：将增强后的反射成分与光照成分相乘，以重构最终的增强图像。

### __init__.py

该程序文件提供多种处理函数，可以根据需要选择和组合，以实现不同级别的图像增强和色彩改进。

-   replace_zeroes(data)：将输入数据中的所有零元素替换为非零最小元素，以避免在取对数时出现问题。
-   single_scale_retinex(img, sigma)：实现单尺度Retinex算法，该算法通过对输入图像进行高斯模糊处理，然后计算对数域表示，最后计算图像的反射成分。返回Retinex增强后的图像。
-   multi_scale_retinex(img, sigma_list)：实现多尺度Retinex算法，通过在多个尺度上应用Retinex算法，并将它们加权求和，最后得到多尺度Retinex增强后的图像。这可以提高对不同光照条件下图像的适应性。
-   color_restoration(img, alpha, beta)：实现色彩还原，该算法通过计算图像的颜色分布以及原始颜色分布之间的差异来增强颜色。参数 alpha 和 beta 控制着颜色还原的强度。
-   simplest_color_balance(img, low_clip, high_clip)：实现简单的白平衡，该算法通过将图像的RGB通道像素值分布压缩到相同的区间内来提高图像的颜色平衡。参数 low_clip 和 high_clip 控制着颜色平衡的强度。
-   MSRCR(img, sigma_list, G, b, alpha, beta, low_clip, high_clip)：实现色彩恢复的多尺度Retinex算法，结合了多尺度Retinex和色彩还原，以提高图像的颜色质量和对比度。参数 G 和 b 控制了Retinex和色彩还原之间的权重。
-   automated_MSRCR(img, sigma_list)：实现色彩增益加权的多尺度Retinex算法，该算法根据图像的像素分布自动调整颜色增益，以提高图像的质量和对比度。
-   MSRCP(img, sigma_list, low_clip, high_clip)：实现带色彩还原的多尺度视网膜增强算法，结合了多尺度Retinex和色彩平衡，用于增强图像的颜色和对比度。参数 low_clip 和 high_clip 控制着颜色平衡的程度。

### retinex_cv.py

Retinex文件夹下retinex_cv,py程序文件。实现了多尺度图像增强（Multi-Scale Retinex with Color Restoration，MSRCR）和多尺度图像增强彩色恢复（Multi-Scale Retinex with Color Restoration and Principle Component Analysis，MSRCP）算法。 具体代码注释已经加入程序中，可以对应参考学习。

###
## 目录结构

```
.
|--- asserts/
|--- awegan/      # GAN相关算法
     |--- datasets/
     |--- models/         # CycleGAN/Pix2Pix/SelfGAN
     |--- options/
     |--- util/
     |--- __init__.py
     |--- train.py
     |--- ...
|--- colorspace/  # 色彩空间转换
|--- edges/       # 边缘检测算法
|--- filters/     # 各种滤波器
|--- histeq/      # 直方图均衡算法
|--- noises/      # 噪声
|--- priors/      # 自然图像先验信息
     |--- __init__.py
     |--- denoising.py
     |--- inpainting.py
     |--- networks.py     # ResNet/SkipNet/UNet
     |--- restoration.py
     |--- ...
|--- retinex/     # Retinex系列算法
     |--- __init__.py
     |--- enhancer.py
     |--- retinex_net.py  # RetinexNet
     |--- ...
|--- utils/       # 一些方法
|--- .gitignore
|--- demo.py
|--- LICENSE
|--- Madison.png
|--- README.md    # 说明文档
|--- requirements.txt     # 依赖文件
```

## 链路运行效果图

### 添加噪声

**噪声**（原图|椒盐噪声|高斯噪声）

![noises](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/1619614042-noises.png)

### 各种滤波器

**滤波器**（椒盐噪声|均值滤波|中值滤波）

![filters1](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/1619614242-filters1.png)

**滤波器**（高斯噪声|高斯滤波|双边滤波|联合双边滤波）

![filters2](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/1619614258-filters2.png)

**滤波器**（高斯噪声|引导滤波）

![filters3](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/1619614271-filters3.png)

### 边缘检测

**检测算子**（灰度图|Laplacian|Sobel|Scharr）

![opt-edge-detection-2](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1619852372-opt-edge-detection-2.png)

**检测算子**（灰度图|LoG|DoG|Gabor）

![opt-edge-detection-3](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1620812279-opt-edge-detection-3.png)

**其他算法**（灰度图|结构森林|HED|HED-feats-5）

![hn-edge-detection](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1619852478-hn-edge-detection.png)

![hed-fs1-fs5](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1619955819-hed-fs1-fs5.png)

### 传统增强算法

**直方图均衡**（原图|HE|AHE|CLAHE）

![hist-equal](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/1619614292-hist-equal.png)

**Gamma 校正**（原图|Gamma|Gamma+MSS）

![adjust-gamma](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/1619684267-adjust-gamma.png)

**Retinex**（原图|MSRCR|AMSRCR|MSRCP）

![retinex](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/1619614304-retinex.png)

**Retinex 增强**（原图|AttnMSR|AttnMSR+MSS）（Mine）

![enlighten](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/1619614316-enlighten.png)

### 自然图像先验

**降噪**（噪声图|降噪1|降噪2）

![prior-denoising](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1621076456-prior-denoising.png)

### 神经网络

**RetinexNet**（原图|RetinexNet）

![retinexnet](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/1619668202-retinexnet.png)

### 生成对抗网络

**Pix2Pix**

（边缘 <=> 图像）

![pix2pix-facades](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1620004141-pix2pix-facades.png)

（低光 <=> 正常）

![pix2pix](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1619955841-pix2pix.png)

![pix2pix4](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1620038713-pix2pix4.png)

**CycleGAN**

（夏天 <=> 冬天）

![summer2winter](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1619937669-summer2winter.png)

（低光 <=> 正常）

![cyclegan4](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1620037334-cyclegan4.png)

## 参考资料文献

- Retinex
  - [Multiscale Retinex](http://www.ipol.im/pub/art/2014/107/)
  - [An automated multi Scale Retinex with Color Restoration for image enhancement](http://ieeexplore.ieee.org/document/6176791/)
  - [A multiscale retinex for bridging the gap between color images and the human observation of scenes](http://ieeexplore.ieee.org/document/597272/)
  - [Deep Retinex Decomposition for Low-Light Enhancement](https://arxiv.org/abs/1808.04560)
- Image Prior
  - [Deep Image Prior](https://dmitryulyanov.github.io/deep_image_prior)
- GAN network
  - [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
  - [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
  - [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/abs/1711.11585)
  - [Toward Multimodal Image-to-Image Translation](https://arxiv.org/abs/1711.11586)
  - [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
  - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
  - [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
  - [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1605.06211v1)

## TODO

- [x] AttnMSR 图像增强算法（Mine）
- [x] RetinexNet 低光增强模型
- [x] ResNet / SkipNet / UNet
- [ ] Deep Image Prior（自然图像先验信息）
- [x] Pix2Pix 模型用于图像增强
- [x] CycleGan 模型用于图像增强
- [ ] SelfGAN 图像增强模型（Mine，完善中）

