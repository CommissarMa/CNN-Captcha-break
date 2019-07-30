# CNN-Captcha-break
基于CNN的固定长度固定字符类型的验证码识别

### 环境要求
Pytorch1.0.0以上版本 [[教程]](https://pytorch.org/get-started/locally/)

### 数据集
+ 本项目解决的验证码类型如下：  
![3chgc](show_imgs/3chgc.png)
+ [[下载链接]](https://pan.baidu.com/s/1sOCbJwOJm2kA5rJWZctEHw) 提取码：22s5  
+ 解压下载文件，将data文件夹放在根目录下
+ 运行check_annotation.py，然后请手动对标注错误的文件进行修正。

### 训练
+ run train.py
+ 训练得到的模型参数会保存在models目录下

### 预测/识别
+ 修改predict.py里面main函数中的模型路径和测试图像路径
+ run predict.py，得到该测试图像的识别结果

### 训练与本文不同的验证码
如果你想要训练与本文不同的验证码，