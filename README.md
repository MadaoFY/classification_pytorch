# classification_pytorch
 个人利用pytorch魔改、搭建的分类网络，利用CSPnet的思想对resnet进行了重构，并且尝试搭建了一个自己的分类网络，当然你也可以从torchvision或者timm中导入分类网络来进行模型的训练。这里提供了训练、验证、onnx模型导出的代码，你可以使用 Caltech_256 数据集，按照以下步骤来跑通整个训练过程。

 
 ### 环境搭建
 ```bash
git clone https://github.com/MadaoFY/classification_pytorch.git # clone
cd classification_pytorch
pip install -r requirements.txt  # install
```

### 数据集下载
Caltech_256：https://data.caltech.edu/records/20087

### 训练步骤
#### 1、数据清洗及训练集、验证集、测试集的划分(split_Caltech_data.py)
假设你已经完成Caltech_256数据集的下载，我们需要对数据进行清洗及划分，直接运行split_Caltech_data.py脚本即可得到清洗和划分好的数据集，同时会生成train.csv、val.csv、test.csv文件用于之后的训练和验证，目前已存在于项目里的Caltech_256文件夹。

split_Caltech_data.py脚本只用来对Caltech_256进行清洗和划分，如果是训练其他的数据集，你需要自己对数据集进行划分,并且数据集表格式要参考train.csv文件。
```python
# Caltech_256数据路径，下载后解压可得到 "256_ObjectCategories" 文件，因此这里默认设置 './256_ObjectCategories/'
origin_dir = './256_ObjectCategories/'
# 划分后数据存放的位置，你可以按自己的需求进行设置，这里我创建了 "Caltech_256" 文件夹来放置清洗、划分后的数据
new_dir = './Caltech_256/'
# 验证集划分参数，建议不用修改
val_step = 8
# 是否对短边进行缩放，这里设置为True
resize_short_edge = True
# 短边缩放值，长边会同比缩放
short_edge_size = 240
```
划分后文件格式可参考如下：
```
|-Caltech_256
    |-train.csv
    |-val.csv
    |-test.csv
    |-train
        |-001_0001.jpg
        |-001_0002.jpg
        |-001_0003.jpg
        |-...
    |-test
        |-001_0008.jpg
        |-001_0010.jpg
        |-001_0011.jpg
        |-...
```

#### 2、训练(train.py)
假设你已经完成数据集的清洗和划分，并且生成了train.csv、val.csv文件，打开train.py脚本确认参数后即可运行，部分参数如下。
```python
# 训练设备类型
parser.add_argument('--device', default='cuda', help='训练设备类型')
# 训练所需图片的根目录
parser.add_argument('--img_dir', default='./Caltech_256/train/', help='训练所用图片根目录')
# 训练集
parser.add_argument('--train_dir', default='./Caltech_256/train.csv', help='训练集文档')
# 验证集
parser.add_argument('--valid_dir', default='./Caltech_256/val.csv', help='验证集文档')
# 图片的size
parser.add_argument('--img_sz', type=int, default=224, help='train, val image size (pixels)')
# 训练信息保存位置
parser.add_argument('--log_save_dir', default=None, help='tensorboard信息保存地址')
# 模型权重保存地址
parser.add_argument('--model_save_dir', default='./models_save/cspconvnext_t', help='模型权重保存地址')
# 学习率
parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate, 0.001 is the default value for training')
```
脚本的第17行为数据增强的相关代码，99行为optimizer相关代码，你可以根据自己的需求进行修改。


#### 3、验证或预测(val.py、predict.py)
使用val.py脚本对训练好的模型进行验证（acc1），支持onnx模型，你可以验证onnx模型的预测精度。

predict.py脚本用于测试集没有标签的情况下，导出预测结果，默认导出文件名为sub.csv。


#### 4、导出onnx模型(onnx_port.py)
如果你需要onnx模型，可使用onnx_port.py脚本。


### 其他

模型搭建参考

mobilenetv2：https://arxiv.org/abs/1801.04381

https://www.bilibili.com/video/BV1AL411G77N/?spm_id_from=333.999.0.0&vd_source=23508829e27bce925740f90e5cd28cf3


ConvNeXt：https://arxiv.org/abs/2201.03545

https://blog.csdn.net/m0_61899108/article/details/122801668?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EESLANDING%7Edefault-2-122801668-blog-126072800.pc_relevant_multi_platform_whitelistv4eslandingctr&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EESLANDING%7Edefault-2-122801668-blog-126072800.pc_relevant_multi_platform_whitelistv4eslandingctr&utm_relevant_index=5




