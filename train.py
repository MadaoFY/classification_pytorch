import os
import time
import timm
import torch
from torch import nn
import pandas as pd
import albumentations as A
from albumentations import pytorch as AT

from models.cspconvnext import cspconvnext_t
from utils.dataset import ReadDataSet
from utils.utils_train import train_classify
from utils.general import SineAnnealingLR, same_seeds


# 数据增强操作
def train_transform():
    transforms = []
    transforms.append(A.RandomResizedCrop(args.img_sz, args.img_sz, scale=(0.2, 1), interpolation=2, p=1))
    transforms.append(A.OneOf([
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=1),
        A.RandomBrightnessContrast(p=1)
    ], p=0.5))
    transforms.append(A.OneOf([
        A.Emboss(p=1),
        A.Sharpen(p=1)
    ], p=0.4))
    transforms.append(A.OneOf([
        A.GaussianBlur(p=1),
        A.MedianBlur(p=1)
    ], p=0.3))
    transforms.append(A.HorizontalFlip(p=0.5))
    # transforms.append(A.Cutout(num_holes=8, max_h_size=12, max_w_size=12, p=0.1))
    transforms.append(A.Normalize())
    transforms.append(AT.ToTensorV2())
    return A.Compose(transforms)


def val_transform():
    transforms = []
    # transforms.append(A.Resize(256, 256, interpolation=2, p=1))
    transforms.append(A.CenterCrop(args.img_sz, args.img_sz))     # 保证最小边大于、等于指定值
    transforms.append(A.Normalize())
    transforms.append(AT.ToTensorV2())
    return A.Compose(transforms)


def main(args):
    # 有gpu就用gpu
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    same_seeds(42)

    # 读取训练集验证集
    train = pd.read_csv(args.train_dir)
    valid = pd.read_csv(args.valid_dir)
    img_dir = os.path.join(args.img_dir)

    train_dataset = ReadDataSet(train, img_dir, train_transform(), repeat=True)
    val_dataset = ReadDataSet(valid, img_dir, val_transform())

    # 设置batch大小
    batch_size = args.batch_size
    if train_dataset.repeat:
        train_batch_size = batch_size // 4
    else:
        train_batch_size = batch_size

    # 构建dataloader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    # 学习率
    lr = args.lr
    weight_decay = args.weight_decay
    # 训练轮次
    epochs = args.epochs
    # 模型权重保存路径
    model_save_dir = args.model_save_dir
    # 创建模型,你可以自行导入pytorch或者timm提供的模型
    model = cspconvnext_t(num_classes=args.num_classes)
    # model = timm.create_model('cspresnext50', pretrained=False, num_classes=args.num_classes).to(device)
    # for i, p in enumerate(model.parameters()):
    #     if i < 1:
    #         p.requires_grad = False

    # 创建损失函数
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    # 创建优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
        betas=(0.9, 0.999), weight_decay=weight_decay
    )

    # 优化策略
    t_max = 20
    lr_cosine = SineAnnealingLR(optimizer, t_max)
    # lr_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    # 是否使用半精度训练
    fp16 = args.fp16
    # 用于计算训练时间
    start = time.time()

    # 模型训练
    train_classify(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        lr_cosine,
        epochs,
        args.num_classes,
        model_save_dir,
        log_save_dir=args.log_save_dir,
        model_save_epochs=args.model_save_epochs,
        mixup=args.mixup,
        gpu=args.device,
        fp16=fp16
    )
    print(f'{epochs} epochs completed in {(time.time() - start) / 3600.:.3f} hours.')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

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
    parser.add_argument('--model_save_dir', default='./models_save/cspconvnext_t',
                        help='模型权重保存地址')
    # 学习率
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='initial learning rate, 0.001 is the default value for training')
    # 训练的总epoch数
    parser.add_argument('--epochs', type=int, default=330, metavar='N', help='number of total epochs to run')
    # 优化器的weight_decay参数
    parser.add_argument('--weight-decay', type=float, default=0.05, metavar='W', help='weight decay')
    # 训练的batch_size
    parser.add_argument('--batch_size', type=int, default=384, metavar='N', help='batch size when training')
    # 训练时是否使用repeat方法拓展数据集
    parser.add_argument('--repeat', type=bool, default=True, choices=[True, False], help='将训练集的一张图片复制成4张')
    # 训练时是否使用mixup方法
    parser.add_argument('--mixup', type=bool, default=True, choices=[True, False], help='训练时是否使用mixup方法')
    # 数据集分类数量
    parser.add_argument('--num_classes', type=int, default=257, help='数据集分类数量')
    # 额外指定权重保存epoch
    parser.add_argument('--model_save_epochs', type=list, default=[], metavar='N', help='额外指定epoch进行权重保存')
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--fp16", type=bool, default=True, choices=[True, False],
                        help="Use fp16 for mixed precision training")

    args = parser.parse_args()
    print(args)

    main(args)
