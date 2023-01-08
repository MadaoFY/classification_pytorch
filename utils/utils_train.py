import os
import math
import torch
import cv2 as cv
import numpy as np

from tqdm import tqdm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

__all__ = [
    'train_classify',
]


def train_classify(
        model,
        train_loader,
        val_loader,
        loss,
        optimizer,
        lr_scheduler,
        epochs,
        num_classes,
        model_save_dir,
        log_save_dir=None,
        model_save_epochs=None,
        mixup=False,
        gpu='cuda',
        fp16=True
):

    if log_save_dir is not None:
        writer = SummaryWriter(log_save_dir)

    best_acc = 0.0
    device = torch.device(gpu) if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    loss = loss.to(device)
    # origin_loss = loss.to(device)
    if mixup:
    # 构建cutmix_mixup方法
        mixup_args = {
            'mixup_alpha': 0.8,
            'cutmix_alpha': 1.0,
            'cutmix_minmax': None,
            'prob': 1.0,
            'switch_prob': 0.5,
            'mode': 'elem',
            'label_smoothing': 0.0,
            'num_classes': num_classes
        }
        mixup_fn = Mixup(**mixup_args)
        mixup_loss = SoftTargetCrossEntropy().to(device)
        # loss = SoftTargetCrossEntropy().to(device)

    if fp16:
        scaler = GradScaler()

    for epoch in range(epochs):
        epoch += 1
        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        # 设置训练模式
        model.train()
        # These are used to record information in training.
        # 记录训练的损失和准确率
        train_losses = []
        train_accs = []

        # Iterate the training set by batches.
        # 初始化迭代次数
        for img, label in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            img = img.float()
            img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
            label = label.reshape(-1)
            # origin_label = label[:].to(device)
            if mixup:
                img, mixed_label = mixup_fn(img, label)
                mixed_label = mixed_label.to(device)

            img = img.to(device)
            label = label.to(device)

            if fp16:
                with autocast():
                    pred = model(img)
                    if mixup:
                        train_loss = mixup_loss(pred, mixed_label)
                    else:
                        train_loss = loss(pred, label)

                # Compute the gradients for parameters.
                # 反向传播
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # Gradients stored in the parameters in the previous step should be cleared out first.
                # 梯度清零
                optimizer.zero_grad()
            else:
                pred = model(img).float()
                if mixup:
                    train_loss = mixup_loss(pred, mixed_label)
                else:
                    train_loss = loss(pred, label)

                # Compute the gradients for parameters.
                # 反向传播
                train_loss.backward()
                optimizer.step()
                # Gradients stored in the parameters in the previous step should be cleared out first.
                # 梯度清零
                optimizer.zero_grad()

            # Compute the accuracy for current batch.
            # 计算准确率
            pred = pred.argmax(dim=-1)
            acc = (pred == label).float().mean()

            # Record the loss and accuracy.
            train_losses.append(train_loss.item())
            train_accs.append(acc)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # The average loss and accuracy of the training set is the average of the recorded values.
        # 计算该轮次的平均损失和准确率
        train_losses = sum(train_losses) / len(train_losses)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        # 打印每轮的loss信息
        print(f"[Train | {epoch:03d}/{epochs:03d} ] lr={optimizer.state_dict()['param_groups'][0]['lr']:.5f}, "
              f"loss = {train_losses:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        # 设置检测模式
        model.eval()
        # These are used to record information in validation.
        valid_losses = []
        valid_accs = []

        # Iterate the validation set by batches.
        for img, label in tqdm(val_loader):
            img = img.float().to(device)
            label = label.to(device)
            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            # 不需要梯度
            with torch.no_grad():
                pred = model(img)
                valid_loss = loss(pred, label)

            # Compute the accuracy for current batch.
            pred = pred.argmax(dim=-1)
            acc = (pred == label).float().mean()

            # Record the loss and accuracy.
            valid_losses.append(valid_loss.item())
            valid_accs.append(acc)

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_losses = sum(valid_losses) / len(valid_losses)
        valid_acc = sum(valid_accs) / len(valid_accs)

        if log_save_dir is not None:
            writer.add_scalars('loss', {'train_loss': train_losses, 'val_loss': valid_losses},
                               global_step=epoch)
            writer.add_scalars('acc', {'train_acc': train_acc, 'val_acc': valid_acc},
                               global_step=epoch)
            writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'],
                               global_step=epoch)

        # Print the information.
        print(f"[Valid | {epoch:03d}/{epochs:03d} ] loss = {valid_losses:.5f}, "
              f"acc = {valid_acc:.5f}")

        # if the model improves, save a checkpoint at this epoch
        # 保持训练权值
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), f'{model_save_dir}_{epoch}_{best_acc:.5f}.pth')
            print('{}[ saving model with best_acc {:.5f} ]{}'.format('-' * 15, best_acc, '-' * 15))
        if epoch in model_save_epochs:
            torch.save(model.state_dict(), f'{model_save_dir}_{epoch}.pth')
            print(f'saving model with epoch {epoch}')
    if log_save_dir is not None:
        writer.close()
    print(f'Done!!!best acc = {best_acc:.5f}')
