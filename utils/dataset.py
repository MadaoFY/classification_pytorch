import torch
import cv2 as cv
import numpy as np
from os import path
from torch.utils.data import Dataset

__all__ = [
    'ReadDataSet',
]


class ReadDataSet(Dataset):
    r"""
    参数说明：
        data_file：记录图片信息的csv文档，要求第一列为图片路径，第二列为图片的标签。
        img_dir：图片所在的文件夹路径。
        transform：图片增强方法。
        test(bool)：是否为测试集。默认值：False
        repeat(bool)：是否使用repeat方法，用于训练模式，使用后将一张图片复制成4张。默认值：False
    """

    def __init__(self, data_file, img_dir, transforms=None, test=False, repeat=False):
        super().__init__()
        self.data = data_file.values
        self.img_dir = img_dir
        self.transforms = transforms
        self.repeat = repeat
        self.test = test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 读取图片和label
        img = cv.imread(path.join(self.img_dir, self.data[idx, 0]))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        if not self.test:
            label = self.data[idx, 1]
            if self.repeat:
                imgs = None
                for _ in range(4):
                    if imgs is None:
                        imgs = self.transforms(image=img)['image'].unsqueeze(0)
                    else:
                        imgs = torch.cat((imgs, self.transforms(image=img)['image'].unsqueeze(0)), dim=0)
                img = imgs
                label = np.repeat(label, 4, 0)
            else:
                img = self.transforms(image=img)['image']
            return img, label

        else:
            if self.transforms:
                img = self.transforms(image=img)['image']
            return img


if __name__ == '__main__':
    pass
    # path = 'train.csv'
    # data = np.loadtxt(path, delimiter=',', dtype=str, skiprows=1, usecols=None, unpack=False)
    # print(data)
