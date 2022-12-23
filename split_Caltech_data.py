import os
import cv2 as cv
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from shutil import copyfile, move

from utils.general import resize_image


def copy_objectcategories(origin_dir, new_dir):

    origin_dir = origin_dir  # './256_ObjectCategories/'
    new_dir = new_dir  # './Caltech_256/'
    img_list = []
    print('Start copy images...')
    for i in tqdm(os.listdir(origin_dir)):
        img_dir = os.path.join(origin_dir, i)
        for i in os.listdir(img_dir):
            img_list.append(i)
            copyfile(f'{img_dir}/{i}', f'{new_dir}/{i}')

    print("Images copy is completed!!!")

    return img_list


def split_data(data, train_nums=13000):

    # 划分成train、val、test
    index = data.index.tolist()
    val_index = np.arange(0, len(index), 8).tolist() + [1]
    index = list(set(index) - set(val_index))
    np.random.seed(24)
    np.random.shuffle(index)
    test_index = index[-train_nums:]
    train_index = index[:-train_nums]
    train = data.drop(index=(val_index + test_index))
    val = data.drop(index=(train_index + test_index))
    test = data.drop(index=(train_index + val_index))

    print("Dataset split is completed!!!")

    return train, val, test


def main(origin_dir, new_dir, csv_save_dir='../', resize_short_edge=False):

    """

    :param origin_dir: 数据集图片的原始位置
    :param new_dir: 图片将要复制到的位置
    :param csv_save_dir: 训练集、验证集、测试集的csv文件保存的文件夹，默认保存仔当前根目录
    :param resize_short_edge: 是否将图片按短边进行resize
    """
    # 对下载好的数据集进行整理和移动
    # 获得图片的信息列表
    imgs_list = copy_objectcategories(origin_dir, new_dir)
    # 用pandas进行处理
    data = pd.DataFrame({'img': imgs_list})
    data['label'] = data['img'].map(lambda x: int(x.split('_')[0]))

    # 传入经过pandas处理的图片信息列表，划分得到train、val、test
    train, val, test = split_data(data)

    # 导出成csv文件保存
    train.to_csv(f'{csv_save_dir}train.csv', index=False)
    val.to_csv(f'{csv_save_dir}val.csv', index=False)
    test.to_csv(f'{csv_save_dir}test.csv', index=False)

    # 是否按短边进行resize
    resize_short_edge = resize_short_edge
    if resize_short_edge:
        for i in tqdm(imgs_list):
            img_dir = f'../images/{i}'
            img = cv.imread(img_dir)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = resize_image(img, 240)
            cv.imwrite(img_dir, img, [int(cv.IMWRITE_JPEG_QUALITY), 100])

    print("Done!!!")


if __name__ == '__main__':
    # 数据集下载地址 Caltech_256：https://data.caltech.edu/records/20087

    # 对下载好的数据集进行整理和移动
    origin_dir = './256_ObjectCategories/'
    new_dir = './Caltech_256/'
    resize_short_edge = True

    main(origin_dir, new_dir, resize_short_edge=resize_short_edge)


