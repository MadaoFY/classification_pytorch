import os
import cv2 as cv
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from shutil import copyfile, move

from utils.general import resize_image


def copy_objectcategories(origin_dir, new_dir, resize_short_edge=False, short_edge_size=256):

    origin_dir = origin_dir
    new_dir = new_dir
    img_list = []
    print(f'Start copy images from {origin_dir}')
    for i in tqdm(os.listdir(origin_dir)):
        img_dir = os.path.join(origin_dir, i)
        for i in os.listdir(img_dir):
            if i.split(sep='.')[-1] in ['jpg']:
                # 是否按短边进行resize
                if resize_short_edge:
                    img = cv.imread(f'{img_dir}/{i}')
                    img = resize_image(img, short_edge_size)
                    cv.imwrite(f'{new_dir}/{i}', img, [int(cv.IMWRITE_JPEG_QUALITY), 100])
                else:
                    copyfile(f'{img_dir}/{i}', f'{new_dir}/{i}')
                img_list.append(i)

    print("Images copy is completed!!!")

    return img_list


def split_data(data, step=8, test_nums=13000):

    # 划分成train、val、test，先按步长抽取出验证集，在按test_nums选取测试集，最后剩下的为训练集

    index = data.index.tolist()
    val_index = np.arange(0, len(index), step).tolist() + [1]
    index = list(set(index) - set(val_index))
    np.random.seed(24)
    np.random.shuffle(index)
    test_index = index[-test_nums:]
    train_index = index[:-test_nums]
    train = data.drop(index=(val_index + test_index))
    val = data.drop(index=(train_index + test_index))
    test = data.drop(index=(train_index + val_index))

    print("Dataset split is completed!!!")

    return train, val, test


def main(origin_dir, new_dir, csv_save_dir=None, val_step=8, resize_short_edge=False, short_edge_size=256):

    """

    origin_dir: 数据集图片的原始位置
    new_dir: 图片将要复制到的位置
    csv_save_dir: 训练集、验证集、测试集的csv文件保存的文件夹，默认保存在new_dir目录
    val_step: 验证集选取步长。每间隔val_step张图片选取1张图片作为验证集
    resize_short_edge: 是否将图片按短边进行resize
    short_edge_size: 短边缩放值，长边等比例缩放
    """
    # 对下载好的数据集进行整理和移动
    # 获得图片的信息列表
    imgs_list = copy_objectcategories(origin_dir, new_dir, resize_short_edge, short_edge_size)
    # 用pandas进行处理
    data = pd.DataFrame({'img': imgs_list})
    data['label'] = data['img'].map(lambda x: int(x.split('_')[0]) - 1)

    # 传入经过pandas处理的图片信息列表，划分得到train、val、test
    train, val, test = split_data(data, val_step)

    # 新建train、test文件夹用于放置训练集、验证集喝测试集
    train_dir = f'{new_dir}/train'
    test_dir = f'{new_dir}/test'

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    print(f'create {train_dir}')
    print(f'Moving train and val images to {train_dir}')
    for d in [train, val]:
        for i in tqdm(d['img']):
            img_dir = f'{new_dir}{i}'
            move(img_dir, f'{train_dir}/{i}')

    print(f'create {test_dir}')
    print(f'Moving test images to {test_dir}')
    for i in tqdm(test['img']):
        img_dir = f'{new_dir}{i}'
        move(img_dir, f'{test_dir}/{i}')


    if csv_save_dir is None:
        csv_save_dir = new_dir

    # 导出成csv文件保存
    train.to_csv(f'{csv_save_dir}/train.csv', index=False)
    val.to_csv(f'{csv_save_dir}/val.csv', index=False)
    test.to_csv(f'{csv_save_dir}/test.csv', index=False)

    print("Done!!!")


if __name__ == '__main__':
    # 数据集下载地址 Caltech_256：https://data.caltech.edu/records/20087

    # 需要设置的参数
    origin_dir = './256_ObjectCategories/'
    new_dir = './Caltech_256/'
    val_step = 8
    resize_short_edge = True
    short_edge_size = 240

    # 对下载好的数据集进行整理和移动
    main(origin_dir, new_dir, val_step=val_step, resize_short_edge=resize_short_edge, short_edge_size=short_edge_size)


