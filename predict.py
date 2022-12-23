import os
import timm
import torch
import onnxruntime
import cv2 as cv
import numpy as np
import pandas as pd
import albumentations as A

from tqdm.auto import tqdm
from torch.utils.data import Dataset
from albumentations import pytorch as AT

from utils.dataset import ReadDataSet
from models.cspconvnext import cspconvnext_t, cspconvnext_s
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def test_transform():
    transforms = []
    transforms.append(A.CenterCrop(args.img_sz, args.img_sz))
    transforms.append(A.Normalize())
    transforms.append(AT.ToTensorV2())
    return A.Compose(transforms)


def main(args):
    img_dir = os.path.join(args.img_dir)  # '../images/'
    test = pd.read_csv(args.test_dir)   # '../test.csv'
    submission = test.copy()

    test_dataset = ReadDataSet(test, img_dir, test_transform(), test=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        )

    weights = args.weights
    last_name = os.path.splitext(weights)[-1]
    assert last_name in ['.pth', '.pt', '.onnx'], f"weights file attribute is {last_name}, not in [.pth , .pt, .onnx]."
    if last_name == '.onnx':
        model = onnxruntime.InferenceSession(
            weights,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        predictions = np.array([], dtype=np.int16)
        for img in tqdm(test_loader):
            img = img.float()
            inputs = {model.get_inputs()[0].name: img.numpy()}
            pred = model.run(None, inputs)
            pred = pred[0].argmax(axis=1)
            predictions = np.concatenate((predictions, pred))

    elif last_name in ['.pth', 'pt']:
        models_dict = {
            'cspconvnext_t': cspconvnext_t,
            'cspconvnext_s': cspconvnext_s
        }
        classify_model = models_dict[args.model]
        model = classify_model(num_classes=args.num_classes).to(device)  #257
        param_weights = torch.load(weights)
        model.load_state_dict(param_weights, strict=True)

        model.eval()
        predictions = torch.tensor([], device=device, dtype=torch.int16)
        for img in tqdm(test_loader):
            img = img.float().to(device)
            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                pred = model(img)

            pred = pred.argmax(dim=-1)
            predictions = torch.cat([predictions, pred])
        predictions = predictions.cpu()

    else:
        pass

    submission['prediction'] = predictions.tolist()

    if args.submission_save_dir:
        submission.to_csv(args.submission_save_dir, index=False)
        print('Submission is saved')

    print("Done!!!!!!!!!!!!!!!!!!!!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # 模型
    parser.add_argument("--model", type=str, default='cspconvnext_t', choices=['cspconvnext_t', 'cspconvnext_s'],
                        help="模型选择")
    # 权重
    parser.add_argument('--weights', default='./models_save/cspconvnext_t_165_0.71224.pth',
                        help='模型文件地址; pth,pt,onnx模型')
    # 测试集
    parser.add_argument('--test_dir', default='./test.csv', help='测试集文档')
    # 测试集图片的文件夹
    parser.add_argument('--img_dir', default='./Caltech_256/', help='训练所用图片根目录')
    # submission保存位置
    parser.add_argument('--submission_save_dir', default='./sub.csv', help='submission保存地址')
    # batch_size
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size when training')
    # 数据集分类数量
    parser.add_argument('--num_classes', type=int, default=257, help='数据集分类数量')
    # 图片的size
    parser.add_argument('--img_sz', type=int, default=224, help='train, val image size (pixels)')

    args = parser.parse_args()
    print(args)

    main(args)