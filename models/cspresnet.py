import torch
import torch.nn as nn

__all__ = [
    'cspresnet50',
    'cspresnet101'
]


class bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_chs, out_chs, stride=1, downsample=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.blk = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chs, out_chs, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chs, out_chs * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_chs * self.expansion),
        )

        if downsample:
            self.down = nn.Sequential(
                nn.Conv2d(in_chs, out_chs * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chs * self.expansion)
            )
        else:
            self.down = None

    def forward(self, x):
        identity = x

        if self.down:
            identity = self.down(x)

        y = self.blk(x)
        y += identity
        return self.relu(y)


class cspneck(nn.Module):
    def __init__(self, in_chs, out_chs, num_layer, s=1):
        super().__init__()
        self.blk = bottleneck
        self.input_channels = in_chs // 2
        self.c1 = nn.Sequential(
            nn.Conv2d(self.input_channels, out_chs * self.blk.expansion // 2, kernel_size=1, stride=s,
                      bias=False),
            nn.BatchNorm2d(out_chs * self.blk.expansion // 2),
            nn.ReLU(inplace=True)
            )

        block = nn.ModuleList()
        # output, block_num, strides
        block.append(self.blk(self.input_channels, out_chs // 2, stride=s, downsample=True))
        for _ in range(1, num_layer):
            block.append(self.blk(out_chs * self.blk.expansion // 2, out_chs // 2))
        self.c2 = nn.Sequential(*block)

        self.trans = nn.Sequential(
            nn.Conv2d(out_chs * self.blk.expansion, out_chs * self.blk.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_chs * self.blk.expansion),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x0, x1 = x.split(int(self.input_channels), dim=1)
        x0 = self.c1(x0)
        x1 = self.c2(x1)
        return self.trans(torch.cat((x0, x1), 1))


class resnet(nn.Module):
    def __init__(self, num_classes, block, parame):
        super().__init__()
        self.blk = block
        self.parame = parame
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.layer1 = self._make_layer(64, 64, self.parame[0], 1)
        self.layer2 = self._make_layer(256, 128, self.parame[1], 2)
        self.layer3 = self._make_layer(512, 256, self.parame[2], 2)
        self.layer4 = self._make_layer(1024, 512, self.parame[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(2048, self.num_classes),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear,)):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_layer(self, i, o, n, s):
        return self.blk(i, o, n, s)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def cspresnet50(num_classes=1000):

    model = resnet(
        num_classes,
        cspneck,
        parame=[3, 4, 6, 3],
    )

    return model


def cspresnet101(num_classes=1000):

    model = resnet(
        num_classes,
        cspneck,
        parame=[3, 4, 23, 3],
    )

    return model


if __name__ == '__main__':
    model = cspresnet101()
    # print(model.state_dict())

    print(model)
