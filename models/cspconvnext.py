import torch
import numpy as np
import torch.nn as nn
__all__ = [
    'cspconvnext_t',
    'cspconvnext_s'
]


# class CNBlock(nn.Module):
#     def __init__(self, dim, h, w):
#         super().__init__()
#         self.blk = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=48, bias=False),
#             # nn.LayerNorm([dim, h, w], eps=1e-6),
#             nn.BatchNorm2d(dim),
#             nn.Conv2d(dim, dim * 4, kernel_size=1),
#             # nn.GELU(),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(dim * 4, dim, kernel_size=1),
#         )
#
#     def forward(self, x):
#         return x + self.blk(x)

class CNBlock(nn.Module):
    expansion = 4

    def __init__(self, dim, groups, k=3, p=1):
        super().__init__()
        self.blk = nn.Sequential(
            nn.Conv2d(dim, dim * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim * self.expansion),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim * self.expansion, dim * self.expansion, kernel_size=k, padding=p, groups=groups),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim * self.expansion, dim, kernel_size=1),
        )

    def forward(self, x):
        return x + self.blk(x)


class csp1(nn.Module):
    def __init__(self, dim, groups, num_block, downsample=True):
        super().__init__()
        self.blk = CNBlock
        self.dim = dim // 2

        block = []
        for _ in range(num_block):
            block.append(self.blk(self.dim, groups))

        self.c1 = nn.Sequential(*block)

        trans = nn.ModuleList()
        trans.append(nn.Conv2d(dim, dim * 2, kernel_size=1, stride=1, bias=False))
        trans.append(nn.BatchNorm2d(dim * 2))
        if downsample:
            trans.append(nn.Conv2d(dim * 2, dim * 2, kernel_size=5, stride=2, padding=2, groups=dim * 2))
        trans.append(nn.LeakyReLU(inplace=True))
        self.trans = nn.Sequential(*trans)

    def forward(self, x):
        x0, x1 = x.split(int(self.dim), dim=1)
        x1 = self.c1(x1).contiguous()
        return self.trans(torch.cat((x0, x1), 1))


class convnext(nn.Module):
    def __init__(self, num_classes, block, dim, groups, parame):
        super().__init__()
        self.blk = block
        self.parame = parame
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.layer1 = self._make_layer(dim*np.power(2, 0), groups, self.parame[0], True)
        self.layer2 = self._make_layer(dim*np.power(2, 1), groups, self.parame[1], True)
        self.layer3 = self._make_layer(dim*np.power(2, 2), groups, self.parame[2], True)
        self.layer4 = self._make_layer(dim*np.power(2, 3), groups, self.parame[3], False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim * np.power(2, 4), self.num_classes)
        )


        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            # elif isinstance(m, (nn.Linear,)):
            #     nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01)
            #     if m.bias is not None:
            #         nn.init.zeros_(m.bias)

    def _make_layer(self, dim, groups, n, downsample=True):
        return self.blk(dim, groups, n, downsample)

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



def cspconvnext_t(num_classes=1000, dim=96, groups=48):

    model = convnext(
        num_classes,
        csp1,
        dim,
        groups,
        parame=[3, 3, 9, 3]
    )

    return model


def cspconvnext_s(num_classes=1000, dim=96, groups=48):

    model = convnext(
        num_classes,
        csp1,
        dim,
        groups,
        parame=[3, 3, 27, 3]
    )

    return model


if __name__ == '__main__':
    model = cspconvnext_t()
    # print(model.state_dict())

    print(model)