import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from utils import tensor2array
from medpy.metric import dc


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Locator(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch, feat_n, loss_weight):
        super(Locator, self).__init__()
        filters = [feat_n, feat_n * 2, feat_n * 4, feat_n * 8, feat_n * 16]
        self.loss_weight = loss_weight

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.ada_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        flat_e5 = self.ada_pool(e5).view(x.size(0), -1)
        out = self.fc(flat_e5)

        return out

    def evaluate(self, x, y):
        out = self.forward(x)
        out_arr = tensor2array(out, True)
        out_arr = np.clip(out_arr, a_min=0, a_max=64)
        y_arr = tensor2array(y, True)
        dist = np.linalg.norm(out_arr - y_arr)
        return dist

    def loss_function(self, x, y):
        out = self.forward(x)
        regression = F.smooth_l1_loss(out, y)
        total = regression * self.loss_weight["Regression"]
        losses = {
            'Total': total.item(),
            'Regression': regression.item()
        }
        return total, losses


class MCLocator(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch, feat_n, loss_weight):
        super(MCLocator, self).__init__()
        filters = [feat_n, feat_n * 2, feat_n * 4, feat_n * 8, feat_n * 16]
        self.loss_weight = loss_weight

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])

        self.Conv4 = conv_block(filters[2]*2, filters[3])

        self.Conv5 = nn.Sequential(
            conv_block(filters[3], filters[2]),
            conv_block(filters[2], filters[2]),
        )

        self.ada_pool = nn.AdaptiveAvgPool2d(output_size=6)

        self.fc = nn.Sequential(
            nn.Linear(9216, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        s, l = x

        s_f = self.feature_extract(s)
        l_f = self.feature_extract(l)

        l_f = self.ada_pool(l_f)

        f = torch.cat((l_f, s_f), dim=1)
        f = self.Conv4(f)

        f = self.Conv5(f)

        flat_f = torch.flatten(f, start_dim=1)
        out = self.fc(flat_f)

        return out

    def feature_extract(self, x):
        e1 = self.Conv1(x)
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        return e3

    def evaluate(self, x, y):
        out = self.forward(x)
        out_arr = tensor2array(out, True)
        out_arr = np.clip(out_arr, a_min=0, a_max=24)
        y_arr = tensor2array(y, True)
        dist = np.linalg.norm(out_arr - y_arr)
        return dist

    def loss_function(self, x, y):
        out = self.forward(x)
        regression = F.smooth_l1_loss(out, y)
        total = regression * self.loss_weight["Regression"]
        losses = {
            'Total': total.item(),
            'Regression': regression.item()
        }
        return total, losses

