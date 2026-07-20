import math
import torch
import torch.nn as nn
from .Netblock import C
import torch.nn.functional as F
from Configs import config as cfg

class GhostConvLite(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, stride=1, padding=0, groups=1, bias=True):
        super().__init__()
        init_channels = math.ceil(out_channels / ratio)
        new_channels = out_channels - init_channels
        self.primary_conv = nn.Conv2d(in_channels, init_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.cheap_operation = nn.Conv2d(
            init_channels,
            new_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=init_channels,
            bias=bias
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)

class PostBranch(nn.Module):
    def __init__(self, configs):
        super(PostBranch, self).__init__()
        self.fc1 = GhostConvLite(cfg.Model_cfg.DAhead_outputchannel, 32, kernel_size=1, padding=0)
        self.layers1 = GhostConvLite(32, 16, kernel_size=3, padding=1)
        self.layers2 = C(16, 6, 3, 1, 1)
        self.fc2 = GhostConvLite(cfg.Model_cfg.DAhead_outputchannel, 32, kernel_size=1, padding=0)
        self.layers4 = GhostConvLite(32, 16, kernel_size=3, padding=1)
        self.layers5 = C(16, 1, 3, 1, 1)
        self.fc3 = GhostConvLite(cfg.Model_cfg.DAhead_outputchannel, 32, kernel_size=1, padding=0)
        self.layers6 = GhostConvLite(32, 16, kernel_size=3, padding=1)
        self.layers9 = C(16, 2, 3, 1, 1)

    def forward(self, x):
        imsize = cfg.Dataprocess_cfg.gtSize
        up = F.interpolate(x, imsize, mode='bilinear', align_corners=True)
        ins_x1 = self.fc1(up)
        ins_x2 = self.layers1(ins_x1)
        ins_x3 = self.layers2(ins_x2)
        insres = ins_x3
        cen_x1 = self.fc2(up)
        cen_x2 = self.layers4(cen_x1)
        cen_x3 = self.layers5(cen_x2)
        cen_x4 = torch.clamp(cen_x3.sigmoid_(), min=1e-4, max=1 - 1e-4)
        cenres = cen_x4
        seg_x1 = self.fc3(up)
        seg_x2 = self.layers6(seg_x1)
        output = self.layers9(seg_x2)
        return insres, output, cenres
