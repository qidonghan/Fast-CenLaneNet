from model_zoo import PostBranch
import torch
import torch.nn as nn
import math
import torchvision.models as models
from collections import OrderedDict

class PositionEmbeddingSine2(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature   = temperature
        self.normalize     = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, h, w):
        y_embed = torch.arange(h).cuda().unsqueeze(1).repeat(1, w)
        x_embed = torch.arange(w).cuda().unsqueeze(0).repeat(h, 1)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed + 0.5) / (h + eps) * self.scale
            x_embed = (x_embed + 0.5) / (w + eps) * self.scale
        dim_t = self.temperature ** (
            2 * torch.div(torch.arange(self.num_pos_feats).cuda(), 2, rounding_mode='floor') / self.num_pos_feats
        )
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=3).flatten(2)
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2)
        pos = pos.permute(2, 0, 1).unsqueeze(0)
        return pos

class SelfBN2d(nn.Module):
    def __init__(self, num_features):
        super(SelfBN2d, self).__init__()
        half = num_features // 2
        self.bn = nn.BatchNorm2d(half)

    def forward(self, x):
        half_channels = x.shape[1] // 2
        x1, x2 = x[:, :half_channels, :, :], x[:, half_channels:, :, :]
        x1 = self.bn(x1)
        x = torch.cat([x1, x2], dim=1)
        return x

class LearnableSpatialSimilarityAttention(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        self.H = H
        self.W = W
        self.N = H * W
        hidden_dim = self.N // 2
        self.reduce_mlp = nn.Sequential(
            nn.Linear(self.N, 1),
            nn.ReLU(inplace=True)
        )
        self.expand_mlp = nn.Sequential(
            nn.Linear(self.N, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.N)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.H and W == self.W
        N = self.N
        x_flat = x.view(B, C, N)
        sim = torch.bmm(x_flat.transpose(1, 2), x_flat)
        sim = sim / (C ** 0.5)
        sim_feat = sim
        reduced_score = self.reduce_mlp(sim_feat).squeeze(-1)
        refined_score = self.expand_mlp(reduced_score)
        mask = torch.sigmoid(refined_score).view(B, 1, H, W)
        return mask

class CBAM(nn.Module):
    def __init__(self, H, W):
        super(CBAM, self).__init__()
        self.sa = LearnableSpatialSimilarityAttention(H, W)

    def forward(self, x):
        mask = self.sa(x)
        return x * mask

class GhostConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, stride=1, padding=0, relu=True):
        super().__init__()
        init_channels = math.ceil(out_channels / ratio)
        new_channels = out_channels - init_channels
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, 1, 1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)

class GhostBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(GhostBasicBlock, self).__init__()
        self.conv1 = GhostConv(inplanes, planes, kernel_size=3, stride=stride, padding=1, relu=True)
        self.conv2 = GhostConv(planes, planes, kernel_size=3, stride=1, padding=1, relu=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class SlimResNet18(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        full_resnet = models.resnet18(pretrained=pretrained)
        full_resnet.layer1 = self._replace_with_ghost(
            original_layer=full_resnet.layer1,
            inplanes=64, planes=64, num_blocks=2, stride=1
        )
        full_resnet.layer2 = self._replace_with_ghost(
            original_layer=full_resnet.layer2,
            inplanes=64, planes=128, num_blocks=2, stride=2
        )
        self.features = nn.Sequential(OrderedDict([
            ('conv1',    full_resnet.conv1),
            ('bn1',      SelfBN2d(full_resnet.bn1.num_features)),
            ('relu',     full_resnet.relu),
            ('maxpool',  full_resnet.maxpool),
            ('layer1',   full_resnet.layer1),
            ('layer2',   full_resnet.layer2),
        ]))

    def _replace_with_ghost(self, original_layer, inplanes, planes, num_blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers.append(GhostBasicBlock(inplanes, planes, stride, downsample))
        for _ in range(1, num_blocks):
            layers.append(GhostBasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)

class LaneNet(nn.Module):
    def __init__(self,backbone,config=None):
        super(LaneNet,self).__init__()
        self.backbone = SlimResNet18(pretrained=False)
        self.postbranch=PostBranch.PostBranch(config)
        self.c_dims = 128
        self.cbam = CBAM(32, 64)
        self.feat_combine = nn.Sequential(
            GhostConv(self.c_dims * 2, self.c_dims, kernel_size=3, padding=1),
            nn.Conv2d(self.c_dims, self.c_dims, 1)
        )
        self.pos_embeds = PositionEmbeddingSine2(num_pos_feats=64)

    def forward(self, x):
        x=self.backbone(x)
        B, _, H, W = x.shape
        pos_embed = self.pos_embeds(H, W).repeat(B, 1, 1, 1)
        x = torch.cat([x, pos_embed], dim=1)
        x = self.cbam(x)
        x = self.feat_combine(x)
        ins,seg,cen=self.postbranch(x)
        return ins,seg,cen
