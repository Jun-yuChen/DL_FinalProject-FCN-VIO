import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=0, dropout=0.3, batchNorm=True):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )
    

class Conv_layers(nn.Module):
    def __init__(self):
        super(Conv_layers, self).__init__()
        self.feature_net = torch.nn.Sequential(
            conv(2, 32, kernel_size=7, stride=2, padding=3, batchNorm=False),
            conv(32, 32, stride=1, padding=1, batchNorm=False),
        )

        self.matching_net = torch.nn.Sequential(
            conv(32, 64  , stride=2, padding=0, batchNorm=True),
            conv(64, 128 , stride=2, padding=0, batchNorm=True),
            conv(128, 256, stride=2, padding=0, batchNorm=True),
            conv(256, 512, stride=2, padding=0, batchNorm=True),
            conv(512, 1024, stride=2, padding=0, batchNorm=True),
        )

    def  forward(self, img_pair):
        feat = self.feature_net(img_pair)
        feat = self.matching_net(feat)

        return feat


class Tiny_Visual_encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.conv_layers = Conv_layers()

        # compute feature dimension safely
        with torch.no_grad():
            tmp = torch.zeros(1, 2, opt.img_w, opt.img_h)
            tmp = self.conv_layers(tmp)
            feat_dim = tmp.flatten(1).size(1)

        self.visual_head = nn.Linear(feat_dim, opt.v_f_len)

    def forward(self, img_pair):
        feat = self.conv_layers(img_pair)

        feat = feat.view(feat.size(0), -1)
        feat = self.visual_head(feat)

        return feat
    
    def load_conv_layers(self, path):
        ckpt = torch.load(path, map_location="cpu")

        # extract only conv_layers.*
        conv_state_dict = {
            k.replace("conv_layers.", ""): v
            for k, v in ckpt.items()
            if k.startswith("conv_layers.")
        }

        self.conv_layers.load_state_dict(conv_state_dict)


#============================= For training only ====================================

def convert_rgb_pair_conv_to_gray(W_old):
    """
    W_old: (C_out, 6, kH, kW)
    returns: (C_out, 2, kH, kW)
    """
    r1, g1, b1, r2, g2, b2 = torch.split(W_old, 1, dim=1)

    W1 = 0.299 * r1 + 0.587 * g1 + 0.114 * b1
    W2 = 0.299 * r2 + 0.587 * g2 + 0.114 * b2

    return torch.cat([W1, W2], dim=1)


class Visual_encoder_for_Distill(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = Conv_layers()

    def rgb_to_gray(self, x):
        # x: (B, 3, H, W)
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def forward(self, img_pair):
        # img_pair: (B, 6, H, W)
        img1, img2 = torch.split(img_pair, 3, dim=1)

        gray1 = self.rgb_to_gray(img1)
        gray2 = self.rgb_to_gray(img2)

        gray_pair = torch.cat([gray1, gray2], dim=1)  # (B, 2, H, W)

        feat = self.conv_layers(gray_pair)
        return feat
    
    def load_from_rgb_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")

        new_state = self.conv_layers.state_dict()

        for k, v in ckpt.items():
            if not k.startswith("conv_layers."):
                continue

            key = k.replace("conv_layers.", "")

            # Special case: first conv weight
            if key == "feature_net.0.0.weight":
                v = convert_rgb_pair_conv_to_gray(v)

            if key in new_state:
                new_state[key] = v

        self.conv_layers.load_state_dict(new_state)
