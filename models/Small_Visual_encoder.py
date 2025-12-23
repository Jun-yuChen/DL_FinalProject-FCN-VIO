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
            conv(6, 32, kernel_size=7, stride=2, padding=3, batchNorm=False),
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


class Visual_encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.conv_layers = Conv_layers()

        # compute feature dimension safely
        with torch.no_grad():
            tmp = torch.zeros(1, 6, opt.img_w, opt.img_h)
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

    

class Visual_encoder_for_Distill(nn.Module):
    def __init__(self):
        super(Visual_encoder_for_Distill, self).__init__()
        self.conv_layers = Conv_layers()
        # self.proj = nn.Conv2d(512, 1024, kernel_size=1, bias=False)

    def forward(self, img_pair):
        feat = self.conv_layers(img_pair)
        # f_s_proj = self.proj(feat)

        return feat
