import torch
import torch.nn as nn

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
        super(Visual_encoder, self).__init__()

        self.conv_layers = Conv_layers()

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.visual_head = nn.Linear(512, opt.v_f_len)

    def forward(self, img_pair):
        feat = self.conv_layers(img_pair)

        feat = self.global_avg_pool(feat)   # [B, 512, 1, 1]
        feat = feat.flatten(1)              # [B, 512]      
        feat = self.visual_head(feat)

        return feat
    

class Visual_encoder_for_Distill(nn.Module):
    def __init__(self):
        super(Visual_encoder_for_Distill, self).__init__()
        self.conv_layers = Conv_layers()
        # self.proj = nn.Conv2d(512, 1024, kernel_size=1, bias=False)

    def forward(self, img_pair):
        feat = self.conv_layers(img_pair)
        # f_s_proj = self.proj(feat)

        return feat
