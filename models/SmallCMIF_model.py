import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property, clamp_probs
import torch.nn.functional as F

from models.Visual_encoder import Visual_encoder

# The inertial encoder for raw imu data
class Inertial_encoder(nn.Module):
    def __init__(self, opt):
        super(Inertial_encoder, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout))
        self.proj = nn.Linear(256 * 1 * 11, opt.i_f_len)

    def forward(self, x):
        # x: (N, seq_len, 11, 6)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))    # x: (N x seq_len, 11, 6)
        x = self.encoder_conv(x.permute(0, 2, 1))                 # x: (N x seq_len, 64, 11)
        out = self.proj(x.view(x.shape[0], -1))                   # out: (N x seq_len, 256)
        return out.view(batch_size, seq_len, 256)


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()

        self.visual_encoder = Visual_encoder(opt)        
        self.inertial_encoder = Inertial_encoder(opt)

    def forward(self, img, imu):
        v = torch.cat((img[:, :-1], img[:, 1:]), dim=2)
        batch_size = v.size(0)
        seq_len = v.size(1)

        # print(f"shape of IMU {imu.shape}")
        # print(f"shape of IMG {img.shape}")

        #============================================================
        # shape of IMU torch.Size([1, 101, 6])
        # shape of IMG torch.Size([1, 11, 3, 256, 512])
        # shape of visual encoder output: torch.Size([1, 10, 512])
        # shape of inertial encoder output: torch.Size([1, 10, 256])
        #============================================================


        # image CNN
        v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4))
        v = self.visual_encoder(v)
        v = v.view(batch_size, seq_len, -1)  # (batch, seq_len, fv)
        
        # IMU CNN
        imu = torch.cat([imu[:, i * 10:i * 10 + 11, :].unsqueeze(1) for i in range(seq_len)], dim=1)
        imu = self.inertial_encoder(imu)

        # print(f'shape of visual encoder output: {v.shape}')
        # print(f'shape of inertial encoder output: {imu.shape}')

        return v, imu


class CMIM(nn.Module):
    def __init__(self, opt):
        super(CMIM, self).__init__()

        self.down_v_i = nn.Linear(opt.v_f_len, opt.i_f_len)  # downsampling of v to fit i
        self.up_i_v   = nn.Linear(opt.i_f_len, opt.v_f_len)  # upsampling of i to fit v

        self.down_vi_i = nn.Linear(opt.i_f_len + opt.v_f_len, opt.i_f_len)  # downsampling of vi to fit i
        self.down_vi_v = nn.Linear(opt.i_f_len + opt.v_f_len, opt.v_f_len)  # downsampling of vi to fit v

        self.down_is_i = nn.Linear(3 * opt.i_f_len, opt.i_f_len)
        self.down_vs_v = nn.Linear(3 * opt.v_f_len, opt.v_f_len)
 
    def forward(self, v, i):
        # shape of v (visual encoder output)  : torch.Size([1, 10, 512]) [batch_size, seq_len, v_f_len]
        # shape of i (inertial encoder output): torch.Size([1, 10, 256]) [batch_size, seq_len, i_f_len]

        batch_size = v.shape[0]
        seq_len = v.shape[1]

        v = v.view(batch_size * seq_len, v.size(2))   # [batch_size*seq_len, v_f_len]
        i = i.view(batch_size * seq_len, i.size(2))   # [batch_size*seq_len, i_f_len]

        vi = torch.cat((v, i), dim=-1)

        vD = self.down_v_i(v)
        iU = self.up_i_v(i)
        vii = self.down_vi_i(vi)
        viv = self.down_vi_v(vi)

        Is =  torch.cat((i, i, i), dim=-1) + torch.cat((vii, vD, i), dim=-1)       # use I instead of i to avoid the key word of Python          
        vs =  torch.cat((v, v, v), dim=-1) + torch.cat((viv, iU, v), dim=-1) 

        Id = self.down_is_i(Is)
        vd = self.down_vs_v(vs)

        fusion_vi = torch.cat((Id, vd), dim=-1)   # [batch_size*seq_len, v_f_len]
        fusion_vi = fusion_vi.view(batch_size, seq_len, -1)
        return fusion_vi



# The pose estimation network
class Pose_RNN(nn.Module):
    def __init__(self, opt):
        super(Pose_RNN, self).__init__()

        # The main RNN network
        f_len = opt.v_f_len + opt.i_f_len
        self.rnn = nn.LSTM(
            input_size=f_len,
            hidden_size=opt.rnn_hidden_size,
            num_layers=2,
            dropout=opt.rnn_dropout_between,
            batch_first=True)

        # The output networks
        self.rnn_drop_out = nn.Dropout(opt.rnn_dropout_out)
        self.regressor = nn.Sequential(
            nn.Linear(opt.rnn_hidden_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6))

    def forward(self, fusion_vi, prev=None):
        if prev is not None:
            prev = (prev[0].transpose(1, 0).contiguous(), prev[1].transpose(1, 0).contiguous())

        # print(f"fused model input shape: {v_in.shape}, {fi.shape}")   # [1, 1, 512], [1, 1, 256]
        # print(f"fused feature shape: {fused.shape}")   # [1, 1, 768]
        
        out, hc = self.rnn(fusion_vi) if prev is None else self.rnn(fusion_vi, prev)
        out = self.rnn_drop_out(out)
        pose = self.regressor(out)

        hc = (hc[0].transpose(1, 0).contiguous(), hc[1].transpose(1, 0).contiguous())
        return pose, hc


class SmallCMIF_VIO(nn.Module):
    def __init__(self, opt):
        super(SmallCMIF_VIO, self).__init__()

        self.Encoders_net = Encoder(opt)
        self.CMIM = CMIM(opt)
        self.LSTM_net = Pose_RNN(opt)

        self.opt = opt
        
        initialization(self)

    def forward(self, img, imu, hc=None):

        fv, fi = self.Encoders_net(img=img, imu=imu)
        fusion_vi = self.CMIM(v=fv, i=fi)

        batch_size = fv.shape[0]
        seq_len = fv.shape[1]

        poses= []
        hidden = torch.zeros(batch_size, self.opt.rnn_hidden_size).to(fv.device) if hc is None else hc[0].contiguous()[:, -1, :]
        
        for i in range(seq_len):
            pose, hc = self.LSTM_net(fusion_vi=fusion_vi[:, i:i+1, :], prev=hc)

            poses.append(pose)
            hidden = hc[0].contiguous()[:, -1, :]

        poses = torch.cat(poses, dim=1)

        return poses, hc


def initialization(net):
    #Initilization
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(0)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    start, end = n//4, n//2
                    param.data[start:end].fill_(1.)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()