import torch
import torch.nn as nn
from blocks import conv1d_block,conv2d_block

class SFC_module(nn.Module):
    def __init__(self, in_ch, out_ch, expansion, num):
        super(SFC_module, self).__init__()
        exp_ch = int(in_ch * expansion)
        if num == 1:
            self.se_conv = nn.Conv2d(in_ch, exp_ch, 3, 1, 1, groups=in_ch)
        else:
            self.se_conv = nn.Conv2d(in_ch, exp_ch, 3, 2, 1, groups=in_ch)
        self.se_bn = nn.BatchNorm2d(exp_ch)
        self.se_relu = nn.ReLU()
        self.hd_conv = nn.Conv2d(exp_ch, exp_ch, 3, 1, 1, groups=in_ch)
        self.hd_bn = nn.BatchNorm2d(exp_ch)
        self.hd_relu = nn.ReLU()
        self.cp_conv = nn.Conv2d(exp_ch, out_ch, 1, 1, groups=in_ch)
        self.cp_bn = nn.BatchNorm2d(out_ch)
        self.pw_conv = nn.Conv2d(out_ch, out_ch, 1, 1)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.pw_relu = nn.ReLU()
        self.ca_gap = nn.AdaptiveAvgPool2d(1)
        self.ca_map = nn.AdaptiveMaxPool2d(1)
        self.ca_conv = nn.Conv1d(1, 1, 3, 1, 1)
        self.ca_sig = nn.Sigmoid()

    def forward(self, x):
        x = self.se_conv(x)
        x = self.se_bn(x)
        x = self.se_relu(x)
        x = self.hd_conv(x)
        x = self.hd_bn(x)
        x = self.hd_relu(x)
        x = self.cp_conv(x)
        x = self.cp_bn(x)
        x = self.pw_conv(x)
        x = self.pw_bn(x)
        x = self.pw_relu(x)

        aa = self.ca_gap(x)
        ma = self.ca_map(x)
        aa = aa.squeeze(3)
        aa = aa.permute(0, 2, 1)
        ma = ma.squeeze(3)
        ma = ma.permute(0, 2, 1)
        aa = self.ca_conv(aa)
        ma = self.ca_conv(ma)
        a = aa + ma
        a = a.permute(0, 2, 1)
        a = self.ca_sig(a)
        a = a.unsqueeze(3)
        x = x * a
        return x


class channel_scale(nn.modules):
    def __init__(self, in_ch):
        super.__init__(self)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.conv_1d = conv1d_block(in_ch=in_ch,out_ch=in_ch, kernel=3, stride=1, padding=1)
        self.sig = nn.Sigmoid()

    def forward(self, x): # x:(n,n,c)

        ###### shape 변환 -> gap / conv 방향 조절 필요!!!!!!!!!!!!!!!!!!!!!!!!!!!
        x_a = self.gap(x)
        x_m = self.gmp(x)
        x_a = self.conv_1d(x_a)
        x_m = self.conv_1d(x_m)

        s = x_a + x_m
        s = self.sig(s)
        x = x*s # x:(n,n,c)
        return x
        

class CS_SFC(nn.modules): # in_channel: out_channel: kernel:
    def __init__(self, in_ch):
        super.__init__()
        expansion = 4
        self.sfc_iteration = 7
        in_channel = in_ch
        self.sfc = nn.ModuleList()
        for i in range(1, self.sfc_iteration+1):

            # x:(n,n,c)
            self.sfc.append(SFC_module(in_ch=in_channel, out_ch=in_channel*2 ,expansion=expansion, num=i)) 
            # x:(n/2, n/2, c*2)

            in_channel = in_channel*2
            self.sfc.append(channel_scale(in_channel))

            

    def forward(self, x): # x:(256, 256, 6)
        for sfc_layer in self.sfc:
            x = sfc_layer(x)
        # x:(n,n,c)

        

