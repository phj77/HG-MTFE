import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import conv1d_block, conv2d_block
from CS_SFC import CS_SFC

class IFE(nn.modules): # in: (n,n, c) out: (1,1,2306). 2306 = num_trans * 768
    def __init__(self, in_ch):
        super().__init__()

        cs_sfc_iter = 7
        num_trans = 3

        # x: (n,n,c)
        self.layer1 = conv2d_block(in_ch=in_ch, out_ch=in_ch*2, kernel=3, stride=1, padding=0) # x:(n,n,2c)
        self.layer2 = nn.ModuleList()
        for _ in range(cs_sfc_iter):
            self.layer2.append(CS_SFC)
        # x:((1/2)**6*n, (1/2)**6*n, 2**7*c)
        self.layer3 = conv2d_block(in_ch = 2 ** cs_sfc_iter * in_ch, out_ch = 768 * num_trans, kernel=1, stride=1)
        self.layer4 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.layer1(x)
        for layer in self.layer2:
            x = layer(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class HG_MTFE(nn.modules):
    def __init__(self):
        super.__init__()

