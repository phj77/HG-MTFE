import torch
import torch.nn as nn
from blocks import conv1d_block,conv2d_block

class HGA(nn.modules): #in: (1,1,c) x2 , out: (1,1,c)
    def __init__(self, in_ch):
        super.__init__()
        self.layer1 = conv1d_block(in_ch= 2, out_ch=1, kernel=1, stride=1)
        self.layer2_img = conv1d_block(in_ch= 1, out_ch=1, kernel=3, stride=1, padding=1)
        self.layer2_hist = conv1d_block(in_ch= 1, out_ch=1, kernel=3, stride=1, padding=1)
        self.layer3_img = nn.Linear(in_ch, in_ch)  # FC - RELU -FC -SIGMOID 구조 필요 X?
        self.layer3_hist = nn.Linear(in_ch, in_ch)
        self.layer4 = nn.Linear(in_ch, in_ch)

    def forward(self, f, h): #f,h:(b,c,1,1)
        f = f.squeeze(3)
        h = h.squeeze(3) #f,h:(b,c,1)
        x = torch.cat((f,h), dim=2)
        x = x.permute(0, 2, 1) #(b,2,c)
        x = self.layer1(x) #(b,1,c)
        x_f = self.layer2_img(x)
        x_h = self.layer2_hist(x)
        fx = f*x_f
        hx = h*x_h
        k = fx * (self.layer3_img(fx))
        q = hx * (self.layer3_hist(hx)) #(b,1,c)

        k = k.permute(0,2,1) #(b,c,1)
        k_norm = torch.linalg.norm(k, ord='fro', dim=(1,2), keepdim = True)
        q_norm = torch.linalg.norm(q, ord='fro', dim=(1,2), keepdim = True)
        r = torch.matmul(k, q.permute(0,1,3,2)) / k_norm * q_norm #(b,c,c)

        f_r = torch.matmul(r, f) #(b,c,1)
        g = f + f_r + self.layer4(f + f_r)

        return g #(b,c,1)

