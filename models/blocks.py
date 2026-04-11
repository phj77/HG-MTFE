import torch
import torch.nn as nn

class conv2d_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=stride, padding=padding, groups=groups),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.block(x)
        return x
    
class conv1d_block(nn.modules):
    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0, groups=1):
        super.__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=stride, padding=padding, groups=groups),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.block(x)
        return x