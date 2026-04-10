import torch
import torch.nn as nn
import torch.nn.functional as F

class conv2d_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0, groups=1):
        super().__init__()
        