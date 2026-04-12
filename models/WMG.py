import torch
import torch.nn as nn
from blocks import conv1d_block,conv2d_block

class WMG(nn.modules):
    def __init__(self):
        super.__init__()