import torch
import torch.nn as nn
import torch.nn.functional as F


class MrnCombiner(nn.Module):

    def __init__(self, config):
        super(MrnCombiner, self).__init__()
        self.config = config
        self.residual_conv = nn.Conv2d(512, 512, kernel_size=1)
        self.f1_conv1 = nn.Conv2d(512, 512, kernel_size=1)
        self.f1_conv2 = nn.Conv2d(512, 512, kernel_size=1)
        self.f2_conv1 = nn.Conv2d(512, 512, kernel_size=1)
        self.f2_conv2 = nn.Conv2d(512, 512, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, features):

        rez = features[0]
        for i in range(1, len(features)):
            rez = self.combine(rez, features[i])
        return rez


    def combine(self, f1, f2):
        f1 = f1.unsqueeze(2).unsqueeze(2)
        f2 = f2.unsqueeze(2).unsqueeze(2)
        residual = self.residual_conv(f1)
        f1_path = self.tanh(self.f1_conv1(f1))
        f1_path = self.tanh(self.f1_conv2(f1_path))
        f2_path = self.tanh(self.f2_conv1(f2))
        f2_path = self.tanh(self.f2_conv2(f2_path))
        out = residual + f1_path * f2_path
        return out.squeeze()