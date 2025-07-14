import os
import time
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm

# Generalized Mean (GeM) プーリング層
class GeM(nn.Module):
    def __init__(self, kernel_size=8, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool1d(x.clamp(min=eps).pow(p), self.kernel_size).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + '(p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', eps=' + str(self.eps) + ')'


class CNN1d_low(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=5),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.SiLU()
        )
        self.pool = nn.AdaptiveAvgPool1d(64)
        self.debug = debug

    def forward(self, x, pos=None):
        x = self.cnn(x)
        x = self.pool(x)
        return x


class CNN1d_middle(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=1),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=5, padding=1),
            GeM(kernel_size=2),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            GeM(kernel_size=2),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.SiLU()
        )
        self.pool = nn.AdaptiveAvgPool1d(64)
        self.debug = debug

    def forward(self, x, pos=None):
        x = self.cnn(x)
        x = self.pool(x)
        return x


class CNN1d_hight(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=32),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=32),
            GeM(kernel_size=4),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=32, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=16, padding=1),
            GeM(kernel_size=4),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=16, padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=16, padding=1),
            GeM(kernel_size=2),
            nn.BatchNorm1d(64),
            nn.SiLU()
        )
        self.pool = nn.AdaptiveAvgPool1d(64)
        self.debug = debug

    def forward(self, x, pos=None):
        x = self.cnn(x)
        x = self.pool(x)
        return x


class CNN1d_with_resnet(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.cnn1 = CNN1d_low()
        self.cnn2 = CNN1d_middle()
        self.cnn3 = CNN1d_hight()
        self.resnet = timm.create_model('resnet18', pretrained=False, num_classes=2)

    def forward(self, x, pos=None):
        scale_factor = 2.5e24
        x = x * scale_factor
        x = torch.log(x)
        low = x[:, :, 0:256]
        middle = x[:, :, 256:1024]
        hight = x[:, :, 1024:3000]

        low = self.cnn1(low).unsqueeze(1)
        middle = self.cnn2(middle).unsqueeze(1)
        hight = self.cnn3(hight).unsqueeze(1)

        x = torch.cat([low, middle, hight], dim=1)  # shape: (B, 3, 64, 64)
        x = self.resnet(x)
        return x
