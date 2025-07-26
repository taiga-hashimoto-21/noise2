# 1. SimpleAttention（今入れたやつ）
# 特徴：self-attention の“超軽量版”
# 利点：導入が楽で軽いから、とりあえず試したいときにちょうどいい
# 今回向いてるか？：データが短くてパターンが明瞭なら合う。すでに使ってるからこれはOK。

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

# SimpleAttention モジュール（超シンプルな自己注意）
class SimpleAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key   = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)
        weights = F.softmax(scores, dim=-1)
        out = torch.bmm(weights, V)
        return out.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)

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
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class CNN1d_low(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5), nn.BatchNorm1d(16), nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=5), nn.BatchNorm1d(16), nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=3), nn.BatchNorm1d(32), nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=3), nn.BatchNorm1d(32), nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=3), nn.BatchNorm1d(64), nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3), nn.BatchNorm1d(64), nn.SiLU(),
        )
        self.debug = debug

    def forward(self, x, pos=None):
        return self.cnn1(x)

class CNN1d_middle(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=1), nn.BatchNorm1d(16), nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=5, padding=1), GeM(kernel_size=2), nn.BatchNorm1d(16), nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1), GeM(kernel_size=2), nn.BatchNorm1d(32), nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3), nn.BatchNorm1d(64), nn.SiLU(),
        )
        self.attn = SimpleAttention(64)
        self.debug = debug

    def forward(self, x, pos=None):
        x = self.cnn1(x)
        x = self.attn(x)
        return x

class CNN1d_hight(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=32), nn.BatchNorm1d(16), nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=32), GeM(kernel_size=4), nn.BatchNorm1d(16), nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=32, padding=1), nn.BatchNorm1d(32), nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=16, padding=1), GeM(kernel_size=4), nn.BatchNorm1d(32), nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=16, padding=1), nn.BatchNorm1d(64), nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=16, padding=1), GeM(kernel_size=2), nn.BatchNorm1d(64), nn.SiLU(),
        )
        self.debug = debug

    def forward(self, x, pos=None):
        return self.cnn1(x)

import timm
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
        low = x[:, :, 0:80]
        middle = x[:, :, 30:300]
        hight = x[:, :, 300:3000]
        low = self.cnn1(low)
        middle = self.cnn2(middle)
        hight = self.cnn3(hight)
        low = low.unsqueeze(1)
        middle = middle.unsqueeze(1)
        hight = hight.unsqueeze(1)
        x = torch.cat([low, middle, hight], dim=1)
        x = self.resnet(x)
        return x