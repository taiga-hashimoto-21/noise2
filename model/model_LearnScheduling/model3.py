# Test Loss (eval mode): 0.003834

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

# Generalized Mean (GeM) プーリング層
class GeM(nn.Module):
    def __init__(self, kernel_size=8, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool1d(x.clamp(min=eps).pow(p), self.kernel_size).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'

class CNN1d_low(nn.Module):


    def __init__(self, debug=False):
        super().__init__()
        # 畳み込み層1
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5),
            nn.BatchNorm1d(16),
            nn.SiLU(),
        )
        # 畳み込み層2
        self.cnn2 = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=5),
            nn.BatchNorm1d(16),
            nn.SiLU(),
        )
        # 畳み込み層3
        self.cnn3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.SiLU(),
        )
        # 畳み込み層4
        self.cnn4 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.SiLU(),
        )
        # 畳み込み層5
        self.cnn5 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )
        # 畳み込み層6
        self.cnn6 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )
        self.debug = debug

    def forward(self, x, pos=None):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)
        return x


class CNN1d_middle(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        # 畳み込み層1
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5,padding=1),
            nn.BatchNorm1d(16),
            nn.SiLU(),
        )
        # 畳み込み層2
        self.cnn2 = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=5,padding=1),
            GeM(kernel_size=2),
            nn.BatchNorm1d(16),
            nn.SiLU(),
        )
        # 畳み込み層3
        self.cnn3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3,padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
        )
        # 畳み込み層4
        self.cnn4 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3,padding=1),
            GeM(kernel_size=2),
            nn.BatchNorm1d(32),
            nn.SiLU(),
        )
        # 畳み込み層5
        self.cnn5 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3,padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )
        # 畳み込み層6
        self.cnn6 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )
        self.debug = debug

    def forward(self, x, pos=None):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)
        return x


class CNN1d_hight(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        # 畳み込み層1
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=32),
            nn.BatchNorm1d(16),
            nn.SiLU(),
        )
        # 畳み込み層2
        self.cnn2 = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=32),
            GeM(kernel_size=4),
            nn.BatchNorm1d(16),
            nn.SiLU(),
        )
        # 畳み込み層3
        self.cnn3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=32,padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
        )
        # 畳み込み層4
        self.cnn4 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=16,padding=1),
            GeM(kernel_size=4),
            nn.BatchNorm1d(32),
            nn.SiLU(),
        )
        # 畳み込み層5
        self.cnn5 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=16,padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )
        # 畳み込み層6
        self.cnn6 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=16,padding=1),
            GeM(kernel_size=2),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )
        self.debug = debug

    def forward(self, x, pos=None):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)
        return x

import timm
class CNN1d_with_resnet(nn.Module):

    def __init__(self, debug=False):
        super().__init__()
        # 畳み込み層1
        self.cnn1 = CNN1d_low()
        self.cnn2 = CNN1d_middle()
        self.cnn3 = CNN1d_hight()
        self.resnet = timm.create_model('resnet18', pretrained=False, num_classes=2)

    def forward(self, x, pos=None):
        scale_factor = 2.5e24
        x = x * scale_factor
        x = torch.log(x)
        low = x[:,:,0:80]
        middle = x[:,:,30:300]
        hight = x[:,:,300:3000]
        low = self.cnn1(low)
        middle = self.cnn2(middle)
        hight = self.cnn3(hight)
        #全てB,64,64なので、結合してB,3,64,64にする。まず1次元目に1を追加して結合
        low = low.unsqueeze(1)
        middle = middle.unsqueeze(1)
        hight = hight.unsqueeze(1)
        x = torch.cat([low,middle,hight],dim=1)
        x = self.resnet(x)
        return x
    


def train_with_onecycle(model, X, Y, device):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 100
    batch_size = 32
    num_samples = X.size(0)
    total_steps = (num_samples // batch_size + 1) * epochs  # ←イテレーションベース

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-3,         # ピーク学習率
        total_steps=total_steps, 
        pct_start=0.1,       # 最初の10%を上昇期間に
        anneal_strategy='cos',
        final_div_factor=1e4 # 最後は lr / 10000 に落ちる
    )

    model.train()
    loss_log = []

    for epoch in range(epochs):
        perm = torch.randperm(num_samples)
        running_loss = 0.0

        for i in range(0, num_samples, batch_size):
            idx = perm[i:i+batch_size]
            batch_x = X[idx]
            batch_y = Y[idx]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()  # ← OneCycle はここで毎ステップ呼ぶ！

            running_loss += loss.item() * batch_x.size(0)

        epoch_loss = running_loss / num_samples
        loss_log.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.6f}")

    return loss_log