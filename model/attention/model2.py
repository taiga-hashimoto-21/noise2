# 2. Squeeze-and-Excitation (SE) Block
# 特徴：チャネル方向だけを見て“重要な特徴だけを強調”
# 利点：軽いし、CNNにそのまま挿せる
# 今回向いてるか？：ResNet入れてるならこいつは“相性のいい親戚”

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# === SE Block ===
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# === GeM ===
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

# === 各 CNN ブロック ===
class CNN1d_low(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.cnn1 = nn.Sequential(nn.Conv1d(1, 16, 5), nn.BatchNorm1d(16), nn.SiLU())
        self.cnn2 = nn.Sequential(nn.Conv1d(16, 16, 5), nn.BatchNorm1d(16), nn.SiLU())
        self.cnn3 = nn.Sequential(nn.Conv1d(16, 32, 3), nn.BatchNorm1d(32), nn.SiLU())
        self.cnn4 = nn.Sequential(nn.Conv1d(32, 32, 3), nn.BatchNorm1d(32), nn.SiLU())
        self.cnn5 = nn.Sequential(nn.Conv1d(32, 64, 3), nn.BatchNorm1d(64), nn.SiLU())
        self.cnn6 = nn.Sequential(nn.Conv1d(64, 64, 3), nn.BatchNorm1d(64), nn.SiLU())

    def forward(self, x, pos=None):
        x = self.cnn1(x); x = self.cnn2(x); x = self.cnn3(x)
        x = self.cnn4(x); x = self.cnn5(x); x = self.cnn6(x)
        return x

class CNN1d_middle(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.cnn1 = nn.Sequential(nn.Conv1d(1, 16, 5, padding=1), nn.BatchNorm1d(16), nn.SiLU())
        self.cnn2 = nn.Sequential(
            nn.Conv1d(16, 16, 5, padding=1),
            GeM(kernel_size=2),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            SEBlock(16)  # ←ここで SE を追加
        )
        self.cnn3 = nn.Sequential(nn.Conv1d(16, 32, 3, padding=1), nn.BatchNorm1d(32), nn.SiLU())
        self.cnn4 = nn.Sequential(
            nn.Conv1d(32, 32, 3, padding=1),
            GeM(kernel_size=2),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            SEBlock(32)
        )
        self.cnn5 = nn.Sequential(nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.SiLU())
        self.cnn6 = nn.Sequential(nn.Conv1d(64, 64, 3), nn.BatchNorm1d(64), nn.SiLU())

    def forward(self, x, pos=None):
        x = self.cnn1(x); x = self.cnn2(x); x = self.cnn3(x)
        x = self.cnn4(x); x = self.cnn5(x); x = self.cnn6(x)
        return x

class CNN1d_hight(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.cnn1 = nn.Sequential(nn.Conv1d(1, 16, 32), nn.BatchNorm1d(16), nn.SiLU())
        self.cnn2 = nn.Sequential(nn.Conv1d(16, 16, 32), GeM(kernel_size=4), nn.BatchNorm1d(16), nn.SiLU())
        self.cnn3 = nn.Sequential(nn.Conv1d(16, 32, 32, padding=1), nn.BatchNorm1d(32), nn.SiLU())
        self.cnn4 = nn.Sequential(nn.Conv1d(32, 32, 16, padding=1), GeM(kernel_size=4), nn.BatchNorm1d(32), nn.SiLU())
        self.cnn5 = nn.Sequential(nn.Conv1d(32, 64, 16, padding=1), nn.BatchNorm1d(64), nn.SiLU())
        self.cnn6 = nn.Sequential(nn.Conv1d(64, 64, 16, padding=1), GeM(kernel_size=2), nn.BatchNorm1d(64), nn.SiLU())

    def forward(self, x, pos=None):
        x = self.cnn1(x); x = self.cnn2(x); x = self.cnn3(x)
        x = self.cnn4(x); x = self.cnn5(x); x = self.cnn6(x)
        return x

# === 最終モデル ===
class CNN1d_with_resnet(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.cnn1 = CNN1d_low()
        self.cnn2 = CNN1d_middle()  # ← SEBlock入り
        self.cnn3 = CNN1d_hight()
        self.resnet = timm.create_model('resnet18', pretrained=False, num_classes=2)

    def forward(self, x, pos=None):
        scale_factor = 2.5e24
        x = x * scale_factor
        x = torch.log(x)

        low = self.cnn1(x[:,:,0:80])
        middle = self.cnn2(x[:,:,30:300])
        hight = self.cnn3(x[:,:,300:3000])

        low = low.unsqueeze(1)
        middle = middle.unsqueeze(1)
        hight = hight.unsqueeze(1)

        x = torch.cat([low, middle, hight], dim=1)
        x = self.resnet(x)
        return x
