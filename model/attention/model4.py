# 4. Convolutional Block Attention Module (CBAM)
# 特徴：チャネル + 空間 attention のハイブリッド
# 利点：軽いのに効く。脳筋CNNに知性が宿る
# 今回向いてるか？：オシロ時系列っぽいデータには効果高いことが多い

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# === GeM プーリング層 ===
class GeM(nn.Module):
    def __init__(self, kernel_size=8, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        return F.avg_pool1d(x.clamp(min=self.eps).pow(self.p), self.kernel_size).pow(1. / self.p)

# === CBAM Block ===
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM1D(nn.Module):
    def __init__(self, channels):
        super(CBAM1D, self).__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# === 各 CNN モジュール ===
class CNN1d_low(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
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
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)

class CNN1d_middle(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=1),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=5, padding=1),
            GeM(kernel_size=4),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            GeM(kernel_size=4),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )
        self.cbam = CBAM1D(64)

    def forward(self, x):
        x = self.net(x)
        x = self.cbam(x)
        return x

class CNN1d_hight(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
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
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)

# === 統合モデル ===
class CNN1d_with_resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = CNN1d_low()
        self.cnn2 = CNN1d_middle()
        self.cnn3 = CNN1d_hight()
        self.resnet = timm.create_model('resnet18', pretrained=False, num_classes=2, in_chans=3)

    def forward(self, x):
        scale_factor = 2.5e24
        x = x * scale_factor
        x = torch.log(x)

        low = self.cnn1(x[:, :, 0:80])
        middle = self.cnn2(x[:, :, 30:300])
        hight = self.cnn3(x[:, :, 300:3000])

        low = F.adaptive_avg_pool1d(low, 64).unsqueeze(1)
        middle = F.adaptive_avg_pool1d(middle, 64).unsqueeze(1)
        hight = F.adaptive_avg_pool1d(hight, 64).unsqueeze(1)

        x = torch.cat([low, middle, hight], dim=1)
        x = self.resnet(x)
        return x
