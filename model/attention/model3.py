# 3. Multi-Head Self Attention (MHSA)
# 特徴：Transformerに出てくる本物の“ガチattention”
# 利点：パターン検出能力は最強クラス
# 欠点：重い、計算多い、GPU泣く
# 今回向いてるか？：精度全振り・学習時間犠牲にできるならあり

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# === Multi-Head Self Attention (MHSA) ===
class MHSA1D(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.model_name = "これattention_モデル3です。"
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)
        B, L, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B, L, self.heads, C // self.heads).transpose(1, 2), qkv)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, C)
        out = self.proj(out)
        return out.transpose(1, 2)  # back to (B, C, L)

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

# === CNN Blocks ===
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
            nn.Conv1d(16, 16, 5, padding=1), GeM(kernel_size=2), nn.BatchNorm1d(16), nn.SiLU()
        )
        self.cnn3 = nn.Sequential(nn.Conv1d(16, 32, 3, padding=1), nn.BatchNorm1d(32), nn.SiLU())
        self.cnn4 = nn.Sequential(
            nn.Conv1d(32, 32, 3, padding=1), GeM(kernel_size=2), nn.BatchNorm1d(32), nn.SiLU()
        )
        self.cnn5 = nn.Sequential(nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.SiLU())
        self.cnn6 = nn.Sequential(nn.Conv1d(64, 64, 3), nn.BatchNorm1d(64), nn.SiLU())
        self.attn = MHSA1D(dim=64)

    def forward(self, x, pos=None):
        x = self.cnn1(x); x = self.cnn2(x); x = self.cnn3(x)
        x = self.cnn4(x); x = self.cnn5(x); x = self.cnn6(x)
        x = self.attn(x)
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

class CNN1d_with_resnet(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.cnn1 = CNN1d_low()
        self.cnn2 = CNN1d_middle()  # ← MHSA入り
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