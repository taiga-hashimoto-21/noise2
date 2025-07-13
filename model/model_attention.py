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

# ★ 新規追加: Attention機構のクラス
class ChannelAttention(nn.Module):
    """チャンネル注意機構 - どのチャンネル（特徴）が重要かを学習"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        # グローバル平均プーリング → 小さなMLP → Sigmoid
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 各チャンネルの平均値を計算
            nn.Conv1d(channels, channels // reduction, 1, bias=False),  # 次元削減
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1, bias=False),  # 元の次元に戻す
            nn.Sigmoid()  # 0-1の重みを生成
        )
    
    def forward(self, x):
        # 注意重みを計算して元の特徴マップに掛け算
        attention_weights = self.attention(x)
        return x * attention_weights

class SpatialAttention(nn.Module):
    """空間注意機構 - どの位置（時間軸）が重要かを学習"""
    def __init__(self, kernel_size=7):
        super().__init__()
        # チャンネル方向の平均と最大を使って空間的な重要度を計算
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # チャンネル方向の平均と最大を計算
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, L]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, L]
        
        # 平均と最大を結合して空間注意を計算
        combined = torch.cat([avg_out, max_out], dim=1)  # [B, 2, L]
        attention_map = self.sigmoid(self.conv(combined))  # [B, 1, L]
        
        return x * attention_map

class CBAM(nn.Module):
    """CBAM (Convolutional Block Attention Module) - チャンネルと空間の両方の注意機構"""
    def __init__(self, channels, reduction=8, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        # 1. チャンネル注意を適用
        x = self.channel_attention(x)
        # 2. 空間注意を適用
        x = self.spatial_attention(x)
        return x

# Generalized Mean (GeM) プーリング層（元のまま）
class GeM(nn.Module):
    def __init__(self, kernel_size=8, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool1d(x.clamp(min=eps).pow(p), self.kernel_size).pow(1. / p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.item():.4f}, eps={self.eps})"

# ★ 修正: Attention機構を追加したCNN1d_low
class CNN1d_low(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
        
        # 既存のConv層
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(16)
        
        self.conv2 = nn.Conv1d(16, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(16)
        # ★ 新規追加: 16チャンネル用のAttention
        self.attention1 = CBAM(16)
        
        self.conv3 = nn.Conv1d(16, 32, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(32)
        
        self.conv4 = nn.Conv1d(32, 32, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(32)
        # ★ 新規追加: 32チャンネル用のAttention
        self.attention2 = CBAM(32)
        
        self.conv5 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn5 = nn.BatchNorm1d(64)
        
        self.conv6 = nn.Conv1d(64, 64, kernel_size=3)
        self.bn6 = nn.BatchNorm1d(64)
        # ★ 新規追加: 64チャンネル用のAttention
        self.attention3 = CBAM(64)

    def forward(self, x, pos=None):
        # Layer 1-2: 基本的な特徴抽出
        x = F.silu(self.bn1(self.conv1(x)))
        x = F.silu(self.bn2(self.conv2(x)))
        # ★ 修正: Attention適用で重要な特徴を強調
        x = self.attention1(x)
        
        # Layer 3-4: より複雑な特徴抽出
        x = F.silu(self.bn3(self.conv3(x)))
        x = F.silu(self.bn4(self.conv4(x)))
        # ★ 修正: Attention適用で重要な特徴を強調
        x = self.attention2(x)
        
        # Layer 5-6: 高レベル特徴抽出
        x = F.silu(self.bn5(self.conv5(x)))
        x = F.silu(self.bn6(self.conv6(x)))
        # ★ 修正: 最終的なAttention適用
        x = self.attention3(x)
        
        return x

# ★ 修正: Attention機構を追加したCNN1d_middle
class CNN1d_middle(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
        
        # 既存のConv層
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        
        self.conv2 = nn.Conv1d(16, 16, kernel_size=5, padding=1)
        self.gem1 = GeM(kernel_size=2)
        self.bn2 = nn.BatchNorm1d(16)
        # ★ 新規追加: 16チャンネル用のAttention
        self.attention1 = CBAM(16)
        
        self.conv3 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        
        self.conv4 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.gem2 = GeM(kernel_size=2)
        self.bn4 = nn.BatchNorm1d(32)
        # ★ 新規追加: 32チャンネル用のAttention
        self.attention2 = CBAM(32)
        
        self.conv5 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(64)
        
        self.conv6 = nn.Conv1d(64, 64, kernel_size=3)
        self.bn6 = nn.BatchNorm1d(64)
        # ★ 新規追加: 64チャンネル用のAttention
        self.attention3 = CBAM(64)

    def forward(self, x, pos=None):
        # Layer 1-2: パディング付きで情報保持
        x = F.silu(self.bn1(self.conv1(x)))
        x = self.gem1(F.silu(self.bn2(self.conv2(x))))
        # ★ 修正: Attention適用で重要な特徴を強調
        x = self.attention1(x)
        
        # Layer 3-4: 中間レベル特徴抽出
        x = F.silu(self.bn3(self.conv3(x)))
        x = self.gem2(F.silu(self.bn4(self.conv4(x))))
        # ★ 修正: Attention適用で重要な特徴を強調
        x = self.attention2(x)
        
        # Layer 5-6: 高レベル特徴抽出
        x = F.silu(self.bn5(self.conv5(x)))
        x = F.silu(self.bn6(self.conv6(x)))
        # ★ 修正: 最終的なAttention適用
        x = self.attention3(x)
        
        return x

# ★ 修正: Attention機構を追加したCNN1d_hight
class CNN1d_hight(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
        
        # 既存のConv層（大きなkernel_sizeで広範囲の特徴を捉える）
        self.conv1 = nn.Conv1d(1, 16, kernel_size=32)
        self.bn1 = nn.BatchNorm1d(16)
        
        self.conv2 = nn.Conv1d(16, 16, kernel_size=32)
        self.gem1 = GeM(kernel_size=4)
        self.bn2 = nn.BatchNorm1d(16)
        # ★ 新規追加: 16チャンネル用のAttention
        self.attention1 = CBAM(16)
        
        self.conv3 = nn.Conv1d(16, 32, kernel_size=32, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        
        self.conv4 = nn.Conv1d(32, 32, kernel_size=16, padding=1)
        self.gem2 = GeM(kernel_size=4)
        self.bn4 = nn.BatchNorm1d(32)
        # ★ 新規追加: 32チャンネル用のAttention
        self.attention2 = CBAM(32)
        
        self.conv5 = nn.Conv1d(32, 64, kernel_size=16, padding=1)
        self.bn5 = nn.BatchNorm1d(64)
        
        self.conv6 = nn.Conv1d(64, 64, kernel_size=16, padding=1)
        self.gem3 = GeM(kernel_size=2)
        self.bn6 = nn.BatchNorm1d(64)
        # ★ 新規追加: 64チャンネル用のAttention
        self.attention3 = CBAM(64)

    def forward(self, x, pos=None):
        # Layer 1-2: 大きなカーネルで広範囲パターン検出
        x = F.silu(self.bn1(self.conv1(x)))
        x = self.gem1(F.silu(self.bn2(self.conv2(x))))
        # ★ 修正: Attention適用で重要な特徴を強調
        x = self.attention1(x)
        
        # Layer 3-4: 中間レベル特徴抽出
        x = F.silu(self.bn3(self.conv3(x)))
        x = self.gem2(F.silu(self.bn4(self.conv4(x))))
        # ★ 修正: Attention適用で重要な特徴を強調
        x = self.attention2(x)
        
        # Layer 5-6: 高レベル特徴抽出
        x = F.silu(self.bn5(self.conv5(x)))
        x = self.gem3(F.silu(self.bn6(self.conv6(x))))
        # ★ 修正: 最終的なAttention適用
        x = self.attention3(x)
        
        return x

# メインモデル（元のまま）
class CNN1d_with_resnet(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
        # ★ 修正済み: 各CNNにAttention機構が追加されている
        self.cnn1 = CNN1d_low()
        self.cnn2 = CNN1d_middle()
        self.cnn3 = CNN1d_hight()
        self.resnet = timm.create_model('resnet18', pretrained=False, num_classes=2)

    def forward(self, x, pos=None):
        scale_factor = 2.5e24
        x = torch.log(x * scale_factor)

        # 3つの周波数帯域で並列処理（Attention付きCNNで処理）
        low = self.cnn1(x[:, :, 0:80])      # 低周波数：0-80
        middle = self.cnn2(x[:, :, 30:300])  # 中周波数：30-300
        hight = self.cnn3(x[:, :, 300:3000]) # 高周波数：300-3000

        # ResNetの入力形式に合わせて次元追加
        low = low.unsqueeze(1)
        middle = middle.unsqueeze(1)
        hight = hight.unsqueeze(1)

        # 3つの特徴マップを結合してResNetに入力
        x = torch.cat([low, middle, hight], dim=1)
        return self.resnet(x)