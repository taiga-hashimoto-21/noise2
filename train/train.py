# ─── Google Drive をマウント ───────
from google.colab import drive
drive.mount('/content/drive')

# ─── noise フォルダの中を import 可能にする ─────
import sys
sys.path.append('/content/drive/MyDrive/noise')

# ─── ライブラリ ───────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# noise フォルダから model と loss を import
from model_7_11 import CNN1d_with_resnet
from loss_function import WeightedMSELoss

# ─── 保存先ディレクトリ作成（タイムスタンプ付き） ─────
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'/content/drive/MyDrive/noise/model_logs/{timestamp}'
os.makedirs(output_dir, exist_ok=True)

# ─── ノイズ関数の定義 ─────────────────
def process_noise(noise, clip_range=0.5, smoothing_factor=0.1):
    scaled_noise = noise / clip_range
    processed_noise = torch.tanh(scaled_noise) * clip_range
    smoothed_noise = processed_noise * (1 - smoothing_factor) + noise * smoothing_factor
    return smoothed_noise   

def add_structured_noise(batch_x):
    x = torch.linspace(1, 3000, 3000, device=batch_x.device)
    var = 0.2 + 0.1 * x / 1000
    var = torch.clamp(var, max=0.3)
    std = torch.sqrt(var).unsqueeze(0).unsqueeze(1)
    std = std.expand(batch_x.size(0), 1, 3000)
    noise = torch.normal(mean=0.0, std=std).to(batch_x.device)
    processed = process_noise(noise)
    return batch_x * (1 + processed)

# ─── FFT関数 ─────
def compute_power_spectrum(signal, fs):
    N = signal.shape[-1]
    freq = np.fft.rfftfreq(N, d=1/fs)
    fft_vals = np.fft.rfft(signal, axis=-1)
    power = np.abs(fft_vals)**2 / N
    return freq, power

# ─── デバイス & データ読み込み ───────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用デバイス:", device)

with open('/content/drive/MyDrive/noise/data_lowF_noise.pickle', 'rb') as f:
    data = pickle.load(f)
X = data['x'].float().to(device)
Y = data['y'].float().to(device)
print("データ形状 X:", X.shape, " Y:", Y.shape)

# ─── モデルと訓練設定 ───────────────
model = CNN1d_with_resnet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ─── 学習ループ ────────────────────
epochs = 2  # 好きな回数に
batch_size = 32
num_samples = X.size(0)

for epoch in range(1, epochs + 1):
    perm = torch.randperm(num_samples)
    running_loss = 0.0

    for i in range(0, num_samples, batch_size):
        idx = perm[i:i+batch_size]
        batch_x = X[idx]
        batch_y = Y[idx]

        noisy_batch_x = add_structured_noise(batch_x)

        optimizer.zero_grad()
        outputs = model(noisy_batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)

    epoch_loss = running_loss / num_samples
    print(f"Epoch {epoch}/{epochs}  平均損失: {epoch_loss:.6f}")

# ─── 推論＆FFT用データ準備 ──────────────
model.eval()
with torch.no_grad():
    x_sample = X[:1]
    y_sample = Y[:1]
    ea_pred = model(x_sample)
    eb_pred = model(add_structured_noise(x_sample))

ea_np = ea_pred.squeeze().cpu().numpy()
eb_np = eb_pred.squeeze().cpu().numpy()
y_np  = y_sample.squeeze().cpu().numpy()

# 長さを揃える（3000点にパディング）
target_len = 3000
ea_np = np.pad(ea_np, (0, target_len - len(ea_np)), mode='constant')
eb_np = np.pad(eb_np, (0, target_len - len(eb_np)), mode='constant')
y_np  = np.pad(y_np,  (0, target_len - len(y_np)),  mode='constant')

fs = 30000  # サンプリング周波数（仮）

# FFT実行
freq, y_power_ea  = compute_power_spectrum(y_np, fs)
_, ea_power       = compute_power_spectrum(ea_np, fs)
_, y_power_eb     = compute_power_spectrum(y_np, fs)
_, eb_power       = compute_power_spectrum(eb_np, fs)

# ─── グラフ①：正解 vs ea ─────
plt.figure(figsize=(9,6))
plt.scatter(freq, y_power_ea, label='Y (answer)', color='red', marker='o', s=1, alpha=0.5)
plt.scatter(freq, ea_power,   label='ea (prediction)', color='blue', marker='o', s=1, alpha=0.5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (a.u.)')
plt.title('Power Spectrum: Y vs ea')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
ea_path = os.path.join(output_dir, 'power_spectrum_ea_vs_Y.png')
plt.savefig(ea_path)
plt.show()
print(f"✅ グラフ 'ea vs Y' を保存しました: {ea_path}")

# ─── グラフ②：正解 vs eb ─────
plt.figure(figsize=(9,6))
plt.scatter(freq, y_power_eb, label='Y (answer)', color='red', marker='o', s=1, alpha=0.5)
plt.scatter(freq, eb_power,   label='eb (prediction)', color='blue', marker='o', s=1, alpha=0.5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (a.u.)')
plt.title('Power Spectrum: Y vs eb')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
eb_path = os.path.join(output_dir, 'power_spectrum_eb_vs_Y.png')
plt.savefig(eb_path)
plt.show()
print(f"✅ グラフ 'eb vs Y' を保存しました: {eb_path}")

# ─── モデル保存（日時付きフォルダ） ─────
model_path = os.path.join(output_dir, 'model.pth')
torch.save(model.state_dict(), model_path)
print(f"✅ モデルを Google Drive に保存しました: {model_path}")