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
from datetime import datetime, timedelta


# noise フォルダから model と loss を import
from model import CNN1d_with_resnet
from loss_function import WeightedMSELoss

# ─── 保存先ディレクトリ作成（タイムスタンプ付き） ─────
custom_tag = "attention_model2（SEBlock） 100エポック-2回目"
jst_now = datetime.utcnow() + timedelta(hours=9)
timestamp = jst_now.strftime('%m%d_%H:%M')
folder_name = f"{timestamp}_{custom_tag}"

drive_output_dir = f'/content/drive/MyDrive/noise/model_logs/{folder_name}'
local_output_dir = f'model_logs/{folder_name}'
os.makedirs(drive_output_dir, exist_ok=True)
os.makedirs(local_output_dir, exist_ok=True)

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
print("\u4f7f\u7528\u30c7\u30d0\u30a4\u30b9:", device)

with open('/content/drive/MyDrive/noise/data_lowF_noise.pickle', 'rb') as f:
    data = pickle.load(f)
X = data['x'].float().to(device)
Y = data['y'].float().to(device)
print("\u30c7\u30fc\u30bf\u5f62\u72b6 X:", X.shape, " Y:", Y.shape)

# ─── モデルと訓練設定 ───────────────
model = CNN1d_with_resnet().to(device)
print(f"\n⚠️ 現在のモデル: {model.model_name}\n")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ─── 学習ループ ────────────────────
epochs = 100
batch_size = 32
num_samples = X.size(0)
loss_log = []

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
    loss_log.append(epoch_loss)
    print(f"Epoch {epoch}/{epochs}  \u5e73\u5747\u640d\u5931: {epoch_loss:.6f}")

# ─── 損失ロググラフ ─────
np.save(os.path.join(drive_output_dir, 'loss_log.npy'), np.array(loss_log))
np.save(os.path.join(local_output_dir, 'loss_log.npy'), np.array(loss_log))

plt.figure()
plt.plot(range(1, epochs+1), loss_log, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(drive_output_dir, 'loss_curve.png'))
plt.savefig(os.path.join(local_output_dir, 'loss_curve.png'))
plt.show()

# ─── 推論とFFT ─────
model.eval()
with torch.no_grad():
    x_sample = X[:1]
    y_sample = Y[:1]
    ea_pred = model(x_sample)
    eb_pred = model(add_structured_noise(x_sample))

ea_np = ea_pred.squeeze().cpu().numpy()
eb_np = eb_pred.squeeze().cpu().numpy()
y_np  = y_sample.squeeze().cpu().numpy()

target_len = 3000
ea_np = np.pad(ea_np, (0, target_len - len(ea_np)), mode='constant')
eb_np = np.pad(eb_np, (0, target_len - len(eb_np)), mode='constant')
y_np  = np.pad(y_np,  (0, target_len - len(y_np)),  mode='constant')

fs = 30000
freq, y_power = compute_power_spectrum(y_np, fs)
_, ea_power = compute_power_spectrum(ea_np, fs)
_, eb_power = compute_power_spectrum(eb_np, fs)

# ─── グラフ描画関数 ─────
def save_plot(freq, y, pred, title, name):
    plt.figure()
    plt.scatter(freq, y, s=1, alpha=0.5, label='Y (answer)', color='red')
    plt.scatter(freq, pred, s=1, alpha=0.5, label=name + ' (prediction)', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (a.u.)')
    plt.title(f'Scatter Power Spectrum: Y vs {name}')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.savefig(os.path.join(drive_output_dir, f'scatter_power_{name}.png'))
    plt.savefig(os.path.join(local_output_dir, f'scatter_power_{name}.png'))
    plt.show()

    plt.figure()
    plt.plot(freq, y, label='Y (answer)', color='red')
    plt.plot(freq, pred, label=f'{name} (prediction)', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (a.u.)')
    plt.title(f'Plot Power Spectrum: Y vs {name}')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.savefig(os.path.join(drive_output_dir, f'plot_power_{name}.png'))
    plt.savefig(os.path.join(local_output_dir, f'plot_power_{name}.png'))
    plt.show()

# ─── グラフ出力 ─────
save_plot(freq, y_power, ea_power, 'ea', 'ea')
save_plot(freq, y_power, eb_power, 'eb', 'eb')

# ─── モデル保存 ─────
torch.save(model.state_dict(), os.path.join(drive_output_dir, 'model.pth'))
torch.save(model.state_dict(), os.path.join(local_output_dir, 'model.pth'))
print("\n✅ モデルとグラフをすべて保存しました（Drive & ローカル）")

# ─── キャッシュ（__pycache__）削除 ─────
import shutil
import glob

for pycache in glob.glob('/content/drive/MyDrive/noise/**/__pycache__', recursive=True):
    shutil.rmtree(pycache, ignore_errors=True)

print("\n🧹 __pycache__ フォルダをすべて削除しました（Drive 内）")