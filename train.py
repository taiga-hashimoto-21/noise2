# ─── ライブラリ ───────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from model import CNN1d_with_resnet

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

    std = torch.sqrt(var).unsqueeze(0).unsqueeze(1)  # → shape: (1, 1, 3000)
    std = std.expand(batch_x.size(0), 1, 3000)        # バッチサイズ分 broadcast

    noise = torch.normal(mean=0.0, std=std).to(batch_x.device)
    processed = process_noise(noise)
    return batch_x * (1 + processed)

# ─── デバイス & データ読み込み ───────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用デバイス:", device)

with open('data_lowF_noise.pickle', 'rb') as f:
    data = pickle.load(f)
X = data['x'].float().to(device)
Y = data['y'].float().to(device)
print("データ形状 X:", X.shape, " Y:", Y.shape)

# ─── モデルと訓練設定 ───────────────
model = CNN1d_with_resnet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ─── 学習ループ ────────────────────
epochs = 100
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

# ─── モデル保存 ───────────────────────
torch.save(model.state_dict(), 'model.pth')
print("✅ 学習が完了し、'model.pth' を保存しました。")

# ─── 推論テスト ───────────────────────
model2 = CNN1d_with_resnet().to(device)
model2.load_state_dict(torch.load('model.pth', map_location=device))
model2.eval()
with torch.no_grad():
    dummy = torch.randn(3, 1, X.size(2)).to(device)
    out = model2(dummy)
    print("推論テスト出力サイズ:", out.shape)
    print("推論結果例:", out[:2])
