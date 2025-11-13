#!/usr/bin/env python3
# train_mlp_surrogate.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# -----------------------------
# 1. 설정
# -----------------------------
CSV_PATH = "beam_dataset.csv"
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 300
VAL_RATIO = 0.2
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------
# 2. 데이터 로드 및 train/val 분할
# -----------------------------
df = pd.read_csv(CSV_PATH)

# 입력(feature): E, I, L, P
# 출력(target): y_max
feature_cols = ["E", "I", "L", "P"]
target_col = "y_max"

X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)

# 간단한 정규화(로그 스케일을 써도 좋지만 여기서는 min-max 예시)
# 실제 프로젝트에서는 단위/스케일을 고려해 적절한 스케일링 선택 필요
X_min = X.min(axis=0, keepdims=True)
X_max = X.max(axis=0, keepdims=True)
X_norm = (X - X_min) / (X_max - X_min + 1e-12)

y_min = y.min(axis=0, keepdims=True)
y_max = y.max(axis=0, keepdims=True)
y_norm = (y - y_min) / (y_max - y_min + 1e-12)

n_samples = X.shape[0]
indices = np.arange(n_samples)
np.random.shuffle(indices)

n_val = int(n_samples * VAL_RATIO)
val_idx = indices[:n_val]
train_idx = indices[n_val:]

X_train, y_train = X_norm[train_idx], y_norm[train_idx]
X_val, y_val = X_norm[val_idx], y_norm[val_idx]

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
)
val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32),
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# 3. MLP 모델 정의
# -----------------------------
class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(64, 64), out_dim: int = 1):
        super().__init__()
        layers = []
        last_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = MLPRegressor(in_dim=len(feature_cols), hidden_dims=(64, 64), out_dim=1).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# 4. 학습 루프
# -----------------------------
def evaluate(loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

train_losses = []
val_losses = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

    train_loss = evaluate(train_loader)
    val_loss = evaluate(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if epoch % 50 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d} | train MSE: {train_loss:.4e} | val MSE: {val_loss:.4e}")

# -----------------------------
# 5. 학습 곡선 및 예측 vs 실측 그리기 (Day 6 내용)
# -----------------------------
# (1) 학습/검증 loss 곡선
plt.figure()
plt.plot(train_losses, label="train MSE")
plt.plot(val_losses, label="val MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=200)
plt.close()
print("Saved loss_curve.png")

# (2) 예측 vs 실측 (val set 기준)
model.eval()
with torch.no_grad():
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_pred_norm = model(X_val_t).cpu().numpy()

# 역정규화
y_val_true = y_val * (y_max - y_min + 1e-12) + y_min
y_val_pred = y_val_pred_norm * (y_max - y_min + 1e-12) + y_min

# 스칼라이므로 flatten
y_val_true = y_val_true.flatten()
y_val_pred = y_val_pred.flatten()

# MSE, MAE (engineering metric 관점에서 baseline 확인)
mse = np.mean((y_val_true - y_val_pred) ** 2)
mae = np.mean(np.abs(y_val_true - y_val_pred))
print(f"Validation MSE (physical): {mse:.4e}")
print(f"Validation MAE (physical): {mae:.4e}")

# 산점도 (예측 vs 실측)
plt.figure()
plt.scatter(y_val_true, y_val_pred, s=15)
min_val = min(y_val_true.min(), y_val_pred.min())
max_val = max(y_val_true.max(), y_val_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "k--")  # y=x 선
plt.xlabel("True y_max")
plt.ylabel("Predicted y_max")
plt.title("Validation: True vs Predicted deflection")
plt.grid(True)
plt.tight_layout()
plt.savefig("pred_vs_true_scatter.png", dpi=200)
plt.close()
print("Saved pred_vs_true_scatter.png")

# (선택) index 기준으로 라인 플롯도 가능
plt.figure()
idx = np.argsort(y_val_true)
plt.plot(y_val_true[idx], label="true")
plt.plot(y_val_pred[idx], label="pred", linestyle="--")
plt.xlabel("Sorted sample index")
plt.ylabel("Deflection y_max")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pred_vs_true_line.png", dpi=200)
plt.close()
print("Saved pred_vs_true_line.png")

# 모델 저장 (추후 재사용 가능)
torch.save(model.state_dict(), "mlp_surrogate.pth")
print("Saved model to mlp_surrogate.pth")
