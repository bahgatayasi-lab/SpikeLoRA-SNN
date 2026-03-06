# -*- coding: utf-8 -*-
"""spikelora_ts-Githup.ipynb



**Cell 0 — Imports + config**
"""

# install pytorch (pick the one matching your CUDA; here is CPU example)
!pip install torch torchvision torchaudio

# spikingjelly (activation_based)
!pip install spikingjelly

!pip install numpy pandas scikit-learn tqdm

import os, time, math, random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import spikingjelly

from spikingjelly.activation_based import neuron, surrogate, functional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

import torch
import spikingjelly
print(torch.__version__)

"""**Notebook: E1 End-to-End (ANN + PatchTST + SNN with SpikingJelly LIF)**

**Cell 1 — Metrics (RMSE/MAE/SMAPE)**
"""

def rmse_np(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae_np(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def smape_np(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(200.0 * np.mean(np.abs(y_pred - y_true) / denom))

@torch.no_grad()
def eval_metrics(y_true_t, y_pred_t):
    y_true = y_true_t.detach().cpu().numpy()
    y_pred = y_pred_t.detach().cpu().numpy()
    return {
        "rmse": rmse_np(y_true, y_pred),
        "mae": mae_np(y_true, y_pred),
        "smape": smape_np(y_true, y_pred)
    }

"""Cell 2 — Dataset paths + loader

Update only the time/target columns if needed.
I’ll make a robust loader that guesses time + target if you didn’t set them.
"""

import pandas as pd

DATASETS = {
    "SR": {
        "path": "Palestine-Solar.csv",
        "time_col": None,   # auto
        "target_col": None, # auto
    },
    "WS": {
        "path": "Palestine-wind.csv",
        "time_col": None,
        "target_col": None,
    },
    "WP": {
        "path": "Turky-Wind-power-Turbine.csv",
        "time_col": None,
        "target_col": None,
    },
    "EC": {
        "path": "Moroco-power-consumption.csv",
        "time_col": None,
        "target_col": None,
    },
}

def _guess_time_col(df: pd.DataFrame) -> str:
    # common names
    for c in ["date", "Date", "datetime", "Datetime", "timestamp", "Timestamp", "time", "Time"]:
        if c in df.columns:
            return c
    # fallback: first column
    return df.columns[0]

def _guess_target_col(df: pd.DataFrame) -> str:
    # prefer known patterns
    preferred = {"target", "y", "power", "ghi", "load", "consumption", "windspeed", "wind_speed"}
    cand = [c for c in df.columns if c.lower() in preferred]
    if cand:
        return cand[0]

    # otherwise pick last numeric column
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("No numeric columns found for target.")
    return num_cols[-1]

def load_df(task_id: str):
    cfg = DATASETS[task_id]
    path = cfg["path"]
    df = pd.read_csv(path)

    time_col = cfg["time_col"] or _guess_time_col(df)
    target_col = cfg["target_col"] or _guess_target_col(df)

    # parse datetime if possible
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    # keep numeric features only (except time)
    feat_cols = [c for c in df.columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])]
    if target_col not in feat_cols:
        raise ValueError(f"Target col '{target_col}' is not numeric or not found among numeric features.")

    df = df[[time_col] + feat_cols].copy()

    # --- requested prints ---
    print(f"\n=== Dataset {task_id} ===")
    print(f"Path: {path}")
    print(f"Time column: {time_col}")
    print(f"Target variable: {target_col}")
    #print("First 3 rows:")
    #print(df.head(3))
    # ------------------------

    return df, {"time_col": time_col, "target_col": target_col, "feat_cols": feat_cols}

if __name__ == "__main__":
    # Load all datasets and print target + head(3) for each
    loaded = {}
    meta = {}

    for task_id in DATASETS.keys():
        df, info = load_df(task_id)
        loaded[task_id] = df
        meta[task_id] = info

    # Optional: quick summary at the end
    print("\n=== Summary ===")
    for task_id, info in meta.items():
        print(f"{task_id}: time_col={info['time_col']}, target_col={info['target_col']}, n_features={len(info['feat_cols'])}")

"""**Cell 3 — Windowing + splits + standardization**"""

def make_supervised(df, time_col, target_col, lookback, horizons, add_calendar=True):
    feat_cols = [c for c in df.columns if c != time_col]
    X_raw = df[feat_cols].values.astype(np.float32)

    # optional calendar features
    if add_calendar:
        t = df[time_col]
        cal = np.stack([
            t.dt.hour.fillna(0).values,
            t.dt.dayofweek.fillna(0).values,
            t.dt.month.fillna(0).values,
        ], axis=1).astype(np.float32)
        cal[:,0] /= 23.0
        cal[:,1] /= 6.0
        cal[:,2] /= 12.0
        X_raw = np.concatenate([X_raw, cal], axis=1)

    # target is the specific column (before calendar features!)
    target_idx = feat_cols.index(target_col)

    H = len(horizons)
    L = lookback
    N = len(X_raw)

    X = []
    Y = []
    max_h = max(horizons)
    for i in range(L, N - max_h):
        xw = X_raw[i-L:i]  # (L, F)
        y = []
        for h in horizons:
            y.append(df[target_col].values[i + h].astype(np.float32))
        X.append(xw)
        Y.append(y)

    X = np.stack(X, axis=0)  # (Nw, L, F)
    Y = np.stack(Y, axis=0)  # (Nw, H)
    return X, Y

def chronological_split(X, Y, train_ratio=0.7, val_ratio=0.1):
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    Xtr, Ytr = X[:n_train], Y[:n_train]
    Xva, Yva = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    Xte, Yte = X[n_train+n_val:], Y[n_train+n_val:]
    return (Xtr,Ytr),(Xva,Yva),(Xte,Yte)

def standardize_fit_transform(Xtr, Xva, Xte):
    mu = Xtr.mean(axis=(0,1), keepdims=True)
    sd = Xtr.std(axis=(0,1), keepdims=True) + 1e-6
    return (Xtr-mu)/sd, (Xva-mu)/sd, (Xte-mu)/sd, (mu,sd)

class WindowDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

"""**ANN baselines**"""

class MLPBaseline(nn.Module):
    def __init__(self, lookback, F, K, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lookback * F, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, K),
        )

    def forward(self, x):
        return self.net(x)


class RNNBaseline(nn.Module):
    def __init__(self, kind, F, K, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        RNN = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = RNN(
            input_size=F, hidden_size=hidden,
            num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.head = nn.Linear(hidden, K)

    def forward(self, x):
        out, _ = self.rnn(x)           # (B,L,H)
        return self.head(out[:, -1])   # last step


class TCNBaseline(nn.Module):
    def __init__(self, F, K, channels=128, levels=3, k=3, dropout=0.1):
        super().__init__()
        layers = []
        in_ch = F
        for i in range(levels):
            dilation = 2 ** i
            pad = (k - 1) * dilation
            layers += [
                nn.Conv1d(in_ch, channels, kernel_size=k, dilation=dilation, padding=pad),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = channels
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(channels, K)

    def forward(self, x):
        # x: (B,L,F) -> (B,F,L)
        z = x.transpose(1, 2)
        z = self.net(z)
        z = z[:, :, -1]      # last time step
        return self.head(z)


class PatchTST(nn.Module):
    """
    Simple PatchTST-style baseline: patchify -> linear embed -> Transformer encoder -> head
    (Not the full official PatchTST, but a strong TSFormer-style baseline.)
    """
    def __init__(self, F, K, patch_len=16, stride=8, d_model=128, nhead=4, layers=3, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

        self.proj = nn.Linear(patch_len * F, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.tf = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, K))

    def patchify(self, x):
        # x: (B,L,F) -> (B,Np,patch_len,F)
        B, L, F = x.shape
        patches = []
        for s in range(0, L - self.patch_len + 1, self.stride):
            patches.append(x[:, s:s+self.patch_len, :])
        p = torch.stack(patches, dim=1)
        return p

    def forward(self, x):
        p = self.patchify(x)
        B, Np, Pl, F = p.shape
        tok = p.reshape(B, Np, Pl*F)
        z = self.proj(tok)
        h = self.tf(z)
        return self.head(h[:, -1])

"""**Cell 4 — Training utilities (fixed tensors + RMSE/MAE/SMAPE)**"""

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_params(model):
    return sum(p.numel() for p in model.parameters())

def _to_device(x):
    if torch.is_tensor(x):
        return x.to(device)
    return torch.from_numpy(x).to(device)

def train_one_epoch(model, loader, optim):
    model.train()
    crit = nn.MSELoss()
    total = 0.0
    for xb, yb in loader:
        xb = _to_device(xb)
        yb = _to_device(yb)

        optim.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = crit(pred, yb)
        loss.backward()
        optim.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = _to_device(xb)
        yb = _to_device(yb)
        pred = model(xb)
        preds.append(pred)
        trues.append(yb)
    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)
    m = eval_metrics(trues, preds)
    return m["rmse"], m["mae"], m["smape"]

def fit(model, train_loader, val_loader, lr=1e-3, wd=1e-4, max_epochs=30, patience=10):
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)

    best_rmse = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0
    t0 = time.time()

    for ep in range(1, max_epochs+1):
        _ = train_one_epoch(model, train_loader, optim)
        val_rmse, val_mae, val_smape = eval_model(model, val_loader)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_epoch": best_epoch,
        "best_val_rmse": best_rmse,
        "train_time_sec": time.time() - t0,
        "trainable_params": count_trainable_params(model),
        "total_params": count_total_params(model),
    }

"""**Cell 5 — LoRA + SpikeLoRA modules (complete)**"""

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

        in_dim = base.in_features
        out_dim = base.out_features

        self.A = nn.Parameter(torch.zeros(r, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        base_out = self.base(x)
        a = self.drop(x) @ self.A.t()
        delta = a @ self.B.t()
        return base_out + self.scale * delta


class SpikeLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, Vtheta=0.1, dropout=0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

        in_dim = base.in_features
        out_dim = base.out_features

        self.A = nn.Parameter(torch.zeros(r, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        for p in self.base.parameters():
            p.requires_grad = False

        self.gate = neuron.LIFNode(
            v_threshold=Vtheta,
            surrogate_function=surrogate.Sigmoid(),
            detach_reset=True
        )
        self.last_sparsity_pct = None

    def forward(self, x):
        functional.reset_net(self.gate)
        base_out = self.base(x)

        a = self.drop(x) @ self.A.t()
        g = self.gate(a)
        a_tilde = g * a

        with torch.no_grad():
            self.last_sparsity_pct = float((g == 0).float().mean().item() * 100.0)

        delta = a_tilde @ self.B.t()
        return base_out + self.scale * delta

"""Cell 6 — Two-stage PEFT wrapper for your working SpikingTCN

This is the key fix.

snn_tcn = full training

snn_tcn_lora / snn_tcn_spikelora = Stage A pretrain full base model, then Stage B freeze and adapt
"""

def freeze_module(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False

def unfreeze_module(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = True


class SpikingTCN_Adapters(nn.Module):
    """
    Takes a PRETRAINED SpikingTCN base, then replaces head with LoRA/SpikeLoRA,
    freezes backbone, and trains adapters (and optionally LayerNorms).
    """
    def __init__(self, base_model: nn.Module, peft="lora", r=8, alpha=16, Vtheta=0.1):
        super().__init__()
        assert hasattr(base_model, "head"), "Your SpikingTCN must have .head"
        self.base = base_model
        self.peft = peft

        base_head = self.base.head
        if peft == "lora":
            self.base.head = LoRALinear(base_head, r=r, alpha=alpha, dropout=0.0)
        elif peft == "spikelora":
            self.base.head = SpikeLoRALinear(base_head, r=r, alpha=alpha, Vtheta=Vtheta, dropout=0.0)
        else:
            raise ValueError("peft must be 'lora' or 'spikelora'")

        # freeze everything then unfreeze adapters
        freeze_module(self.base)
        unfreeze_module(self.base.head)

        # (optional) if your TCN has LayerNorms, let them adapt
        for m in self.base.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True

    def forward(self, x):
        return self.base(x)

    def get_spikelora_sparsity(self):
        if hasattr(self.base.head, "last_sparsity_pct"):
            return self.base.head.last_sparsity_pct
        return None

"""**SNN_TCN**"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import neuron, surrogate, functional


class Chomp1d(nn.Module):
    """Remove extra right-padding to keep causal length."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Standard TCN residual block:
    Conv1d -> ReLU -> Dropout -> Conv1d -> ReLU -> Dropout + residual
    Uses causal padding + chomp to preserve sequence length.
    """
    def __init__(self, c_in, c_out, kernel=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel - 1) * dilation

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(c_in, c_out, kernel_size=kernel, stride=1, padding=pad, dilation=dilation)
        )
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(c_out, c_out, kernel_size=kernel, stride=1, padding=pad, dilation=dilation)
        )
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(c_in, c_out, kernel_size=1) if c_in != c_out else None
        self.relu = nn.ReLU()

        # init
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class SpikeEncoder(nn.Module):
    """
    Delta encoder:
    spike at time t if |x_t - x_{t-1}| > thr
    Returns (B,L,F) in {0,1}
    """
    def __init__(self, thr=0.05):
        super().__init__()
        self.thr = thr

    def forward(self, x):
        dx = x[:, 1:, :] - x[:, :-1, :]
        sp = (dx.abs() > self.thr).float()
        z0 = torch.zeros_like(x[:, :1, :])
        return torch.cat([z0, sp], dim=1)


class SpikingTCN(nn.Module):
    """
    Your original (better) SpikingTCN.
    Key points:
    - projects features F -> C before spiking (in_proj)
    - uses LIFNode on projected features
    - uses TemporalBlock stack
    - uses functional.reset_net(self)
    """
    def __init__(self, F, K, channels=(64, 64, 64), T=8, thr=1.0):
        super().__init__()
        self.T = T
        self.encoder = SpikeEncoder(thr=0.05)
        self.in_proj = nn.Linear(F, channels[0])

        self.lif = neuron.LIFNode(
            v_threshold=thr,
            surrogate_function=surrogate.Sigmoid(),
            detach_reset=True
        )

        blocks = []
        c_in = channels[0]
        for i, c_out in enumerate(channels):
            blocks.append(TemporalBlock(c_in, c_out, kernel=3, dilation=2**i, dropout=0.1))
            c_in = c_out
        self.tcn = nn.Sequential(*blocks)

        self.head = nn.Linear(c_in, K)

    def forward(self, x):
        functional.reset_net(self)  # IMPORTANT in spikingjelly
        spk = self.encoder(x)       # (B,L,F)

        acc = 0.0
        for _ in range(self.T):
            z = self.in_proj(spk)                   # (B,L,C)
            s = self.lif(z)                         # spikes (B,L,C)
            h = self.tcn(s.transpose(1, 2))[:, :, -1]  # (B,C)
            acc = acc + self.head(h)                # (B,K)
        return acc / self.T

# quick compile check
B, L, Fdim, K = 8, 96, 10, 5
x = torch.randn(B, L, Fdim)
m = SpikingTCN(Fdim, K).to(device)
y = m(x.to(device))
print("Output shape:", y.shape)  # should be (B, K)

"""Cell 7 — run_one() with correct two-stage logic

This is the part you’re missing.

If model is snn_tcn_lora or snn_tcn_spikelora:

Train base SpikingTCN fully for pretrain_epochs

Wrap it with adapters + freeze backbone

Adapt for adapt_epochs
"""

# put this helper near top of your script
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def run_one(
    task_id, model_name, lookback, horizons,
    seed=0, batch_size=128, max_epochs=30,
    spkT=4, r=8, alpha=16, Vtheta=0.1,
    pretrain_epochs=20, adapt_epochs=20,
    lr_full=1e-3, lr_adapt=5e-3
):
    set_seed(seed)

    df, cfg = load_df(task_id)
    X, Y = make_supervised(
        df, cfg["time_col"], cfg["target_col"],
        lookback, horizons, add_calendar=True
    )

    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = chronological_split(X, Y, train_ratio=0.7, val_ratio=0.1)
    Xtr, Xva, Xte, _ = standardize_fit_transform(Xtr, Xva, Xte)

    train_loader = DataLoader(WindowDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(WindowDataset(Xva, Yva), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(WindowDataset(Xte, Yte), batch_size=batch_size, shuffle=False)

    F = Xtr.shape[-1]
    K = Ytr.shape[-1]

    sp_spars = ""

    stats = {}  # ensure stats exists no matter what branch

    # ---- build + train ----
    # ANN baselines
    if model_name == "mlp":
        model = MLPBaseline(lookback, F, K).to(device)
        model, stats = fit(model, train_loader, val_loader, max_epochs=max_epochs, lr=lr_full)

    elif model_name == "lstm":
        model = RNNBaseline("lstm", F, K).to(device)
        model, stats = fit(model, train_loader, val_loader, max_epochs=max_epochs, lr=lr_full)

    elif model_name == "gru":
        model = RNNBaseline("gru", F, K).to(device)
        model, stats = fit(model, train_loader, val_loader, max_epochs=max_epochs, lr=lr_full)

    elif model_name == "tcn":
        model = TCNBaseline(F, K).to(device)
        model, stats = fit(model, train_loader, val_loader, max_epochs=max_epochs, lr=lr_full)

    elif model_name == "patchtst":
        model = PatchTST(F, K).to(device)
        model, stats = fit(model, train_loader, val_loader, max_epochs=max_epochs, lr=lr_full)

    # SNN baseline
    elif model_name == "snn_tcn":
        model = SpikingTCN(F, K, T=spkT).to(device)
        model, stats = fit(model, train_loader, val_loader, max_epochs=max_epochs, lr=lr_full)

    # SNN + PEFT (two-stage: pretrain full -> adapt adapters)
    elif model_name in ["snn_tcn_lora", "snn_tcn_spikelora"]:
        # Stage A: full pretrain base model
        base = SpikingTCN(F, K, T=spkT).to(device)
        base, stats_pre = fit(base, train_loader, val_loader, max_epochs=pretrain_epochs, lr=lr_full)

        # Stage B: wrap with adapters + adapt
        peft_kind = "lora" if model_name == "snn_tcn_lora" else "spikelora"
        model = SpikingTCN_Adapters(base, peft=peft_kind, r=r, alpha=alpha, Vtheta=Vtheta).to(device)
        model, stats = fit(model, train_loader, val_loader, max_epochs=adapt_epochs, lr=lr_adapt)

        # total time = pretrain + adapt
        # ensure stats_pre keys exist and stats exists
        if "train_time_sec" in stats_pre:
            stats["train_time_sec"] = stats.get("train_time_sec", 0) + stats_pre.get("train_time_sec", 0)

        # optional: record pretrain/adapt times separately if you want
        stats["pretrain_time_sec"] = stats_pre.get("train_time_sec", None)

        if hasattr(model, "get_spikelora_sparsity"):
            sp = model.get_spikelora_sparsity()
            sp_spars = "" if sp is None else sp

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # --- compute parameter counts on the *final* model (after adapters/wrapping) ---
    total_p, trainable_p = count_params(model)
    stats["total_params"] = total_p
    stats["trainable_params"] = trainable_p
    stats["trainable_ratio"] = trainable_p / max(1, total_p)

    # evaluate
    test_rmse, test_mae, test_smape = eval_model(model, test_loader)

    return {
        "task": task_id,
        "model": model_name,
        "seed": seed,
        "lookback": lookback,
        "horizons": str(horizons),
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_smape": test_smape,
        "train_time_sec": stats.get("train_time_sec", None),
        "trainable_params": stats["trainable_params"],
        "total_params": stats["total_params"],
        "trainable_ratio": stats["trainable_ratio"],
        "spikelora_sparsity_pct": sp_spars,
        # keep any other useful stats if desired
    }

"""Immediate step 1: Run E1 on all four datasets with three seeds"""

tasks = ["SR","WS","WP","EC"]
models = ["mlp","lstm","gru","tcn","patchtst","snn_tcn","snn_tcn_lora","snn_tcn_spikelora"]
seeds = [0]
#seeds = [0,1,2]
lookback = 96
horizons = [1,2,4,8,24]

results = []
for task in tasks:
    for model in models:
        for seed in seeds:
            r = run_one(task, model, lookback, horizons, seed=seed, max_epochs=30, spkT=4, r=8, Vtheta=0.1)
            results.append(r)
            print(task, model, seed, "RMSE", round(r["test_rmse"],3), "MAE", round(r["test_mae"],3), "SMAPE", round(r["test_smape"],3))

import pandas as pd
E1_df = pd.DataFrame(results)
E1_df.to_csv("E1_results.csv", index=False)
E1_df.head()

import pandas as pd

with pd.option_context(
    'display.max_rows', None,
    'display.max_columns', None,
    'display.width', None,
    'display.max_colwidth', None
):
    print(E1_df)

summary = (E1_df
           .groupby(["task","model"])
           .agg(test_rmse_mean=("test_rmse","mean"),
                test_rmse_std=("test_rmse","std"),
                test_mae_mean=("test_mae","mean"),
                test_mae_std=("test_mae","std"),
                test_smape_mean=("test_smape","mean"),
                test_smape_std=("test_smape","std"),
                trainable_ratio_mean=("trainable_ratio","mean"))
           .reset_index())
summary

"""**E2--Cell E2-1 —tools to prepare  few-shot loaders (chronological)**"""

def make_fewshot_loaders(Xtr, Ytr, Xva, Yva, Xte, Yte, frac, batch_size=128):
    n = len(Xtr)
    n_fs = max(1, int(n * frac))

    Xtr_fs = Xtr[:n_fs]
    Ytr_fs = Ytr[:n_fs]

    train_loader = DataLoader(WindowDataset(Xtr_fs, Ytr_fs), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(WindowDataset(Xva, Yva), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(WindowDataset(Xte, Yte), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, n_fs

"""**Cell E2-1 — pretrain base SNN-TCN once for each task/seed**"""

def pretrain_base(task_id, lookback, horizons, seed=0, spkT=8, batch_size=128, max_epochs=30, lr=1e-3):
    set_seed(seed)

    df, cfg = load_df(task_id)
    X, Y = make_supervised(df, cfg["time_col"], cfg["target_col"], lookback, horizons, add_calendar=True)
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = chronological_split(X, Y, train_ratio=0.7, val_ratio=0.1)
    Xtr, Xva, Xte, _ = standardize_fit_transform(Xtr, Xva, Xte)

    train_loader = DataLoader(WindowDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(WindowDataset(Xva, Yva), batch_size=batch_size, shuffle=False)

    F = Xtr.shape[-1]
    K = Ytr.shape[-1]

    base = SpikingTCN(F, K, T=spkT).to(device)
    base, stats_pre = fit(base, train_loader, val_loader, max_epochs=max_epochs, lr=lr)

    base_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}

    splits = (Xtr, Ytr, Xva, Yva, Xte, Yte)
    dims = (F, K)
    return base_state, splits, dims, stats_pre

"""**Cell E2-2 — Adaptation modes: FullFT vs LoRA vs SpikeLoRA**"""

def adapt_and_eval(base_state, F, K, spkT, mode, loaders,
                   r=8, alpha=16, Vtheta=0.1,
                   adapt_epochs=20, lr_adapt=1e-3):
    train_loader, val_loader, test_loader = loaders

    # load base
    base = SpikingTCN(F, K, T=spkT).to(device)
    base.load_state_dict(base_state, strict=True)

    sp_spars = ""

    if mode == "fullft":
        # fine-tune all params from pretrained init
        for p in base.parameters():
            p.requires_grad = True
        model = base

    elif mode in ["lora", "spikelora"]:
        model = SpikingTCN_Adapters(base, peft=mode, r=r, alpha=alpha, Vtheta=Vtheta).to(device)

    else:
        raise ValueError("mode must be: fullft, lora, spikelora")

    model, stats = fit(model, train_loader, val_loader, max_epochs=adapt_epochs, lr=lr_adapt)

    test_rmse, test_mae, test_smape = eval_model(model, test_loader)

    if hasattr(model, "get_spikelora_sparsity"):
        sp = model.get_spikelora_sparsity()
        sp_spars = "" if sp is None else sp

    return {
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_smape": test_smape,
        "train_time_sec": stats["train_time_sec"],
        "trainable_params": stats["trainable_params"],
        "total_params": stats["total_params"],
        "trainable_ratio": stats["trainable_params"] / max(1, stats["total_params"]),
        "spikelora_sparsity_pct": sp_spars
    }

"""**(B) Full E2 run (each tasks + 3 seeds)**"""

import pandas as pd

tasks = ["SR","WS","WP","EC"]
seeds = [0]
#seeds = [0,1,2]
fractions = [0.1, 0.2, 0.5]

lookback = 96
horizons = [1,2,4,8,24]

spkT = 8
r = 8
alpha = 16
Vtheta = 0.05   # <-- important (validated)
batch_size = 128

pretrain_epochs = 30
adapt_epochs = 20

lr_full = 1e-3
lr_fullft = 1e-3
lr_peft = 5e-3

E2_results = []

for task in tasks:
    for seed in seeds:
        base_state, splits, (F,K), pre_stats = pretrain_base(
            task, lookback, horizons,
            seed=seed, spkT=spkT, batch_size=batch_size,
            max_epochs=pretrain_epochs, lr=lr_full
        )
        Xtr, Ytr, Xva, Yva, Xte, Yte = splits

        for frac in fractions:
            train_loader, val_loader, test_loader, n_fs = make_fewshot_loaders(
                Xtr, Ytr, Xva, Yva, Xte, Yte, frac, batch_size
            )

            # FullFT
            out = adapt_and_eval(base_state, F, K, spkT, "fullft",
                                 (train_loader,val_loader,test_loader),
                                 adapt_epochs=adapt_epochs, lr_adapt=lr_fullft)
            out.update({"task":task, "seed":seed, "frac":frac, "n_train_windows":n_fs, "mode":"fullft"})
            E2_results.append(out)

            # LoRA
            out = adapt_and_eval(base_state, F, K, spkT, "lora",
                                 (train_loader,val_loader,test_loader),
                                 r=r, alpha=alpha, Vtheta=Vtheta,
                                 adapt_epochs=adapt_epochs, lr_adapt=lr_peft)
            out.update({"task":task, "seed":seed, "frac":frac, "n_train_windows":n_fs, "mode":"lora"})
            E2_results.append(out)

            # SpikeLoRA
            out = adapt_and_eval(base_state, F, K, spkT, "spikelora",
                                 (train_loader,val_loader,test_loader),
                                 r=r, alpha=alpha, Vtheta=Vtheta,
                                 adapt_epochs=adapt_epochs, lr_adapt=lr_peft)
            out.update({"task":task, "seed":seed, "frac":frac, "n_train_windows":n_fs, "mode":"spikelora"})
            E2_results.append(out)

            print(task, seed, frac,
                  "fullft", round(E2_results[-3]["test_rmse"],3),
                  "lora", round(E2_results[-2]["test_rmse"],3),
                  "spike", round(E2_results[-1]["test_rmse"],3),
                  "spars", E2_results[-1]["spikelora_sparsity_pct"])

E2_df = pd.DataFrame(E2_results)
E2_df.to_csv("E2_fewshot_results.csv", index=False)
E2_df.head()

import pandas as pd

with pd.option_context(
    'display.max_rows', None,
    'display.max_columns', None,
    'display.width', None,
    'display.max_colwidth', None
):
    print(E2_df)

"""Number of trainable paramters

**Cell E2-4 — Summary mean±std + plot curves**
"""

E2_summary = (E2_df
              .groupby(["task","mode","frac"])
              .agg(
                  rmse_mean=("test_rmse","mean"),
                  rmse_std=("test_rmse","std"),
                  mae_mean=("test_mae","mean"),
                  mae_std=("test_mae","std"),
                  smape_mean=("test_smape","mean"),
                  smape_std=("test_smape","std"),
                  trainable_ratio_mean=("trainable_ratio","mean"),
                  train_time_mean=("train_time_sec","mean"),
              )
              .reset_index())

E2_summary

"""**Cell E2-5 — plot Curves (RMSE vs fraction) for each Task**"""

import matplotlib.pyplot as plt

def plot_e2(task, metric="rmse_mean"):
    sub = E2_summary[E2_summary["task"] == task].copy()
    plt.figure()
    for mode in ["fullft","lora","spikelora"]:
        m = sub[sub["mode"] == mode].sort_values("frac")
        plt.plot(m["frac"], m[metric], marker="o", label=mode)
    plt.xlabel("Few-shot fraction")
    plt.ylabel(metric)
    plt.title(f"E2 curves ({metric}) - {task}")
    plt.grid(True)
    plt.legend()
    plt.show()

for t in ["SR","WS","WP","EC"]:
    plot_e2(t, "rmse_mean")

"""**E3**"""

def _calendar_features_from_datetime(dt_series):
    # dt_series: pandas datetime
    # returns dataframe of calendar features
    import pandas as pd
    out = pd.DataFrame(index=dt_series.index)
    out["hour"] = dt_series.dt.hour
    out["dayofweek"] = dt_series.dt.dayofweek
    out["month"] = dt_series.dt.month
    out["dayofyear"] = dt_series.dt.dayofyear
    return out

def get_feature_cols(df, cfg, add_calendar=True):
    import numpy as np

    time_col = cfg["time_col"]
    target_col = cfg["target_col"]

    # numeric columns only, excluding time and target
    num_cols = [c for c in df.columns
                if c not in [time_col, target_col] and np.issubdtype(df[c].dtype, np.number)]

    # we will optionally append calendar feats later
    base_cols = num_cols.copy()

    if add_calendar:
        # calendar cols are standardized names
        cal_cols = ["hour","dayofweek","month","dayofyear"]
        return base_cols, cal_cols
    else:
        return base_cols, []

"""✅ Cell E3-F2 — Unionize DF on a column list (Union) + fill missing"""

def align_df_to_feature_space(df, cfg, base_cols_union, add_calendar=True):
    import pandas as pd
    import numpy as np

    time_col = cfg["time_col"]
    target_col = cfg["target_col"]

    out = pd.DataFrame(index=df.index)
    out[time_col] = df[time_col]
    out[target_col] = df[target_col]

    # add base cols union (fill missing with 0)
    for c in base_cols_union:
        if c in df.columns:
            out[c] = df[c]
        else:
            out[c] = 0.0

    # add calendar
    if add_calendar:
        dt = pd.to_datetime(out[time_col], errors="coerce")
        cal = _calendar_features_from_datetime(dt)
        for c in cal.columns:
            out[c] = cal[c].astype(float)

    return out

"""Cell E3-F3 — Supervised windows from DF are “ready”.

This is the same idea as make_supervised, but here we control the feature columns.
"""

def make_supervised_aligned(df, cfg, lookback, horizons, base_cols_union, add_calendar=True):
    import numpy as np

    time_col = cfg["time_col"]
    target_col = cfg["target_col"]

    # align
    df2 = align_df_to_feature_space(df, cfg, base_cols_union, add_calendar=add_calendar)

    # feature columns order = union + calendar (if enabled)
    feat_cols = list(base_cols_union)
    if add_calendar:
        feat_cols += ["hour","dayofweek","month","dayofyear"]

    X_raw = df2[feat_cols].to_numpy(dtype=np.float32)
    y_raw = df2[target_col].to_numpy(dtype=np.float32)

    L = lookback
    H = horizons
    max_h = max(H)

    Xs, Ys = [], []
    for i in range(L, len(df2) - max_h):
        x_win = X_raw[i-L:i, :]                    # (L,F)
        y_vec = np.array([y_raw[i+h] for h in H], dtype=np.float32)  # (K,)
        Xs.append(x_win)
        Ys.append(y_vec)

    X = np.stack(Xs, axis=0)  # (N,L,F)
    Y = np.stack(Ys, axis=0)  # (N,K)
    return X, Y, feat_cols

"""✅ Cell E3-F4 — Prepare splits with feature union (instead of assert)



"""

def prepare_task_splits_E3(task_id, lookback, horizons, base_cols_union, seed=0):
    set_seed(seed)

    df, cfg = load_df(task_id)

    X, Y, feat_cols = make_supervised_aligned(
        df, cfg, lookback, horizons, base_cols_union, add_calendar=True
    )

    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = chronological_split(X, Y, train_ratio=0.7, val_ratio=0.1)

    # standardize (per-task) like E1/E2:
    Xtr, Xva, Xte, scaler = standardize_fit_transform(Xtr, Xva, Xte)

    F = Xtr.shape[-1]
    K = Ytr.shape[-1]
    return (Xtr, Ytr, Xva, Yva, Xte, Yte), (F, K), feat_cols

"""✅ Cell E3-F5 — Pretrain multi-task base (Union features)"""

def build_union_feature_space(tasks, seed=0):
    """
    Builds UNION of numeric feature columns across given tasks (excluding time/target),
    and uses fixed calendar cols.
    """
    base_union = []
    base_union_set = set()

    for t in tasks:
        df, cfg = load_df(t)
        base_cols, cal_cols = get_feature_cols(df, cfg, add_calendar=True)
        for c in base_cols:
            if c not in base_union_set:
                base_union_set.add(c)
                base_union.append(c)

    return base_union

def pretrain_multitask_base_E3(source_tasks, lookback, horizons, base_cols_union,
                               seed=0, spkT=8, batch_size=128, max_epochs=30, lr=1e-3):
    set_seed(seed)

    Xtr_all, Ytr_all = [], []
    Xva_all, Yva_all = [], []

    F = K = None

    for t in source_tasks:
        splits, (Ft, Kt), feat_cols = prepare_task_splits_E3(t, lookback, horizons, base_cols_union, seed=seed)
        Xtr, Ytr, Xva, Yva, Xte, Yte = splits

        if F is None:
            F, K = Ft, Kt
        else:
            assert Ft == F and Kt == K, "Still mismatch (should not happen now)"

        Xtr_all.append(Xtr); Ytr_all.append(Ytr)
        Xva_all.append(Xva); Yva_all.append(Yva)

    Xtr_all = np.concatenate(Xtr_all, axis=0)
    Ytr_all = np.concatenate(Ytr_all, axis=0)
    Xva_all = np.concatenate(Xva_all, axis=0)
    Yva_all = np.concatenate(Yva_all, axis=0)

    train_loader = DataLoader(WindowDataset(Xtr_all, Ytr_all), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(WindowDataset(Xva_all, Yva_all), batch_size=batch_size, shuffle=False)

    base = SpikingTCN(F, K, T=spkT).to(device)
    base, stats_pre = fit(base, train_loader, val_loader, max_epochs=max_epochs, lr=lr)

    base_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    return base_state, (F, K), stats_pre

"""✅ Cell E3-F6 — Run E3 target (SR only) after repair"""

def run_E3_one_target_E3(target_task, source_tasks, lookback, horizons,
                         seed=0, spkT=8, batch_size=128,
                         pretrain_epochs=30, adapt_epochs=20,
                         lr_full=1e-3, lr_fullft=1e-3, lr_peft=5e-3,
                         r=8, alpha=16, Vtheta=0.05,
                         fractions=(0.1, 0.2, 0.5)):

    # union feature space built from SOURCES (foundation pretrain world)
    base_cols_union = build_union_feature_space(source_tasks, seed=seed)

    # 1) pretrain on sources (aligned)
    base_state, (F,K), pre_stats = pretrain_multitask_base_E3(
        source_tasks, lookback, horizons, base_cols_union,
        seed=seed, spkT=spkT, batch_size=batch_size,
        max_epochs=pretrain_epochs, lr=lr_full
    )

    # 2) prepare target splits with SAME union feature space (missing filled)
    (Xtr, Ytr, Xva, Yva, Xte, Yte), (Ft, Kt), feat_cols = prepare_task_splits_E3(
        target_task, lookback, horizons, base_cols_union, seed=seed
    )
    assert Ft == F and Kt == K

    results = []
    for frac in fractions:
        train_loader, val_loader, test_loader, n_fs = make_fewshot_loaders(
            Xtr, Ytr, Xva, Yva, Xte, Yte, frac, batch_size
        )

        out_full = adapt_and_eval(base_state, F, K, spkT, "fullft",
                                  (train_loader, val_loader, test_loader),
                                  adapt_epochs=adapt_epochs, lr_adapt=lr_fullft)
        out_full.update({"E":"E3","target":target_task,"sources":"+".join(source_tasks),
                         "seed":seed,"frac":frac,"n_train_windows":n_fs,"mode":"fullft",
                         "pretrain_time_sec":pre_stats["train_time_sec"]})
        results.append(out_full)

        out_lora = adapt_and_eval(base_state, F, K, spkT, "lora",
                                  (train_loader, val_loader, test_loader),
                                  r=r, alpha=alpha, Vtheta=Vtheta,
                                  adapt_epochs=adapt_epochs, lr_adapt=lr_peft)
        out_lora.update({"E":"E3","target":target_task,"sources":"+".join(source_tasks),
                         "seed":seed,"frac":frac,"n_train_windows":n_fs,"mode":"lora",
                         "pretrain_time_sec":pre_stats["train_time_sec"]})
        results.append(out_lora)

        out_sp = adapt_and_eval(base_state, F, K, spkT, "spikelora",
                                (train_loader, val_loader, test_loader),
                                r=r, alpha=alpha, Vtheta=Vtheta,
                                adapt_epochs=adapt_epochs, lr_adapt=lr_peft)
        out_sp.update({"E":"E3","target":target_task,"sources":"+".join(source_tasks),
                       "seed":seed,"frac":frac,"n_train_windows":n_fs,"mode":"spikelora",
                       "pretrain_time_sec":pre_stats["train_time_sec"]})
        results.append(out_sp)

        print(f"[E3] target={target_task} seed={seed} frac={frac} | "
              f"fullft {out_full['test_rmse']:.3f} | "
              f"lora {out_lora['test_rmse']:.3f} | "
              f"spike {out_sp['test_rmse']:.3f} spars={out_sp['spikelora_sparsity_pct']}")

    return results

"""

✅ Now only run E3 on SR"""

lookback = 96
horizons = [1,2,4,8,24]
spkT = 8
E3_SR_fast = run_E3_one_target_E3(
    target_task="SR",
    source_tasks=["WS","WP","EC"],
    lookback=lookback,
    horizons=horizons,
    seed=0,
    spkT=spkT,
    pretrain_epochs=10,   # faster
    adapt_epochs=10,      # faster
    r=8, alpha=16, Vtheta=0.05,
    fractions=(0.1, 0.2)  # faster sanity
)
pd.DataFrame(E3_SR_fast)

"""✅ After the success of SR: E3 on the four datasets (leave-one-task-out)



"""

tasks = ["SR","WS","WP","EC"]
seeds = [0,1,2]
fractions = (0.1,0.2,0.5)

E3_all = []
for target in tasks:
    sources = [t for t in tasks if t != target]
    for seed in seeds:
        out = run_E3_one_target_E3(
            target_task=target,
            source_tasks=sources,
            lookback=lookback,
            horizons=horizons,
            seed=seed,
            spkT=spkT,
            pretrain_epochs=30,
            adapt_epochs=20,
            r=8, alpha=16, Vtheta=0.05,
            fractions=fractions
        )
        E3_all.extend(out)

E3_df = pd.DataFrame(E3_all)
E3_df.to_csv("E3_leave_one_task_out_aligned.csv", index=False)
E3_df.head()

"""Full Summary for E3 (mean±std)"""

E3_summary = (E3_df
              .groupby(["target","mode","frac"])
              .agg(
                  rmse_mean=("test_rmse","mean"),
                  rmse_std=("test_rmse","std"),
                  mae_mean=("test_mae","mean"),
                  mae_std=("test_mae","std"),
                  smape_mean=("test_smape","mean"),
                  smape_std=("test_smape","std"),
                  trainable_ratio_mean=("trainable_ratio","mean"),
              )
              .reset_index())
E3_summary

"""Drawing Curves for each target"""

import matplotlib.pyplot as plt

def plot_e3(target, metric="rmse_mean"):
    sub = E3_summary[E3_summary["target"] == target].copy()
    plt.figure()
    for mode in ["fullft","lora","spikelora"]:
        m = sub[sub["mode"] == mode].sort_values("frac")
        plt.plot(m["frac"], m[metric], marker="o", label=mode)
    plt.xlabel("Few-shot fraction")
    plt.ylabel(metric)
    plt.title(f"E3 Leave-one-task-out transfer - target={target}")
    plt.grid(True)
    plt.legend()
    plt.show()

for t in tasks:
    plot_e3(t, "rmse_mean")

"""Enhanced E3 (within-domain transfer),

1) We create Union features between WS and SR (except time/target)
"""

1) We create Union features between WS and SR (except time/target)

"""Cell E3’-1 — Adapt on Target (SR أو WP)




"""

def build_union_feature_space_two_tasks(task_a, task_b):
    df_a, cfg_a = load_df(task_a)
    df_b, cfg_b = load_df(task_b)

    base_a, _ = get_feature_cols(df_a, cfg_a, add_calendar=True)
    base_b, _ = get_feature_cols(df_b, cfg_b, add_calendar=True)

    union = []
    seen = set()
    for c in base_a + base_b:
        if c not in seen:
            seen.add(c)
            union.append(c)
    return union

"""2) Pretrain on WS but using the union feature space


"""

def pretrain_base_single_task_aligned(task_id, lookback, horizons, base_cols_union,
                                      seed=0, spkT=8, batch_size=128,
                                      max_epochs=20, lr=1e-3):
    set_seed(seed)

    df, cfg = load_df(task_id)
    X, Y, feat_cols = make_supervised_aligned(
        df, cfg, lookback, horizons, base_cols_union, add_calendar=True
    )
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = chronological_split(X, Y, 0.7, 0.1)
    Xtr, Xva, Xte, _ = standardize_fit_transform(Xtr, Xva, Xte)

    train_loader = DataLoader(WindowDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(WindowDataset(Xva, Yva), batch_size=batch_size, shuffle=False)

    F = Xtr.shape[-1]
    K = Ytr.shape[-1]

    base = SpikingTCN(F, K, T=spkT).to(device)
    base, stats = fit(base, train_loader, val_loader, max_epochs=max_epochs, lr=lr)

    base_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    return base_state, (F, K), stats

"""3) Run E3’ aligned (WS→SR أو WS→WP)"""

def run_E3_prime_aligned(source_task, target_task,
                         lookback, horizons,
                         seed=0, spkT=8,
                         fractions=(0.1, 0.2, 0.5),
                         pretrain_epochs=20, adapt_epochs=15,
                         lr_full=1e-3, lr_fullft=1e-3, lr_peft=5e-3,
                         r=8, alpha=16, Vtheta=0.05,
                         batch_size=128):

    # build union feature space between the 2 tasks
    base_cols_union = build_union_feature_space_two_tasks(source_task, target_task)

    # Stage A: pretrain on source with aligned feature space
    base_state, (F, K), pre_stats = pretrain_base_single_task_aligned(
        source_task, lookback, horizons, base_cols_union,
        seed=seed, spkT=spkT, batch_size=batch_size,
        max_epochs=pretrain_epochs, lr=lr_full
    )

    # Stage B: prepare target splits (aligned)
    df_t, cfg_t = load_df(target_task)
    X, Y, feat_cols = make_supervised_aligned(
        df_t, cfg_t, lookback, horizons, base_cols_union, add_calendar=True
    )
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = chronological_split(X, Y, 0.7, 0.1)
    Xtr, Xva, Xte, _ = standardize_fit_transform(Xtr, Xva, Xte)

    results = []
    for frac in fractions:
        train_loader, val_loader, test_loader, n_fs = make_fewshot_loaders(
            Xtr, Ytr, Xva, Yva, Xte, Yte,
            frac=frac, batch_size=batch_size
        )

        out_full = adapt_and_eval(
            base_state, F, K, spkT, "fullft",
            (train_loader, val_loader, test_loader),
            adapt_epochs=adapt_epochs, lr_adapt=lr_fullft
        )
        out_full.update({"E":"E3-prime","source":source_task,"target":target_task,
                         "seed":seed,"frac":frac,"n_train_windows":n_fs,"mode":"fullft",
                         "pretrain_time_sec": pre_stats["train_time_sec"]})
        results.append(out_full)

        out_lora = adapt_and_eval(
            base_state, F, K, spkT, "lora",
            (train_loader, val_loader, test_loader),
            r=r, alpha=alpha, Vtheta=Vtheta,
            adapt_epochs=adapt_epochs, lr_adapt=lr_peft
        )
        out_lora.update({"E":"E3-prime","source":source_task,"target":target_task,
                         "seed":seed,"frac":frac,"n_train_windows":n_fs,"mode":"lora",
                         "pretrain_time_sec": pre_stats["train_time_sec"]})
        results.append(out_lora)

        out_sp = adapt_and_eval(
            base_state, F, K, spkT, "spikelora",
            (train_loader, val_loader, test_loader),
            r=r, alpha=alpha, Vtheta=Vtheta,
            adapt_epochs=adapt_epochs, lr_adapt=lr_peft
        )
        out_sp.update({"E":"E3-prime","source":source_task,"target":target_task,
                       "seed":seed,"frac":frac,"n_train_windows":n_fs,"mode":"spikelora",
                       "pretrain_time_sec": pre_stats["train_time_sec"]})
        results.append(out_sp)

        print(f"[E3’ aligned] {source_task}->{target_task} seed={seed} frac={frac} | "
              f"fullFT {out_full['test_rmse']:.3f} | "
              f"lora {out_lora['test_rmse']:.3f} | "
              f"spike {out_sp['test_rmse']:.3f} spars={out_sp['spikelora_sparsity_pct']}")

    return results

"""✅ Now run WS → SR (after repair)



"""

lookback = 96
horizons = [1,2,4,8,24]
spkT = 8

E3p_SR = run_E3_prime_aligned(
    source_task="WS",
    target_task="SR",
    lookback=lookback,
    horizons=horizons,
    seed=0,
    spkT=spkT,
    fractions=(0.1, 0.2, 0.5),
    pretrain_epochs=20,
    adapt_epochs=15,
    r=8, alpha=16, Vtheta=0.05
)

pd.DataFrame(E3p_SR)

"""✅ WS → WP (This often gives a better transfer)"""

E3p_WP = run_E3_prime_aligned(
    source_task="WS",
    target_task="WP",
    lookback=lookback,
    horizons=horizons,
    seed=0,
    spkT=spkT,
    fractions=(0.1, 0.2, 0.5),
    pretrain_epochs=20,
    adapt_epochs=15,
    r=8, alpha=16, Vtheta=0.05
)

pd.DataFrame(E3p_WP)
