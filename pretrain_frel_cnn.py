#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pretrain_frel_cnn.py
--------------------
Pretrain a CNN encoder on one environment (e.g., m1) to produce
reusable embeddings with CE + Supervised Contrastive (SupCon) loss.

Inputs (from datagen_temporal_csi.py):
  Features/<ENV>/X.npy        # [N, C, W] float32
  Features/<ENV>/y.npy        # [N] int64
  Features/<ENV>/classes.json

Outputs (ckpt_dir):
  frel_encoder.pt             # encoder state_dict (for adaptation)
  frel_classifier.pt          # linear classifier head (optional)
  frel_proj_head.pt           # MLP projection head used during pretrain
  frel_mean.npy               # [1,C,1] channel mean (fit on train)
  frel_std.npy                # [1,C,1] channel std  (fit on train)
  classes.json                # copy-through
  meta.json                   # training meta
"""

import os, json, argparse, math
from pathlib import Path
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report, confusion_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True


# ---------------------- model ----------------------

class TemporalAttentionPool(nn.Module):
    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(hidden, 1, 1, bias=False),
        )

    def forward(self, x):  # x: [B,C,W]
        a = torch.softmax(self.net(x), dim=-1)  # [B,1,W]
        return torch.sum(a * x, dim=-1)         # [B,C]


class CNNEncoder(nn.Module):
    """1D CNN encoder -> 256-d embedding (penultimate)."""
    def __init__(self, in_ch: int, dropout: float = 0.15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        # Block 1
        self.b1a = nn.Conv1d(128, 256, 5, padding=2, dilation=1)
        self.b1b = nn.Conv1d(256, 256, 3, padding=1, dilation=1)
        self.bn1a, self.bn1b = nn.BatchNorm1d(256), nn.BatchNorm1d(256)
        self.skip1 = nn.Conv1d(128, 256, 1)
        self.drop1 = nn.Dropout(dropout)
        # Block 2 (dilated)
        self.b2a = nn.Conv1d(256, 256, 5, padding=4, dilation=2)
        self.b2b = nn.Conv1d(256, 256, 3, padding=2, dilation=2)
        self.bn2a, self.bn2b = nn.BatchNorm1d(256), nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout)
        # Block 3 (dilated)
        self.b3a = nn.Conv1d(256, 256, 3, padding=4, dilation=4)
        self.b3b = nn.Conv1d(256, 256, 3, padding=4, dilation=4)
        self.bn3a, self.bn3b = nn.BatchNorm1d(256), nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(dropout)

        self.head_act = nn.GELU()
        self.pool = TemporalAttentionPool(256, hidden=64)
        self.proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):  # x: [B,C,W]
        x = self.stem(x)
        y = self.bn1a(self.b1a(x)); y = F.gelu(y)
        y = self.bn1b(self.b1b(y))
        x = F.gelu(y + self.skip1(x)); x = self.drop1(x)

        y = self.bn2a(self.b2a(x)); y = F.gelu(y)
        y = self.bn2b(self.b2b(y))
        x = F.gelu(y + x); x = self.drop2(x)

        y = self.bn3a(self.b3a(x)); y = F.gelu(y)
        y = self.bn3b(self.b3b(y))
        x = F.gelu(y + x); x = self.drop3(x)

        x = self.head_act(x)
        pooled = self.pool(x)         # [B,256]
        emb = self.proj(pooled)       # [B,256]
        return emb


class FRELNet(nn.Module):
    """Encoder + classifier + SupCon projection head."""
    def __init__(self, in_ch: int, n_classes: int, dropout: float = 0.15, proj_dim: int = 128):
        super().__init__()
        self.encoder = CNNEncoder(in_ch, dropout=dropout)
        self.classifier = nn.Linear(256, n_classes)
        # projection head used only during pretraining (SupCon)
        self.proj_head = nn.Sequential(
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, proj_dim)
        )

    def forward(self, x, return_emb=False, return_z=False):
        emb = self.encoder(x)            # [B,256]
        logits = self.classifier(emb)    # [B,K]
        if return_z:
            z = F.normalize(self.proj_head(emb), dim=1)
            return logits, emb, z
        if return_emb:
            return logits, emb
        return logits


# ---------------------- data ----------------------

class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, augment=False, max_shift=4, noise_std=0.01, time_mask=True):
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.augment = augment
        self.max_shift = max_shift
        self.noise_std = noise_std
        self.time_mask = time_mask

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, i):
        x = self.X[i]  # [C,W]
        if self.augment:
            if self.max_shift > 0:
                s = np.random.randint(-self.max_shift, self.max_shift + 1)
                if s != 0:
                    x = np.roll(x, s, axis=1)
            if self.time_mask and x.shape[1] > 16:
                m = np.random.randint(0, x.shape[1] // 6)
                s0 = np.random.randint(0, x.shape[1] - m + 1)
                x[:, s0:s0+m] = 0.0
            if self.noise_std > 0:
                x = x + np.random.randn(*x.shape).astype(np.float32) * self.noise_std
            # light per-channel gain jitter
            scale = 1.0 + np.random.randn(x.shape[0]).astype(np.float32) * 0.05
            x = (x.T * scale).T
        return torch.from_numpy(x), torch.tensor(self.y[i])


def stratified_split(y: np.ndarray, train=0.7, val=0.15, test=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx_tr, idx_va, idx_te = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        ntr = int(round(train * n))
        nva = int(round(val * n))
        idx_tr.append(idx[:ntr])
        idx_va.append(idx[ntr:ntr+nva])
        idx_te.append(idx[ntr+nva:])
    return np.concatenate(idx_tr), np.concatenate(idx_va), np.concatenate(idx_te)


def fit_channel_norm(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=(0,2), keepdims=True)   # [1,C,1]
    std  = X.std(axis=(0,2), keepdims=True)
    std  = np.clip(std, 1e-6, None)
    return mean.astype(np.float32), std.astype(np.float32)

def apply_channel_norm(X, mean, std):
    return ((X - mean) / std).astype(np.float32, copy=False)


# ---------------------- losses ----------------------

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al. 2020)
    Expects z: [B, dim] normalized, y: [B].
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.t = temperature

    def forward(self, z, y):
        z = F.normalize(z, dim=1)
        logits = z @ z.T / self.t               # [B,B]
        logits = logits - torch.max(logits, dim=1, keepdim=True).values  # stabilize
        mask = torch.eq(y.unsqueeze(1), y.unsqueeze(0)).float()          # positives
        # exclude self-comparisons
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
        mask = mask * logits_mask

        # log_prob for each anchor over all positives
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        # mean over positives for each anchor
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        return loss


# ---------------------- train / eval ----------------------

def train_epoch(model, loader, opt, ce_loss, sc_loss, supcon_weight: float):
    model.train()
    tot, corr, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        logits, emb, z = model(xb, return_z=True)
        loss = ce_loss(logits, yb) + supcon_weight * sc_loss(z, yb)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            pred = logits.argmax(1)
            corr += (pred == yb).sum().item()
            tot  += yb.size(0)
            loss_sum += loss.item() * yb.size(0)
    return loss_sum / max(1, tot), corr / max(1, tot)

@torch.no_grad()
def evaluate_ce(model, loader):
    model.eval()
    ce = nn.CrossEntropyLoss()
    tot, corr, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        loss = ce(logits, yb)
        pred = logits.argmax(1)
        corr += (pred == yb).sum().item()
        tot  += yb.size(0)
        loss_sum += loss.item() * yb.size(0)
    return loss_sum / max(1, tot), corr / max(1, tot)

@torch.no_grad()
def eval_full(model, loader):
    model.eval()
    preds, gts = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        preds.append(logits.argmax(1).cpu().numpy())
        gts.append(yb.numpy())
    yhat = np.concatenate(preds)
    y    = np.concatenate(gts)
    acc  = (yhat == y).mean()
    rep  = classification_report(y, yhat, digits=3)
    cm   = confusion_matrix(y, yhat)
    return acc, rep, cm


# ---------------------- CLI ----------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Pretrain FREL CNN (CE + SupCon)")
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--supcon-weight", type=float, default=0.2)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--test-split", type=float, default=0.15)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--early-stop", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=2)
    return ap.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load data
    feat_dir = Path(args.features_dir).resolve()
    ckpt_dir = Path(args.ckpt_dir).resolve(); ckpt_dir.mkdir(parents=True, exist_ok=True)
    X = np.load(feat_dir / "X.npy")   # [N,C,W]
    y = np.load(feat_dir / "y.npy")   # [N]
    with open(feat_dir / "classes.json") as f:
        classes = json.load(f)

    N, C, W = X.shape
    print(f"Loaded: X={X.shape} classes={len(classes)} (C={C}, W={W})")

    # Splits
    tr_idx, va_idx, te_idx = stratified_split(
        y, train=1.0 - args.val_split - args.test_split,
        val=args.val_split, test=args.test_split, seed=args.seed
    )
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]
    Xte, yte = X[te_idx], y[te_idx]

    # Norm (fit on train only)
    mean = Xtr.mean(axis=(0,2), keepdims=True).astype(np.float32)
    std  = Xtr.std(axis=(0,2), keepdims=True).astype(np.float32)
    std  = np.clip(std, 1e-6, None)
    Xtr = ((Xtr - mean) / std).astype(np.float32)
    Xva = ((Xva - mean) / std).astype(np.float32)
    Xte = ((Xte - mean) / std).astype(np.float32)

    # Loaders
    tr_loader = DataLoader(WindowDataset(Xtr, ytr, augment=args.augment),
                           batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True)
    va_loader = DataLoader(WindowDataset(Xva, yva, augment=False),
                           batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    te_loader = DataLoader(WindowDataset(Xte, yte, augment=False),
                           batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)

    # Model
    n_classes = len(classes)
    model = FRELNet(in_ch=C, n_classes=n_classes, dropout=args.dropout, proj_dim=args.proj_dim).to(DEVICE)

    # Loss/Opt/Sched
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.02)
    sc_loss = SupConLoss(temperature=args.temperature)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine LR w/ warmup (works well with SupCon)
    warmup = max(5, args.epochs // 20)
    total  = args.epochs
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        t = (ep - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Train
    best_val, best_state = float("inf"), None
    patience = 0
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, tr_loader, opt, ce_loss, sc_loss, args.supcon_weight)
        va_loss, va_acc = evaluate_ce(model, va_loader)
        sched.step()

        print(f"[{ep:03d}/{args.epochs}] "
              f"train loss {tr_loss:.4f} acc {tr_acc*100:5.2f}% | "
              f"val loss {va_loss:.4f} acc {va_acc*100:5.2f}%  lr={opt.param_groups[0]['lr']:.3e}")

        if va_loss + 1e-6 < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop:
                print(f"Early stopping at epoch {ep} (no val improvement for {args.early_stop} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    # Final test (sanity)
    test_acc, rep, cm = eval_full(model, te_loader)
    print("\n=== Test (CE head) ===")
    print(f"Accuracy: {test_acc*100:.2f}%")
    print(rep)
    print("Confusion matrix:\n", cm)

    # Save encoder + heads + norm
    torch.save(model.encoder.state_dict(), ckpt_dir / "frel_encoder.pt")
    torch.save(model.classifier.state_dict(), ckpt_dir / "frel_classifier.pt")
    torch.save(model.proj_head.state_dict(), ckpt_dir / "frel_proj_head.pt")
    np.save(ckpt_dir / "frel_mean.npy", mean)
    np.save(ckpt_dir / "frel_std.npy",  std)
    with open(ckpt_dir / "classes.json", "w") as f:
        json.dump(classes, f, indent=2)
    meta = {
        "C": int(C), "W": int(W), "epochs": int(ep),
        "best_val_loss": float(best_val),
        "test_acc": float(test_acc),
        "proj_dim": int(args.proj_dim),
        "temperature": float(args.temperature),
        "supcon_weight": float(args.supcon_weight),
        "dropout": float(args.dropout)
    }
    with open(ckpt_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved encoder & heads to: {ckpt_dir}")


if __name__ == "__main__":
    main()
