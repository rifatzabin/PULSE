#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_cnn.py
------------
Train baseline temporal CNN on one environment (e.g., m1).

Inputs (from datagen_temporal_csi.py):
  Features/<ENV>/X.npy        # [N, C, W] float32
  Features/<ENV>/y.npy        # [N] int64
  Features/<ENV>/classes.json

Outputs:
  <ckpt_dir>/cnn_model.pt
  <ckpt_dir>/cnn_mean.npy     # [1,C,1] mean (fit on train)
  <ckpt_dir>/cnn_std.npy      # [1,C,1] std  (fit on train)
  <ckpt_dir>/classes.json
  <ckpt_dir>/meta.json
"""

import os, json, argparse, math
import numpy as np
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report, confusion_matrix

# import the backbone
from models_1dcnn import CNNClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True


# ---------------------- data helpers ----------------------

class WindowDataset(Dataset):
    def __init__(self, X, y, augment=False, max_shift=4, noise_std=0.01):
        self.X = X.astype(np.float32, copy=False)  # [N,C,W]
        self.y = y.astype(np.int64, copy=False)
        self.augment = augment
        self.max_shift = max_shift
        self.noise_std = noise_std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = self.X[i]  # [C,W]
        if self.augment:
            # small circular time shift
            if self.max_shift > 0:
                s = np.random.randint(-self.max_shift, self.max_shift + 1)
                if s != 0:
                    x = np.roll(x, s, axis=1)
            if self.noise_std > 0:
                x = x + np.random.randn(*x.shape).astype(np.float32) * self.noise_std
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
        idx_va.append(idx[ntr : ntr + nva])
        idx_te.append(idx[ntr + nva :])
    return np.concatenate(idx_tr), np.concatenate(idx_va), np.concatenate(idx_te)


def fit_channel_norm(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # X: [N,C,W]  -> per-channel mean/std over (N,W)
    mean = X.mean(axis=(0, 2), keepdims=True)  # [1,C,1]
    std  = X.std(axis=(0, 2), keepdims=True)   # [1,C,1]
    std  = np.clip(std, 1e-6, None)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_channel_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32, copy=False)


# ---------------------- losses/scheduler ----------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.eps = label_smoothing

    def forward(self, logits, target):
        if self.eps > 0:
            num_classes = logits.size(1)
            with torch.no_grad():
                true = torch.zeros_like(logits).fill_(self.eps / (num_classes - 1))
                true.scatter_(1, target.unsqueeze(1), 1 - self.eps)
            logp = F.log_softmax(logits, dim=1)
            ce = -(true * logp).sum(dim=1)
        else:
            ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ---------------------- training ----------------------

def train_one_epoch(model, loader, opt, crit):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        logits, _ = model(xb)
        loss = crit(logits, yb)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        with torch.no_grad():
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
            loss_sum += loss.item() * yb.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    crit = nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        logits, _ = model(xb)
        loss = crit(logits, yb)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
        loss_sum += loss.item() * yb.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_full(model, loader):
    model.eval()
    preds, gts = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        logits, _ = model(xb)
        preds.append(logits.argmax(1).cpu().numpy())
        gts.append(yb.numpy())
    yhat = np.concatenate(preds)
    y    = np.concatenate(gts)
    acc  = (yhat == y).mean()
    rep  = classification_report(y, yhat, digits=3)
    cm   = confusion_matrix(y, yhat)
    return acc, rep, cm


# ---------------------- CLI / main ----------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Train baseline CNN on one environment")
    ap.add_argument("--features-dir", required=True, help="Folder with X.npy/y.npy/classes.json")
    ap.add_argument("--ckpt-dir", required=True, help="Where to save the checkpoint")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--use-focal", action="store_true", help="Use FocalLoss instead of CE")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--early-stop", type=int, default=30, help="Stop if no val loss improve")
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--test-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=2)
    return ap.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    feat_dir = Path(args.features_dir).resolve()
    ckpt_dir = Path(args.ckpt_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load features/labels/classes
    X = np.load(feat_dir / "X.npy")  # [N,C,W]
    y = np.load(feat_dir / "y.npy")  # [N]
    with open(feat_dir / "classes.json") as f:
        classes = json.load(f)
    id2cls = {v: k for k, v in classes.items()}

    N, C, W = X.shape
    print(f"Loaded: X={X.shape}  y={y.shape}  classes={len(classes)} (C={C}, W={W})")

    # Split
    tr_idx, va_idx, te_idx = stratified_split(
        y, train=1.0 - args.val_split - args.test_split,
        val=args.val_split, test=args.test_split, seed=args.seed
    )
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]
    Xte, yte = X[te_idx], y[te_idx]

    # Fit normalization on train only
    mean, std = fit_channel_norm(Xtr)
    Xtr = apply_channel_norm(Xtr, mean, std)
    Xva = apply_channel_norm(Xva, mean, std)
    Xte = apply_channel_norm(Xte, mean, std)

    # Dataloaders
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
    model = CNNClassifier(in_ch=C, n_classes=n_classes, dropout=args.dropout).to(DEVICE)

    # Loss & Optim
    if args.use_focal:
        crit = FocalLoss(gamma=1.5, label_smoothing=args.label_smoothing)
    else:
        crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    # Train loop with early stopping
    best_val = float("inf")
    best_state = None
    no_improve = 0

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, opt, crit)
        va_loss, va_acc = evaluate(model, va_loader)
        sched.step(va_loss)

        print(f"[{ep:03d}/{args.epochs}] "
              f"train loss {tr_loss:.4f} acc {tr_acc*100:5.2f}% | "
              f"val loss {va_loss:.4f} acc {va_acc*100:5.2f}%  lr={opt.param_groups[0]['lr']:.3e}")

        if va_loss + 1e-6 < best_val:
            best_val = va_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.early_stop:
            print(f"Early stopping at epoch {ep} (no val improvement for {args.early_stop} epochs).")
            break

    # Load best and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    test_acc, rep, cm = eval_full(model, te_loader)
    print("\n=== Test Results ===")
    print(f"Accuracy: {test_acc*100:.2f}%")
    print(rep)
    print("Confusion matrix (rows=true, cols=pred):\n", cm)

    # Save checkpoint and normalization
    np.save(ckpt_dir / "cnn_mean.npy", mean)
    np.save(ckpt_dir / "cnn_std.npy",  std)
    with open(ckpt_dir / "classes.json", "w") as f:
        json.dump(classes, f, indent=2)

    torch.save(model.state_dict(), ckpt_dir / "cnn_model.pt")
    meta = {
        "in_ch": int(C),
        "window": int(W),
        "epochs_trained": int(ep),
        "best_val_loss": float(best_val),
        "test_acc": float(test_acc),
        "augment": bool(args.augment),
        "use_focal": bool(args.use_focal),
        "label_smoothing": float(args.label_smoothing),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
    }
    with open(ckpt_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved checkpoint to: {ckpt_dir}")


if __name__ == "__main__":
    main()
