#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_cnn.py
-----------
Evaluate a trained CNN model on a given feature set.

Usage:
  python eval_cnn.py \
    --features-dir ../Features/Kitchen \
    --ckpt-dir ../Checkpoints/Classroom/CNN_b8_w64 \
    --batch-size 64
"""

import os, json, argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

import torch
from torch.utils.data import Dataset, DataLoader
from models_1dcnn import CNNClassifier  # same backbone as train_cnn.py

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True


# ---------------------- dataset ----------------------

class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])


# ---------------------- helpers ----------------------

def apply_channel_norm(X, mean, std):
    return ((X - mean) / std).astype(np.float32, copy=False)

@torch.no_grad()
def evaluate(model, loader):
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


# ---------------------- main ----------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate trained CNN model")
    ap.add_argument("--features-dir", required=True, help="Folder with X.npy/y.npy/classes.json")
    ap.add_argument("--ckpt-dir", required=True, help="Folder with cnn_model.pt and mean/std")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    return ap.parse_args()


def main():
    args = parse_args()
    feat_dir = Path(args.features_dir).resolve()
    ckpt_dir = Path(args.ckpt_dir).resolve()

    # --- load features ---
    X = np.load(feat_dir / "X.npy")
    y = np.load(feat_dir / "y.npy")
    with open(feat_dir / "classes.json") as f:
        classes = json.load(f)
    id2cls = {v: k for k, v in classes.items()}
    N, C, W = X.shape
    print(f"Loaded target features: X={X.shape} y={y.shape} classes={len(classes)}")

    # --- load model + norm ---
    mean = np.load(ckpt_dir / "cnn_mean.npy")
    std  = np.load(ckpt_dir / "cnn_std.npy")
    Xn = apply_channel_norm(X, mean, std)

    model = CNNClassifier(in_ch=C, n_classes=len(classes)).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_dir / "cnn_model.pt", map_location=DEVICE), strict=True)
    model.eval()

    loader = DataLoader(WindowDataset(Xn, y),
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True)

    # --- evaluate ---
    acc, rep, cm = evaluate(model, loader)
    print(f"\n=== Evaluation Results ===")
    print(f"Accuracy: {acc*100:.2f}%")
    print(rep)
    print("Confusion matrix (rows=true, cols=pred):\n", cm)


if __name__ == "__main__":
    main()
