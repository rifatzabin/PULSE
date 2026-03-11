#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adapt_frel_fewshot.py
---------------------
Few-shot adaptation on a new env (m2, m3, …) using a pretrained FREL encoder.

Flow:
  1) Load target Features/<ENV>/X.npy, y.npy, classes.json
  2) Normalize with source stats (frel_mean/std) or blend with target support
  3) Sample K shots/class for support; rest is query
  4) Compute support embeddings -> class prototypes (cosine)
  5) Classify queries by nearest prototype
  6) (Optional) light fine-tuning on support (CE) with encoder partially frozen
  7) (Optional) TTA / prototype refinement / proto+kNN
  8) Report accuracy & confusion matrix

Requires:
  - frel_encoder.pt  (from pretrain_frel_cnn.py)
  - frel_mean.npy, frel_std.npy  (source stats)
"""

import os, json, argparse
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True


# ---------------------- model (same encoder) ----------------------

class TemporalAttentionPool(nn.Module):
    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(hidden, 1, 1, bias=False),
        )
    def forward(self, x):
        a = torch.softmax(self.net(x), dim=-1)
        return torch.sum(a * x, dim=-1)

class CNNEncoder(nn.Module):
    def __init__(self, in_ch: int, dropout: float = 0.15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 128, 7, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.b1a = nn.Conv1d(128, 256, 5, padding=2)
        self.b1b = nn.Conv1d(256, 256, 3, padding=1)
        self.bn1a, self.bn1b = nn.BatchNorm1d(256), nn.BatchNorm1d(256)
        self.skip1 = nn.Conv1d(128, 256, 1)
        self.drop1 = nn.Dropout(dropout)

        self.b2a = nn.Conv1d(256, 256, 5, padding=4, dilation=2)
        self.b2b = nn.Conv1d(256, 256, 3, padding=2, dilation=2)
        self.bn2a, self.bn2b = nn.BatchNorm1d(256), nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout)

        self.b3a = nn.Conv1d(256, 256, 3, padding=4, dilation=4)
        self.b3b = nn.Conv1d(256, 256, 3, padding=4, dilation=4)
        self.bn3a, self.bn3b = nn.BatchNorm1d(256), nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(dropout)

        self.head_act = nn.GELU()
        self.pool = TemporalAttentionPool(256, hidden=64)
        self.proj = nn.Sequential(nn.Linear(256,256), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):  # [B,C,W]
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
        pooled = self.pool(x)
        emb = self.proj(pooled)   # [B,256]
        return emb


# ---------------------- helpers ----------------------

def set_seed(s: int = 42):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def sample_support_query(X, y, k, seed=0):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    Xs, ys, Xq, yq = [], [], [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        if idx.size < 2:
            continue
        kc = min(k, idx.size - 1)
        sup, qry = idx[:kc], idx[kc:]
        Xs.append(X[sup]); ys.append(np.full((kc,), c, dtype=np.int64))
        Xq.append(X[qry]); yq.append(np.full((qry.size,), c, dtype=np.int64))
    if not Xs:
        raise RuntimeError("No support samples; reduce --shots-per-class or check labels.")
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(Xq), np.concatenate(yq)

@torch.no_grad()
def embed_batches(encoder: nn.Module, X: np.ndarray, bs: int = 512):
    embs = []
    for i in range(0, X.shape[0], bs):
        xb = torch.from_numpy(X[i:i+bs]).to(DEVICE, non_blocking=True)
        z = encoder(xb)
        embs.append(z.detach().cpu())
        del xb, z
        torch.cuda.empty_cache()
    return torch.cat(embs, dim=0)

@torch.no_grad()
def tta_embed(encoder: nn.Module, X: np.ndarray, bs: int = 512, shifts=(0,2,-2,4,-4)):
    """Average embeddings over small temporal rolls."""
    outs = []
    for s in shifts:
        Xs = X if s == 0 else np.roll(X, s, axis=2)
        outs.append(embed_batches(encoder, Xs, bs=bs))
    return torch.stack(outs, dim=0).mean(dim=0)

def class_prototypes(embs: torch.Tensor, labels: torch.Tensor, normalize=True):
    protos = []
    classes = torch.unique(labels).tolist()
    for c in classes:
        m = labels == c
        p = embs[m].mean(dim=0)
        p = F.normalize(p, dim=0) if normalize else p
        protos.append(p)
    P = torch.stack(protos, dim=0)  # [K, D]
    return P, torch.tensor(classes, dtype=torch.long)

def proto_predict_scaled(P: torch.Tensor, embs: torch.Tensor, class_ids: torch.Tensor, scale: float = 16.0):
    Q = F.normalize(embs, dim=1); Pn = F.normalize(P, dim=1)
    S = scale * (Q @ Pn.T)                    # [N,K]
    idx = torch.argmax(S, dim=1)
    return class_ids[idx].cpu().numpy(), S

def refine_prototypes(P: torch.Tensor,
                      class_ids: torch.Tensor,
                      Eq: torch.Tensor,
                      iters: int = 2,
                      tau: float = 0.2,
                      support: tuple | None = None,
                      lam: float = 0.2):
    """
    Soft EM: use queries to refine prototypes.
      tau: softmax temperature over cosine sims
      support: (Es, Ys) to anchor each class prototype (weight 'lam')
    """
    P = F.normalize(P, dim=1)
    Q = F.normalize(Eq, dim=1)
    for _ in range(max(0, iters)):
        S = (Q @ P.T) / max(tau, 1e-6)       # [Nq,K]
        W = torch.softmax(S, dim=1)          # responsibilities
        newP = []
        for k, cid in enumerate(class_ids.tolist()):
            wk = W[:, k].unsqueeze(1)        # [Nq,1]
            pk_q = (wk * Q).sum(0) / (wk.sum() + 1e-8)
            if support is not None:
                Es, Ys = support
                mk = F.normalize(Es[Ys == cid].mean(0), dim=0)
                pk = F.normalize((1 - lam) * pk_q + lam * mk, dim=0)
            else:
                pk = F.normalize(pk_q, dim=0)
            newP.append(pk)
        P = torch.stack(newP, dim=0)
    return P

@torch.no_grad()
def hybrid_proto_knn(P: torch.Tensor,
                     Eq: torch.Tensor,
                     class_ids: torch.Tensor,
                     Es: torch.Tensor,
                     Ys: torch.Tensor,
                     topk: int = 5,
                     margin: float = 0.05):
    """
    Use prototypes first; if (best - second) < margin, fall back to kNN on support.
    """
    Q = F.normalize(Eq, dim=1); Pn = F.normalize(P, dim=1)
    S = Q @ Pn.T
    best2 = torch.topk(S, k=2, dim=1).values
    low_conf = (best2[:, 0] - best2[:, 1]) < margin
    yhat = class_ids[torch.argmax(S, dim=1)].cpu().numpy()

    if low_conf.any():
        Esn = F.normalize(Es, dim=1)
        idx = low_conf.nonzero(as_tuple=False).squeeze(1)
        q = Q[idx]                                        # [M,D]
        dist = torch.cdist(q, Esn)                        # [M,Ns]
        nn_idx = torch.topk(-dist, k=topk, dim=1).indices
        nn_ys = Ys[nn_idx]                                # [M,k]
        vals, _ = torch.mode(nn_ys, dim=1)
        yhat[idx.cpu().numpy()] = vals.cpu().numpy()
    return yhat

def apply_channel_norm(X, mean, std):
    return ((X - mean) / std).astype(np.float32, copy=False)

def parse_shifts(csv: str):
    if csv is None or csv.strip() == "":
        return ()
    return tuple(int(s) for s in csv.split(","))


# ---------------------- main ----------------------

def parse_args():
    ap = argparse.ArgumentParser(description="FREL few-shot adaptation on target env")
    ap.add_argument("--target-features-dir", required=True, help="Features/<target_env>")
    ap.add_argument("--ckpt-dir", required=True, help="Folder with frel_encoder.pt + mean/std")
    ap.add_argument("--shots-per-class", type=int, default=5)
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--freeze-encoder", action="store_true", help="Only prototype classify (no fine-tune).")
    ap.add_argument("--ft-epochs", type=int, default=0, help="Fine-tune epochs on support (0=disable).")
    ap.add_argument("--ft-lr", type=float, default=1e-4)
    ap.add_argument("--ft-weight-decay", type=float, default=1e-5)
    ap.add_argument("--blend-beta", type=float, default=0.0,
                    help="Blend target support stats into source norm: 0=source only, 1=target only.")
    ap.add_argument("--seed", type=int, default=42)

    # ---- Optional accuracy boosters (default OFF) ----
    ap.add_argument("--cos-temp", type=float, default=16.0, help="Cosine scaling (temperature) for prototype scoring.")
    ap.add_argument("--refine-iters", type=int, default=0, help="Prototype refinement EM iterations (0 disables).")
    ap.add_argument("--refine-tau", type=float, default=0.2, help="Temperature for EM responsibilities.")
    ap.add_argument("--refine-lambda", type=float, default=0.2, help="Support anchor weight in EM (0..1).")
    ap.add_argument("--knn-topk", type=int, default=0, help="k for kNN fallback (0 disables).")
    ap.add_argument("--knn-margin", type=float, default=0.05, help="Margin threshold for kNN fallback.")
    ap.add_argument("--tta-shifts", type=str, default="", help="Comma-separated shifts, e.g. '0,2,-2,4,-4'. Empty=disable.")
    ap.add_argument("--ft-augment", action="store_true", help="Augment support batches during FT.")
    ap.add_argument("--l2sp-alpha", type=float, default=0.0, help="L2-SP strength toward source (0 disables).")
    return ap.parse_args()


def jitter_batch(xb, max_shift=4, noise_std=0.01, drop_p=0.05):
    if max_shift > 0:
        s = np.random.randint(-max_shift, max_shift + 1)
        if s != 0:
            xb = torch.roll(xb, shifts=s, dims=2)
    if drop_p > 0 and xb.size(2) > 8:
        m = max(1, int(xb.size(2) * drop_p))
        s0 = np.random.randint(0, xb.size(2) - m + 1)
        xb[:, :, s0:s0 + m] = 0.0
    if noise_std > 0:
        xb = xb + torch.randn_like(xb) * noise_std
    return xb


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load target features
    feat_dir = Path(args.target_features_dir).resolve()
    X = np.load(feat_dir / "X.npy")   # [N,C,W]
    y = np.load(feat_dir / "y.npy")   # [N]
    with open(feat_dir / "classes.json") as f:
        classes = json.load(f)
    N, C, W = X.shape
    print(f"Target loaded: X={X.shape} classes={len(classes)} (C={C}, W={W})")

    # Load encoder + source normalization
    ckpt = Path(args.ckpt_dir).resolve()
    mean_src = np.load(ckpt / "frel_mean.npy")  # [1,C,1]
    std_src  = np.load(ckpt / "frel_std.npy")   # [1,C,1]

    encoder = CNNEncoder(in_ch=C, dropout=0.15).to(DEVICE)
    enc_state = torch.load(ckpt / "frel_encoder.pt", map_location=DEVICE)
    encoder.load_state_dict(enc_state, strict=True)
    encoder.eval()

    # Split support/query
    Xs, ys, Xq, yq = sample_support_query(X, y, args.shots_per_class, seed=args.seed)

    # Optionally blend target support stats into normalization
    if args.blend_beta > 0:
        mean_t = Xs.mean(axis=(0,2), keepdims=True).astype(np.float32)
        std_t  = Xs.std(axis=(0,2), keepdims=True).astype(np.float32)
        std_t  = np.clip(std_t, 1e-6, None)
        b = float(np.clip(args.blend_beta, 0.0, 1.0))
        mean = (1 - b) * mean_src + b * mean_t
        std  = (1 - b) * std_src  + b * std_t
    else:
        mean, std = mean_src, std_src

    # Normalize
    Xs = apply_channel_norm(Xs, mean, std)
    Xq = apply_channel_norm(Xq, mean, std)

    # Optional light FT on support
    if args.ft_epochs > 0:
        encoder.train()
        clf = nn.Linear(256, len(classes)).to(DEVICE)
        # L2-SP: keep a copy of source weights
        src_params = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(clf.parameters()),
            lr=args.ft_lr,
            weight_decay=args.ft_weight_decay
        )
        ce = nn.CrossEntropyLoss()

        if args.freeze_encoder:
            for _, p in encoder.named_parameters():
                p.requires_grad = False
            for p in clf.parameters():
                p.requires_grad = True

        Xs_t = torch.from_numpy(Xs)
        ys_t = torch.from_numpy(ys)
        ds = torch.utils.data.TensorDataset(Xs_t, ys_t)
        dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True, pin_memory=True)

        for ep in range(1, args.ft_epochs + 1):
            total, loss_sum, corr = 0, 0.0, 0
            for xb, yb in dl:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                if args.ft_augment:
                    xb = jitter_batch(xb, max_shift=4, noise_std=0.01, drop_p=0.05)
                z = encoder(xb)
                logits = clf(z)
                loss = ce(logits, yb)

                # L2-SP penalty (if enabled)
                if args.l2sp_alpha > 0:
                    reg = 0.0
                    for name, p in encoder.named_parameters():
                        if not p.requires_grad:
                            continue
                        reg = reg + torch.sum((p - src_params[name].to(p.device))**2)
                    loss = loss + args.l2sp_alpha * reg

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(clf.parameters()), 1.0)
                optimizer.step()
                with torch.no_grad():
                    pred = logits.argmax(1)
                    corr += (pred == yb).sum().item()
                    total += yb.size(0)
                    loss_sum += loss.item() * yb.size(0)
            print(f"    [FT {ep}/{args.ft_epochs}] loss={loss_sum/max(1,total):.4f} acc={corr/max(1,total)*100:.2f}%")
        encoder.eval()

    # Embeddings & prototypes
    Ys = torch.from_numpy(ys)
    # TTA on queries (optional)
    shifts = parse_shifts(args.tta_shifts)
    if len(shifts) > 0:
        Eq = tta_embed(encoder, Xq, bs=args.eval_batch_size, shifts=shifts)
    else:
        Eq = embed_batches(encoder, Xq, bs=args.eval_batch_size)
    Es = embed_batches(encoder, Xs, bs=args.eval_batch_size)

    P, class_ids = class_prototypes(Es, Ys, normalize=True)

    # Prototype refinement (optional)
    if args.refine_iters > 0:
        P = refine_prototypes(P, class_ids, Eq,
                              iters=args.refine_iters,
                              tau=args.refine_tau,
                              support=(Es, Ys),
                              lam=args.refine_lambda)

    # Predict queries (scaled cosine). Optionally hybrid with kNN on low-margin.
    yhat_proto, _ = proto_predict_scaled(P, Eq, class_ids, scale=args.cos_temp)
    if args.knn_topk and args.knn_topk > 0:
        yhat = hybrid_proto_knn(P, Eq, class_ids, Es, Ys, topk=args.knn_topk, margin=args.knn_margin)
    else:
        yhat = yhat_proto

    acc = (yhat == yq).mean()
    rep = classification_report(yq, yhat, digits=3)
    cm  = confusion_matrix(yq, yhat)

    print("\n=== Few-shot (prototype) results ===")
    print(f"K={args.shots_per_class} | acc={acc*100:.2f}% | Nq={len(yq)}")
    print(rep)
    print("Confusion matrix:\n", cm)


if __name__ == "__main__":
    main()
