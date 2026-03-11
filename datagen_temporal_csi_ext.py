#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
datagen_temporal_csi.py  (MAT files, shape (T,F), complex)

Environment = the top folder you pass via --raw (e.g., ../CSI/Classroom).
All subfolders like m1/m2/m3 are merged into ONE consolidated feature set.
Classes are the subfolders (A..T). Each class can appear under multiple m*.

Outputs (under --out):
  X.npy  [N, C, W]
  y.npy  [N]
  mean.npy  [C]
  std.npy   [C]
  classes.json  {"A":0, "B":1, ...}

Per-band features (base 8) + optional log_energy (+1):
  1) amp_mean        = mean_Fb |H|
  2) amp_std         = std_Fb  |H|
  3) amp_median      = median_Fb |H|
  4) amp_iqr         = p75 - p25 of |H|
  5) d_amp_mean      = diff_t amp_mean
  6) phase_mean_dt   = diff_t mean_Fb unwrap(angle(H))
  7) phase_std       = std_Fb unwrap(angle(H))
  8) d_phase_mean    = diff_t mean_Fb unwrap(angle(H)) (alias)
  9) log_energy      = log(eps + mean_Fb |H|^2)   (if --log-energy)

Total channels C = bands * (8 + log_energy?1:0)

Example:
  python datagen_temporal_csi.py \
    --raw ../CSI/Classroom --out ../Features/Classroom \
    --window 64 --stride 64 --bands 8 --log-energy
"""

import argparse, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.io import loadmat


# ------------------------ utils ------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)

def _natkey(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]


# ------------------------ I/O ------------------------

def load_csi(path: Path) -> np.ndarray:
    """
    Load CSI from .mat, expected key 'csi' or first non-meta variable.
    Return complex64 array of shape (T, F).
    """
    mat = loadmat(path)
    if 'csi' in mat:
        X = mat['csi']
    else:
        keys = [k for k in mat.keys() if not k.startswith('__')]
        if not keys:
            raise ValueError(f"No valid CSI variable in {path}")
        X = mat[keys[0]]
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected shape (T,F), got {X.shape} in {path}")
    if not np.iscomplexobj(X):
        raise ValueError(f"Expected complex CSI in {path}, got dtype={X.dtype}")
    return X.astype(np.complex64)


# ------------------------ features ------------------------

def _band_edges(F: int, bands: int) -> np.ndarray:
    bands = max(1, int(bands))
    return np.linspace(0, F, bands + 1, dtype=int)

def compute_temporal_features_banded(
    csi_tf: np.ndarray,
    bands: int = 1,
    use_log_energy: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    csi_tf: complex array (T, F)
    Returns features (C_total, T) float32 with C_total = bands * (8 + logE?1:0).
    """
    T, F = csi_tf.shape
    edges = _band_edges(F, bands)
    out = []

    # Precompute once
    ph_unwrap = np.unwrap(np.angle(csi_tf), axis=0)  # (T, F)
    amp_full  = np.abs(csi_tf)                       # (T, F)

    for b in range(bands):
        s, e = edges[b], edges[b+1]
        amp = amp_full[:, s:e]           # (T, Fb)
        ph  = ph_unwrap[:, s:e]          # (T, Fb)

        # amplitude stats per time
        amp_mean   = amp.mean(axis=1)                          # (T,)
        amp_std    = amp.std(axis=1)                           # (T,)
        amp_med    = np.median(amp, axis=1)                    # (T,)
        q25        = np.percentile(amp, 25, axis=1)
        q75        = np.percentile(amp, 75, axis=1)
        amp_iqr    = (q75 - q25).astype(np.float32)

        # temporal diffs
        d_amp_mean = np.diff(amp_mean, prepend=amp_mean[:1])   # (T,)

        # phase stats per time
        ph_mean    = ph.mean(axis=1)                           # (T,)
        ph_std     = ph.std(axis=1)                            # (T,)
        d_ph_mean  = np.diff(ph_mean, prepend=ph_mean[:1])     # (T,)
        ph_mean_dt = d_ph_mean                                 # alias

        feats = [
            amp_mean.astype(np.float32),
            amp_std.astype(np.float32),
            amp_med.astype(np.float32),
            amp_iqr.astype(np.float32),
            d_amp_mean.astype(np.float32),
            ph_mean_dt.astype(np.float32),
            ph_std.astype(np.float32),
            d_ph_mean.astype(np.float32),
        ]

        if use_log_energy:
            logE = np.log(eps + (amp * amp).mean(axis=1)).astype(np.float32)
            feats.append(logE)

        out.append(np.stack(feats, axis=0))  # (C_band, T)

    Ft = np.concatenate(out, axis=0)  # (C_total, T)
    return Ft


# ------------------------ windowing & stats ------------------------

def window_series(feat_ct: np.ndarray, window: int, stride: int) -> np.ndarray:
    """
    Slice (C,T) -> (N,C,W).
    If stride <= 0, we use stride=window (non-overlap).
    """
    C, T = feat_ct.shape
    stride = window if stride <= 0 else stride
    if T < window:
        return np.zeros((0, C, window), dtype=feat_ct.dtype)
    starts = np.arange(0, T - window + 1, stride, dtype=int)
    out = np.empty((len(starts), C, window), dtype=feat_ct.dtype)
    for i, s in enumerate(starts):
        out[i] = feat_ct[:, s:s + window]
    return out

def per_channel_mean_std(X_ncw: np.ndarray):
    """Compute per-channel mean/std over (N*W) for (N,C,W)."""
    N, C, W = X_ncw.shape
    flat = X_ncw.transpose(1,0,2).reshape(C, N*W)
    mean = flat.mean(axis=1).astype(np.float32)
    std  = flat.std(axis=1).astype(np.float32)
    std  = np.clip(std, 1e-6, None)
    return mean, std


# ------------------------ discovery (merge m1/m2/m3) ------------------------

def discover_class_files_merged(raw_root: Path) -> dict[str, list[Path]]:
    """
    Merge all m* subfolders (if present). Supports two layouts:

    Layout A (with m-folders):
      raw_root/
        m1/A/*.mat
        m1/B/*.mat
        ...
        m2/A/*.mat
        ...

    Layout B (flat):
      raw_root/
        A/*.mat
        B/*.mat
        ...

    Returns: { "A": [paths...], "B": [...], ... }  (sorted by natural order)
    """
    class_to_files = defaultdict(list)

    subs = [p for p in raw_root.iterdir() if p.is_dir()]
    subs.sort(key=lambda p: _natkey(p.name))

    had_any_at_root = False
    # Case B: classes directly under raw_root
    for cdir in subs:
        mats = sorted(cdir.glob("*.mat"))
        if mats:
            had_any_at_root = True
            class_to_files[cdir.name].extend(mats)

    if not had_any_at_root:
        # Case A: m* under raw_root
        for mdir in subs:  # m1, m2, ...
            if not mdir.is_dir():
                continue
            csubs = [p for p in mdir.iterdir() if p.is_dir()]
            csubs.sort(key=lambda p: _natkey(p.name))
            for cdir in csubs:  # A, B, ...
                mats = sorted(cdir.glob("*.mat"))
                if mats:
                    class_to_files[cdir.name].extend(mats)

    # natural sort class names
    ordered = {}
    for cls in sorted(class_to_files.keys(), key=_natkey):
        ordered[cls] = class_to_files[cls]
    return ordered


# ------------------------ build ONE environment (merged) ------------------------

def build_features_for_environment(
    raw_root: Path,
    out_root: Path,
    window: int,
    stride: int,
    bands: int,
    use_log_energy: bool,
    seed: int,
):
    """
    Build ONE consolidated feature set for the given environment folder (raw_root).
    All m* subfolders (if any) are merged. Output saved under out_root.
    """
    set_seed(seed)
    if not raw_root.exists():
        raise FileNotFoundError(f"Missing environment folder: {raw_root}")

    class_to_files = discover_class_files_merged(raw_root)
    if not class_to_files:
        raise RuntimeError(f"No .mat files found under {raw_root} (neither flat nor via m*/class/*).")

    # Stable class id map
    class_names = sorted(class_to_files.keys(), key=_natkey)
    class_to_id = {cls: i for i, cls in enumerate(class_names)}
    print(f"[env={raw_root.name}] classes: {class_to_id}")

    all_X, all_y = [], []
    per_class_counts = defaultdict(int)

    for cls in class_names:
        cid = class_to_id[cls]
        files = class_to_files[cls]
        print(f"  - class '{cls}' ({cid}): {len(files)} files (merged across m*)")
        for fp in files:
            try:
                csi_tf = load_csi(fp)  # (T, F)
                T, Fin = csi_tf.shape
                feat_ct = compute_temporal_features_banded(
                    csi_tf, bands=bands, use_log_energy=use_log_energy
                )  # (C_total, T)
                Ctot = feat_ct.shape[0]
                win_ncw = window_series(feat_ct, window=window, stride=stride)  # (N,C,W)
                Nw = win_ncw.shape[0]
                if Nw == 0:
                    print(f"    skip {fp.name}: T={T} < window={window}")
                    continue
                all_X.append(win_ncw)
                all_y.append(np.full((Nw,), cid, dtype=np.int64))
                per_class_counts[cid] += Nw
                print(f"    file={fp.name}  in:(T={T},F={Fin})  out_windows={Nw}  window=(C={Ctot},W={window})")
            except Exception as e:
                print(f"    ! error {fp.name}: {e}")

    if not all_X:
        raise RuntimeError("No windows produced. Check your window/stride vs T in files.")

    X = np.concatenate(all_X, axis=0).astype(np.float32)  # (N,C,W)
    y = np.concatenate(all_y, axis=0).astype(np.int64)    # (N,)

    # shuffle
    idx = np.random.permutation(X.shape[0])
    X, y = X[idx], y[idx]

    # stats
    mean, std = per_channel_mean_std(X)

    out_root.mkdir(parents=True, exist_ok=True)
    np.save(out_root / "X.npy", X)
    np.save(out_root / "y.npy", y)
    np.save(out_root / "mean.npy", mean)
    np.save(out_root / "std.npy", std)
    with open(out_root / "classes.json", "w") as f:
        json.dump(class_to_id, f, indent=2)

    print("\n[Summary]")
    print(f"  Saved to: {out_root}")
    print(f"  X shape: {X.shape}  (N,C,W)  | y: {y.shape}")
    print(f"  C (feature channels): {X.shape[1]}  | W (window): {X.shape[2]}")
    print(f"  Per-class window counts: {dict(per_class_counts)}")
    print("  mean/std saved per channel.")
    Xc = X.transpose(1,0,2).reshape(X.shape[1], -1)
    print(f"  Feature[0] min/max: {float(Xc[0].min()):.4f}/{float(Xc[0].max()):.4f}")


# ------------------------ CLI ------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Temporal CSI datagen (merge all m1/m2/m3) into ONE set for the given environment."
    )
    ap.add_argument("--raw", required=True, help="Environment root (e.g., ../CSI/Classroom)")
    ap.add_argument("--out", required=True, help="Output folder (e.g., ../Features/Classroom)")
    ap.add_argument("--window", type=int, default=64, help="Window length (time samples)")
    ap.add_argument("--stride", type=int, default=64,
                    help="Stride between windows (default 64). Use same as --window for non-overlap.")
    ap.add_argument("--bands", type=int, default=8, help="Number of subcarrier bands (≥1)")
    ap.add_argument("--log-energy", action="store_true",
                    help="Include log-energy feature (adds +1 feature per band)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    raw_root = Path(args.raw).resolve()
    out_root = Path(args.out).resolve()

    build_features_for_environment(
        raw_root=raw_root,
        out_root=out_root,
        window=int(args.window),
        stride=int(args.stride),
        bands=int(args.bands),
        use_log_energy=bool(args.log_energy),
        seed=int(args.seed),
    )

    # --- sanity echo ---
    X = np.load(out_root / "X.npy")
    y = np.load(out_root / "y.npy")
    print(f"\n✅ Sanity Check:")
    print(f"  X shape = {X.shape}  [N,C,W]")
    print(f"  y shape = {y.shape}  [N]")
    print(f"  per-window shape (C,W) = {X.shape[1:]}")
    print(f"  mean/std shapes: {np.load(out_root/'mean.npy').shape}, {np.load(out_root/'std.npy').shape}")


if __name__ == "__main__":
    main()
