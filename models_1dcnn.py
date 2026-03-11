# models_1dcnn.py
# Minimal temporal CNN backbone + heads for CNN and FREL

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Attention pooling over time
# ------------------------------
class TemporalAttentionPool1D(nn.Module):
    """
    Learn a soft attention distribution over time and aggregate the
    temporal feature map into a single vector.
    """
    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),   
            nn.GELU(),                                              
            nn.Conv1d(hidden, 1, kernel_size=1, bias=False),          
        )

    def forward(self, x):  # x: [B, C, T]
        att = torch.softmax(self.score(x), dim=-1)  # [B,1,T]
        return torch.sum(att * x, dim=-1)           # [B,C]

# ------------------------------
# Backbone (temporal 1D CNN)
# ------------------------------
class CNN1DBackbone(nn.Module):
    """
    Input:  [B, in_ch, W]  (in_ch = # temporal features; e.g., 4 or 5)
    Output: [B, 256] embedding (after attention pool + projection pre-activation)
    """
    def __init__(self, in_ch: int = 5, dropout: float = 0.15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 128, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU()
        )

        # Block 1: expand channels from 128 -> 256.
        self.b1a = nn.Conv1d(128, 256, kernel_size=5, padding=2, dilation=1, bias=False)
        self.b1b = nn.Conv1d(256, 256, kernel_size=3, padding=1, dilation=1, bias=False)
        self.bn1a, self.bn1b = nn.BatchNorm1d(256), nn.BatchNorm1d(256)
        self.skip1 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.drop1 = nn.Dropout(dropout)

        # Block 2 (dilated convolutions increase temporal receptive field)
        self.b2a = nn.Conv1d(256, 256, kernel_size=5, padding=4, dilation=2, bias=False)
        self.b2b = nn.Conv1d(256, 256, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn2a, self.bn2b = nn.BatchNorm1d(256), nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout)

        # Block 3 (further enlarge temporal context)
        self.b3a = nn.Conv1d(256, 256, kernel_size=3, padding=4, dilation=4, bias=False)
        self.b3b = nn.Conv1d(256, 256, kernel_size=3, padding=4, dilation=4, bias=False)
        self.bn3a, self.bn3b = nn.BatchNorm1d(256), nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(dropout)

        self.post = nn.GELU()
        self.pool = TemporalAttentionPool1D(256, hidden=64)
        self.proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):  # x: [B,in_ch,W]
        x = self.stem(x)

        y = self.bn1a(self.b1a(x)); y = F.gelu(y)
        y = self.bn1b(self.b1b(y))
        x = F.gelu(y + self.skip1(x))
        x = self.drop1(x)

        y = self.bn2a(self.b2a(x)); y = F.gelu(y)
        y = self.bn2b(self.b2b(y))
        x = F.gelu(y + x)
        x = self.drop2(x)

        y = self.bn3a(self.b3a(x)); y = F.gelu(y)
        y = self.bn3b(self.b3b(y))
        x = F.gelu(y + x)
        x = self.drop3(x)

        x = self.post(x)            # [B,256,W]
        pooled = self.pool(x)       # [B,256]
        pen = self.proj(pooled)     # [B,256] (penultimate embedding)
        return pen

# ------------------------------
# Cosine classifier (for FREL)
# ------------------------------
class CosineClassifier(nn.Module):
     """
    Cosine-similarity classifier with learnable temperature.
    Commonly used when embedding geometry is important.
    """
    def __init__(self, feat_dim: int, n_classes: int, temperature: float = 10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, feat_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.t = nn.Parameter(torch.tensor(float(temperature)))  # learnable temp

    def forward(self, z):  # z: [B, D]
        z = F.normalize(z, dim=1)
        w = F.normalize(self.weight, dim=1)
        logits = self.t * (z @ w.t())  # [B, C]
        return logits

# ------------------------------
# Projection head for SupCon
# ------------------------------
class SupConHead(nn.Module):
    """
    Projection head used for supervised contrastive learning.
    """
    def __init__(self, in_dim: int = 256, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, proj_dim)
        )

    def forward(self, z):
        z = self.net(z)
        return F.normalize(z, dim=1)

# ------------------------------
# Wrappers
# ------------------------------
import math

class FRELNet(nn.Module):
    """
    For pretraining (CE + SupCon) and adaptation (cosine head).
    - backbone → 256-d embedding
    - head_cos → cosine classifier
    - head_proj → projection for SupCon (train-time only)
    """
    def __init__(self, in_ch: int, n_classes: int, proj_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.backbone = CNN1DBackbone(in_ch=in_ch, dropout=dropout)
        self.head_cos = CosineClassifier(feat_dim=256, n_classes=n_classes)
        self.head_proj = SupConHead(in_dim=256, proj_dim=proj_dim)

    def forward(self, x, return_proj: bool = False):
        pen = self.backbone(x)               # [B,256]
        logits = self.head_cos(pen)          # [B,C]
        if return_proj:
            z = self.head_proj(pen)          # [B,proj_dim]
            return logits, pen, z
        return logits, pen

class CNNClassifier(nn.Module):
    """
    Baseline CNN with linear classifier.
    """
    def __init__(self, in_ch: int, n_classes: int, dropout: float = 0.15):
        super().__init__()
        self.backbone = CNN1DBackbone(in_ch=in_ch, dropout=dropout)
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x):
        pen = self.backbone(x)
        logits = self.classifier(pen)
        return logits, pen
