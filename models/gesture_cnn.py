"""
=============================================================
  GESTURE CNN MODEL  —  built entirely from scratch with PyTorch
=============================================================
  Architecture: Custom lightweight CNN
    Input  : 64×64 RGB image
    Output : 6 gesture classes

  Blocks:
    ConvBlock × 4  (Conv → BN → ReLU → MaxPool → Dropout)
    Classifier     (Flatten → FC → BN → ReLU → Dropout → FC)
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Gesture label map ────────────────────────────────────────────────────────
GESTURE_CLASSES = {
    0: "idle",
    1: "play_pause",
    2: "next_track",
    3: "prev_track",
    4: "volume_up",
    5: "volume_down",
}

NUM_CLASSES = len(GESTURE_CLASSES)
IMG_SIZE    = 64   # square input


# ── Building block ───────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool → Dropout2d"""

    def __init__(self, in_ch, out_ch, pool=True, dropout=0.25):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))   # halves spatial dims
        layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ── Main model ───────────────────────────────────────────────────────────────
class GestureCNN(nn.Module):
    """
    Lightweight custom CNN for real-time gesture recognition.

    Input  : (B, 3, 64, 64)
    Output : (B, NUM_CLASSES)  — raw logits (use CrossEntropyLoss)
    """

    def __init__(self, num_classes=NUM_CLASSES, dropout_fc=0.5):
        super().__init__()

        # ── Feature extractor ────────────────────────────────────
        # 64×64 → 32×32 → 16×16 → 8×8 → 4×4
        self.features = nn.Sequential(
            ConvBlock(3,   32, pool=True,  dropout=0.10),   # 64→32
            ConvBlock(32,  64, pool=True,  dropout=0.15),   # 32→16
            ConvBlock(64, 128, pool=True,  dropout=0.20),   # 16→8
            ConvBlock(128, 256, pool=True, dropout=0.25),   # 8→4
        )

        # Global average pool → remove spatial dims entirely
        self.gap = nn.AdaptiveAvgPool2d(1)   # (B, 256, 4, 4) → (B, 256, 1, 1)

        # ── Classifier ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),                          # (B, 256)
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(128, num_classes),
        )

        # Weight init
        self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

    def predict(self, x):
        """Return (class_idx, confidence) for a single preprocessed tensor."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = F.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        return idx.item(), conf.item()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


# ── Quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    model = GestureCNN()
    dummy = torch.randn(8, 3, IMG_SIZE, IMG_SIZE)
    out   = model(dummy)
    print(f"Model output shape : {out.shape}")   # (8, 6)

    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params       : {total:,}")
    print(f"Trainable params   : {train:,}")
    print("\nModel architecture:")
    print(model)
