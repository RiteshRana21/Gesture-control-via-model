"""
=============================================================
  TRAINING PIPELINE
  Trains GestureCNN from scratch on your collected dataset.
=============================================================
  USAGE:
    python train.py

  OUTPUT:
    models/gesture_cnn_best.pth   — best validation checkpoint
    models/gesture_cnn_final.pth  — final epoch checkpoint
    models/training_history.json  — loss / accuracy per epoch
=============================================================
"""

import os
import sys
import json
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Allow importing from sibling dirs
#sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.gesture_cnn import GestureCNN, GESTURE_CLASSES

# ── Config ───────────────────────────────────────────────────────────────────
#DATASET_DIR  = os.path.join(os.path.dirname(__file__), "..", "dataset")
DATASET_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
#MODEL_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
BEST_MODEL   = os.path.join(MODEL_DIR, "gesture_cnn_best.pth")
FINAL_MODEL  = os.path.join(MODEL_DIR, "gesture_cnn_final.pth")
HISTORY_FILE = os.path.join(MODEL_DIR, "training_history.json")

IMG_SIZE     = 64
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.15
PATIENCE     = 8      # early stopping patience
SEED         = 42


# ── Transforms ───────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[CPU] No GPU found — training on CPU (slower)")
    return dev


def load_datasets():
    full_dataset = datasets.ImageFolder(DATASET_DIR, transform=train_transform)
    print(f"\n[Dataset] {len(full_dataset)} total images")
    print(f"[Classes] {full_dataset.classes}")

    # Verify all expected classes are present
    missing = [c for c in GESTURE_CLASSES.values() if c not in full_dataset.classes]
    if missing:
        print(f"\n[WARNING] Missing classes: {missing}")
        print("Run collect_data.py first to gather images for all gestures!\n")

    val_size   = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    torch.manual_seed(SEED)
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    # Apply different transform to validation subset
    val_set.dataset = copy.deepcopy(full_dataset)
    val_set.dataset.transform = val_transform

    print(f"[Split]   Train: {train_size}  |  Val: {val_size}")
    return train_set, val_set, full_dataset.classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


# ── Main training loop ───────────────────────────────────────────────────────
def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = get_device()

    train_set, val_set, class_names = load_datasets()

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    model     = GestureCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_weights = None
    patience_ctr = 0
    history      = []

    print(f"\n{'='*60}")
    print(f"  Training GestureCNN | {EPOCHS} epochs | batch {BATCH_SIZE}")
    print(f"{'='*60}\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer, device)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(f"Epoch [{epoch:02d}/{EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f}  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}  |  "
              f"LR: {lr_now:.2e}  ({elapsed:.1f}s)")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc":  round(train_acc,  5),
            "val_loss":   round(val_loss,   5),
            "val_acc":    round(val_acc,    5),
        })

        # ── Save best ────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({
                "epoch":       epoch,
                "model_state": best_weights,
                "val_acc":     best_val_acc,
                "class_names": class_names,
            }, BEST_MODEL)
            print(f"  ✓ New best saved ({best_val_acc:.3f})")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n[Early Stop] No improvement for {PATIENCE} epochs.")
                break

    # ── Save final ───────────────────────────────────────────────
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "val_acc":     val_acc,
        "class_names": class_names,
    }, FINAL_MODEL)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best Val Accuracy : {best_val_acc:.4f}")
    print(f"  Best model saved  : {BEST_MODEL}")
    print(f"  History saved     : {HISTORY_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
