#!/usr/bin/env python3
"""
Identity Classification Linear Probe for DN-348 ResNet-18 Baseline.
Evaluates the 1024-D L2-normalized embedding using the EXACT SAME protocol as SIE:
- 200 training identities
- 80/20 per-identity train/val split (seed=42)
- Pre-extracts frozen 1024-D normalized embeddings
- Trains a linear probe (Linear(1024, 200)) for 50 epochs using Adam (lr=1e-3, batch_size=64)
- Reports Best and Final Validation Top-1 and Top-5 accuracy
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(".")
DN_ROOT = ROOT / "DN348_SIE"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DN_ROOT) not in sys.path:
    sys.path.insert(0, str(DN_ROOT))

from src.resnet18_baseline import ResNet18IdentityBaseline
from eval_classification_dn348 import DatasetDN348Linear


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def main():
    parser = argparse.ArgumentParser(description="DN-348 ResNet-18 Baseline Identity Linear Probe")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained checkpoint (model.pth or model_best.pth)")
    parser.add_argument("--dataset-root", type=Path, default=Path("./data/dn348"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=224, help="Matching VeRi/SIE evaluation resolution")
    parser.add_argument("--exp-dir", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.exp_dir is None:
        args.exp_dir = args.checkpoint.parent / "eval_classification"
    args.exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DN-348 RESNET-18 BASELINE LINEAR IDENTITY PROBE")
    print("=" * 70)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Epochs     : {args.epochs}")
    print(f"Device     : {device}")
    print(f"Output Dir : {args.exp_dir}")

    # Load trained model
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_state = ckpt["model"] if "model" in ckpt else ckpt
    clean_state = {k[7:] if k.startswith("module.") else k: v for k, v in model_state.items()}

    model = ResNet18IdentityBaseline(num_classes=200, embedding_dim=1024, pretrained=False)
    msg = model.load_state_dict(clean_state, strict=True)
    print(f"Loaded checkpoint with msg: {msg}")
    model.eval()
    model.to(device)

    # 224x224 transform matching SIE probe evaluation
    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5016, 0.5037, 0.5060], std=[0.1030, 0.0999, 0.0969]),
    ])

    train_ds = DatasetDN348Linear(args.dataset_root, split="train", transform=transform, train_ratio=0.8, seed=args.seed)
    val_ds = DatasetDN348Linear(args.dataset_root, split="val", transform=transform, train_ratio=0.8, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Extract 1024-D normalized representations
    print("\nExtracting 1024-D L2-normalized embeddings for probe training...")
    def extract(loader, desc):
        feats, targets = [], []
        with torch.no_grad():
            for imgs, lbls in tqdm(loader, desc=desc, leave=False):
                imgs = imgs.to(device, non_blocking=True)
                f = model.extract_features(imgs)
                feats.append(f.cpu())
                targets.append(lbls)
        return torch.cat(feats, dim=0), torch.cat(targets, dim=0)

    train_feats, train_targets = extract(train_loader, "Train features")
    val_feats, val_targets = extract(val_loader, "Val features")
    print(f"Extracted Train Features : {train_feats.shape}")
    print(f"Extracted Val Features   : {val_feats.shape}")

    train_tensor_ds = TensorDataset(train_feats, train_targets)
    val_tensor_ds = TensorDataset(val_feats, val_targets)

    train_tensor_loader = DataLoader(train_tensor_ds, batch_size=args.batch_size, shuffle=True)
    val_tensor_loader = DataLoader(val_tensor_ds, batch_size=args.batch_size, shuffle=False)

    # Linear probe classifier
    probe = nn.Linear(1024, 200).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_top1 = 0.0
    best_top5 = 0.0
    best_epoch = 0

    print("\nTraining Linear Probe for 50 epochs...")
    for epoch in range(args.epochs):
        probe.train()
        train_loss = 0.0
        train_top1 = 0.0
        n_train = 0

        for x, y in train_tensor_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = probe(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            acc1 = accuracy(logits, y, topk=(1,))[0]
            train_loss += loss.item() * x.size(0)
            train_top1 += acc1 * x.size(0)
            n_train += x.size(0)

        probe.eval()
        val_top1 = 0.0
        val_top5 = 0.0
        n_val = 0
        with torch.no_grad():
            for x, y in val_tensor_loader:
                x, y = x.to(device), y.to(device)
                logits = probe(x)
                acc1, acc5 = accuracy(logits, y, topk=(1, 5))
                val_top1 += acc1 * x.size(0)
                val_top5 += acc5 * x.size(0)
                n_val += x.size(0)

        ep_train_loss = train_loss / n_train
        ep_train_top1 = train_top1 / n_train
        ep_val_top1 = val_top1 / n_val
        ep_val_top5 = val_top5 / n_val

        if ep_val_top1 > best_top1:
            best_top1 = ep_val_top1
            best_top5 = ep_val_top5
            best_epoch = epoch + 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
            print(f"[Epoch {epoch+1:2d}/{args.epochs}] Train Loss: {ep_train_loss:.4f} | Train Top-1: {ep_train_top1:.2f}% | Val Top-1: {ep_val_top1:.2f}% | Val Top-5: {ep_val_top5:.2f}%")

    print("\n" + "=" * 70)
    print("FINAL LINEAR PROBE RESULTS")
    print("=" * 70)
    print(f"Best Validation Top-1 : {best_top1:.2f}% (at epoch {best_epoch})")
    print(f"Best Validation Top-5 : {best_top5:.2f}%")
    print(f"Final Validation Top-1: {ep_val_top1:.2f}%")
    print(f"Final Validation Top-5: {ep_val_top5:.2f}%")
    print("=" * 70)

    results = {
        "checkpoint": str(args.checkpoint),
        "best_top1": best_top1,
        "best_top5": best_top5,
        "best_epoch": best_epoch,
        "final_top1": ep_val_top1,
        "final_top5": ep_val_top5,
    }
    with open(args.exp_dir / "classification_summary.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved probe summary to: {args.exp_dir / 'classification_summary.json'}\n")


if __name__ == "__main__":
    main()
