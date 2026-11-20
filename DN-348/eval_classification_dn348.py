import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(".")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "DN348_SIE") not in sys.path:
    sys.path.insert(0, str(ROOT / "DN348_SIE"))

import src.resnet as resnet


class DatasetDN348Linear(Dataset):
    """
    Linear evaluation dataset for DN-348.
    Uses ONLY the 200 training identities (disjoint from test set).
    Splits images of each identity into 80% train and 20% validation.
    """

    def __init__(
        self,
        dataset_root="./data/dn348",
        split="train",
        transform=None,
        train_ratio=0.8,
        seed=42,
    ):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transform = transform

        self.split_root = self.dataset_root / "train_test_split"
        self.day_root = self.dataset_root / "day"
        self.night_root = self.dataset_root / "night"

        day_file = self.split_root / "train_list_day.txt"
        night_file = self.split_root / "train_list_night.txt"

        if not day_file.exists() or not night_file.exists():
            raise FileNotFoundError(f"Missing train split files in {self.split_root}")

        print(f"\nInitializing DatasetDN348Linear ({split})...")

        vehicle_dict = defaultdict(list)
        day_count = 0
        with open(day_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vid = line.split("/")[0]
                vehicle_dict[vid].append(self.day_root / line)
                day_count += 1

        night_count = 0
        with open(night_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vid = line.split("/")[0]
                vehicle_dict[vid].append(self.night_root / line)
                night_count += 1

        self.vehicle_ids = sorted(vehicle_dict.keys())
        assert len(self.vehicle_ids) == 200, f"Expected 200 training IDs, got {len(self.vehicle_ids)}"
        self.label_map = {vid: idx for idx, vid in enumerate(self.vehicle_ids)}

        # Deterministic per-identity 80/20 train/val split with seed 42
        rng = random.Random(seed)
        self.samples = []

        for vid in self.vehicle_ids:
            images = vehicle_dict[vid][:]
            rng.shuffle(images)
            split_idx = max(1, int(len(images) * train_ratio))

            if split == "train":
                selected = images[:split_idx]
            else:
                selected = images[split_idx:]
                if len(selected) == 0:
                    selected = images[-1:]

            label = self.label_map[vid]
            for img_path in selected:
                self.samples.append((img_path, label))

        print(f"Split : {split}")
        print(f"Total Images   : {len(self.samples)}")
        print(f"Total Vehicles : {len(self.vehicle_ids)}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class LinearClassifierModel(nn.Module):
    """
    Evaluates frozen 512-D ResNet-18 backbone representations
    using a linear classification probe (nn.Linear(512, num_classes)).
    """

    def __init__(self, arch="resnet18", num_classes=200):
        super().__init__()
        self.backbone, _ = resnet.__dict__[arch](zero_init_residual=True)
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        logits = self.head(features)
        return logits


def accuracy(output, target, topk=(1, 5)):
    """Computes Top-k accuracy over specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def main():
    parser = argparse.ArgumentParser(
        description="DN-348 Linear Identity Evaluation (VeRi-Equivalent Protocol)"
    )
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--weights-file", type=Path, required=True, help="Path to checkpoint (model_best.pth or model.pth)")
    parser.add_argument("--dataset-root", type=Path, default=Path("./data/dn348"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=224, help="Image resolution: 224x224 exactly matching VeRi eval_classification.py")
    parser.add_argument("--exp-dir", type=Path, default=Path("eval/dn348_classification"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DN-348 LINEAR IDENTITY EVALUATION")
    print("=" * 70)
    print("Backbone            : ResNet-18 (Frozen)")
    print("Feature Dim         : 512-D")
    print("Checkpoint          :", args.weights_file)
    print("Training Identities : 200 (Disjoint from 148 test IDs)")
    print("Split Logic         : 80/20 per-identity train/val split (seed=42)")
    print("Image Preprocessing :", f"{args.resolution}x{args.resolution} (matching VeRi classification evaluator)")
    print("Device              :", device)
    print("Output Directory    :", args.exp_dir)

    # Save params
    params_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(args.exp_dir / "params.json", "w") as f:
        json.dump(params_dict, f, indent=4)

    # Data Transforms (224x224 matching VeRi classification evaluator)
    normalize = transforms.Normalize(
        mean=[0.5016, 0.5037, 0.5060],
        std=[0.1030, 0.0999, 0.0969],
    )
    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        normalize,
    ])

    ds_train = DatasetDN348Linear(
        dataset_root=args.dataset_root,
        split="train",
        transform=transform,
        train_ratio=0.8,
        seed=args.seed,
    )
    ds_val = DatasetDN348Linear(
        dataset_root=args.dataset_root,
        split="val",
        transform=transform,
        train_ratio=0.8,
        seed=args.seed,
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model Setup
    model = LinearClassifierModel(arch=args.arch, num_classes=args.num_classes)

    # Load frozen backbone weights
    print(f"\nLoading backbone weights from: {args.weights_file}")
    ckpt = torch.load(args.weights_file, map_location="cpu")
    model_state = ckpt["model"] if "model" in ckpt else ckpt

    backbone_state = {}
    for k, v in model_state.items():
        if k.startswith("module.backbone."):
            backbone_state[k[len("module.backbone."):]] = v
        elif k.startswith("backbone."):
            backbone_state[k[len("backbone."):]] = v

    load_msg = model.backbone.load_state_dict(backbone_state, strict=True)
    print("Backbone loaded successfully with msg:", load_msg)

    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.backbone.eval()

    model = model.to(device)

    # Pre-extract 512-D frozen backbone representations
    # (Since the backbone is frozen and evaluation transform is deterministic,
    # caching features achieves 100% mathematical equivalence with ~40x speedup)
    print("\nPre-extracting 512-D frozen backbone representations...")
    model.backbone.eval()

    def extract_features(loader, desc="Extracting"):
        all_feats = []
        all_targets = []
        with torch.no_grad():
            for imgs, lbls in tqdm(loader, desc=desc, leave=False):
                imgs = imgs.to(device, non_blocking=True)
                feats = model.backbone(imgs)
                all_feats.append(feats.cpu())
                all_targets.append(lbls)
        return torch.cat(all_feats, dim=0), torch.cat(all_targets, dim=0)

    train_feats, train_targets = extract_features(train_loader, desc="Train features")
    val_feats, val_targets = extract_features(val_loader, desc="Val features")
    print(f"Extracted Train Features : {train_feats.shape} ({train_feats.element_size() * train_feats.nelement() / 1e6:.1f} MB)")
    print(f"Extracted Val Features   : {val_feats.shape} ({val_feats.element_size() * val_feats.nelement() / 1e6:.1f} MB)")

    train_feature_ds = TensorDataset(train_feats, train_targets)
    train_feature_loader = DataLoader(
        train_feature_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    # Optimizer (only trains the linear probe head)
    optimizer = torch.optim.Adam(
        model.head.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
    )

    # CSV Logging
    metrics_file = open(args.exp_dir / "metrics.csv", "w", newline="")
    csv_writer = csv.writer(metrics_file)
    csv_writer.writerow([
        "Epoch",
        "Train_Loss",
        "Train_Top1",
        "Train_Top5",
        "Val_Top1",
        "Val_Top5",
    ])

    best_val_top1 = 0.0
    best_val_top5 = 0.0
    best_epoch = 0

    print(f"\nStarting Linear Probe Training ({args.epochs} epochs)...")
    for epoch in range(args.epochs):
        model.head.train()
        train_loss_sum = 0.0
        train_top1_sum = 0.0
        train_top5_sum = 0.0
        n_train_batches = 0

        for feats, labels in train_feature_loader:
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model.head(feats)
            loss = F.cross_entropy(outputs, labels)
            top1, top5 = accuracy(outputs, labels, topk=(1, 5))

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_top1_sum += top1.item()
            train_top5_sum += top5.item()
            n_train_batches += 1

        avg_train_loss = train_loss_sum / n_train_batches
        avg_train_top1 = (train_top1_sum / n_train_batches) * 100.0
        avg_train_top5 = (train_top5_sum / n_train_batches) * 100.0

        # Global Validation Top-1 and Top-5 evaluation
        model.head.eval()
        with torch.no_grad():
            val_outputs = model.head(val_feats.to(device))
            val_top1, val_top5 = accuracy(val_outputs.cpu(), val_targets, topk=(1, 5))
            val_top1_pct = val_top1.item() * 100.0
            val_top5_pct = val_top5.item() * 100.0

        if val_top1_pct > best_val_top1:
            best_val_top1 = val_top1_pct
            best_val_top5 = val_top5_pct
            best_epoch = epoch + 1
            torch.save(
                {
                    "epoch": epoch + 1,
                    "head_state_dict": model.head.state_dict(),
                    "val_top1": val_top1_pct,
                    "val_top5": val_top5_pct,
                },
                args.exp_dir / "best_head.pth",
            )

        print(
            f"[Epoch {epoch + 1:2d}/{args.epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Top-1: {avg_train_top1:5.2f}% | "
            f"Val Top-1: {val_top1_pct:5.2f}% | "
            f"Val Top-5: {val_top5_pct:5.2f}%"
        )

        csv_writer.writerow([
            epoch + 1,
            f"{avg_train_loss:.6f}",
            f"{avg_train_top1:.4f}",
            f"{avg_train_top5:.4f}",
            f"{val_top1_pct:.4f}",
            f"{val_top5_pct:.4f}",
        ])
        metrics_file.flush()

    metrics_file.close()

    print("\n" + "=" * 70)
    print("FINAL LINEAR PROBE RESULTS")
    print("=" * 70)
    print(f"Checkpoint            : {args.weights_file}")
    print(f"Checkpoint Epoch      : {ckpt.get('epoch', -1)}")
    print(f"Number of Identities  : {args.num_classes}")
    print(f"Probe Training Samples: {len(ds_train)}")
    print(f"Probe Val Samples     : {len(ds_val)}")
    print(f"Probe Epochs          : {args.epochs}")
    print(f"Best Validation Top-1 : {best_val_top1:.2f}% (at epoch {best_epoch})")
    print(f"Best Validation Top-5 : {best_val_top5:.2f}%")
    print(f"Final Validation Top-1: {val_top1_pct:.2f}%")
    print(f"Final Validation Top-5: {val_top5_pct:.2f}%")
    print("=" * 70 + "\n")

    summary_results = {
        "checkpoint": str(args.weights_file),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_type": str(ckpt.get("checkpoint_type", "N/A")),
        "num_classes": args.num_classes,
        "num_train_samples": len(ds_train),
        "num_val_samples": len(ds_val),
        "resolution": f"{args.resolution}x{args.resolution}",
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_top1": float(best_val_top1),
        "best_val_top5": float(best_val_top5),
        "final_val_top1": float(val_top1_pct),
        "final_val_top5": float(val_top5_pct),
    }

    with open(args.exp_dir / "classification_summary.json", "w") as jf:
        json.dump(summary_results, jf, indent=4)
    print(f"Summary saved to: {args.exp_dir / 'classification_summary.json'}")


if __name__ == "__main__":
    main()
