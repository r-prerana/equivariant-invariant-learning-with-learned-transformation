#!/usr/bin/env python3
"""
Training Script for DN-348 Standard ResNet-18 Identity Retrieval Baseline.
Clean non-SIE diagnostic baseline:
- ImageNet-pretrained ResNet-18 backbone
- 1024-D embedding projection
- 200-class identity classification cross-entropy loss
- 200 epochs, batch size 64
- Zero SIE losses, zero transformation heads, zero Day/Night target labels
- Evaluated via exact same retrieval evaluator (eval_map_cmc_dn348.py)
"""

import argparse
import csv
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT = Path(".")
DN_ROOT = ROOT / "DN348_SIE"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DN_ROOT) not in sys.path:
    sys.path.insert(0, str(DN_ROOT))

from src.resnet18_baseline import ResNet18IdentityBaseline


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


class DatasetDN348Classification(Dataset):
    """
    Standard vehicle identity classification dataset for DN-348.
    Loads all training images (Day + Night) from the official train splits.
    Returns: (image, vehicle_label) where vehicle_label in [0, 199].
    """

    def __init__(self, dataset_root, transform=None):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        self.split_root = self.dataset_root / "train_test_split"
        self.day_root = self.dataset_root / "day"
        self.night_root = self.dataset_root / "night"

        day_file = self.split_root / "train_list_day.txt"
        night_file = self.split_root / "train_list_night.txt"

        if not day_file.exists() or not night_file.exists():
            raise FileNotFoundError(f"Missing train split files in {self.split_root}")

        self.samples = []
        vehicle_ids = set()

        with open(day_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vid = line.split("/")[0]
                vehicle_ids.add(vid)
                self.samples.append((self.day_root / line, vid))

        with open(night_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vid = line.split("/")[0]
                vehicle_ids.add(vid)
                self.samples.append((self.night_root / line, vid))

        self.vehicle_ids = sorted(vehicle_ids)
        assert len(self.vehicle_ids) == 200, f"Expected 200 training IDs, got {len(self.vehicle_ids)}"
        self.label_map = {vid: idx for idx, vid in enumerate(self.vehicle_ids)}

        print(f"\nInitializing DatasetDN348Classification...")
        print(f"  Training Vehicles : {len(self.vehicle_ids)}")
        print(f"  Total Images      : {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, vid = self.samples[idx]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.label_map[vid]
        return img, torch.tensor(label, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description="DN-348 ResNet-18 Identity Baseline Training")
    parser.add_argument("--dataset-root", type=Path, default=Path("./data/dn348"))
    parser.add_argument("--exp-dir", type=Path, default=Path("checkpoints/dn348_resnet18_identity_baseline"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-lr", type=float, default=1e-3, help="LR for embedding projector and classifier")
    parser.add_argument("--backbone-lr", type=float, default=1e-4, help="Differential LR for pretrained ResNet18 backbone")
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Verify directory isolation
    args.exp_dir = DN_ROOT / args.exp_dir
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    for protected in ["dn348_v1", "dn348_sie_oracle", "dn348_sie_unfrozen", "dn348_sie_trans20", "dn348_sie_identity"]:
        assert not str(args.exp_dir).endswith(protected), f"Cannot overwrite protected directory: {protected}"

    # Mirror stdout and stderr to training.log
    log_file = open(args.exp_dir / "training.log", "a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("DN-348 STANDARD RESNET-18 IDENTITY RETRIEVAL BASELINE")
    print("=" * 70)
    print("Architecture        : ResNet-18 (ImageNet Pretrained) + 1024-D Embedding")
    print("Supervision         : Standard Cross-Entropy on 200 Vehicle Identities")
    print("Epochs              :", args.epochs)
    print("Batch Size          :", args.batch_size)
    print("Backbone LR         :", args.backbone_lr)
    print("Heads LR            :", args.base_lr)
    print("Output Directory    :", args.exp_dir)
    print("Device              :", device)

    # Save configs
    dict_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    dict_args["experience"] = "ResNet18IdentityBaseline"
    dict_args["model_type"] = "resnet18_baseline"
    with open(args.exp_dir / "params.json", "w") as f:
        json.dump(dict_args, f, indent=2)
    with open(args.exp_dir / "config.txt", "w") as f:
        f.write("CMD: " + " ".join(sys.argv) + "\n\n")
        for k, v in sorted(dict_args.items()):
            f.write(f"{k}: {v}\n")

    # CSV metrics header
    csv_file = args.exp_dir / "metrics.csv"
    with open(csv_file, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["epoch", "train_loss", "train_acc", "lr_backbone", "lr_heads", "epoch_time_sec"])

    tb_dir = args.exp_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    # Transforms (exact match with SIE: 256x256, Normalize)
    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5016, 0.5037, 0.5060], std=[0.1030, 0.0999, 0.0969]),
    ])

    dataset = DatasetDN348Classification(dataset_root=args.dataset_root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Instantiate model
    model = ResNet18IdentityBaseline(num_classes=200, embedding_dim=1024, pretrained=True)
    model.to(device)

    # Differential learning rates
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.projector_inv.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": args.backbone_lr, "name": "backbone"},
        {"params": head_params, "lr": args.base_lr, "name": "heads"},
    ], weight_decay=args.wd)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    print("\nStarting 200-epoch training...")
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        for x, y in progress:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, _ = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total_samples += x.size(0)

            progress.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{100.0 * correct / total_samples:.2f}%"
            })

        scheduler.step()
        epoch_time = time.time() - epoch_start
        epoch_loss = total_loss / total_samples
        epoch_acc = 100.0 * correct / total_samples
        lr_bb = optimizer.param_groups[0]["lr"]
        lr_heads = optimizer.param_groups[1]["lr"]

        print(
            f"[Epoch {epoch+1}/{args.epochs}] "
            f"train_loss={epoch_loss:.4f} | "
            f"train_acc={epoch_acc:.2f}% | "
            f"lr_bb={lr_bb:.2e} | "
            f"lr_head={lr_heads:.2e} | "
            f"time={epoch_time:.1f}s"
        )

        # CSV logging
        with open(csv_file, "a", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                epoch + 1,
                f"{epoch_loss:.6f}",
                f"{epoch_acc:.4f}",
                f"{lr_bb:.6e}",
                f"{lr_heads:.6e}",
                f"{epoch_time:.2f}",
            ])

        # TensorBoard
        writer.add_scalar("Train/Loss", epoch_loss, epoch + 1)
        writer.add_scalar("Train/Accuracy", epoch_acc, epoch + 1)
        writer.add_scalar("Train/LR_Backbone", lr_bb, epoch + 1)
        writer.add_scalar("Train/LR_Heads", lr_heads, epoch + 1)
        writer.flush()

        # Checkpoint state
        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": dict_args,
            "best_acc": best_acc,
        }

        # Save latest model.pth
        torch.save(state, args.exp_dir / "model.pth")

        # Save model_best.pth
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_state = dict(state)
            best_state["best_acc"] = best_acc
            torch.save(best_state, args.exp_dir / "model_best.pth")
            print(f"  [BEST CHECKPOINT] Saved model_best.pth with train_acc = {best_acc:.2f}%")

        # Save periodic snapshot every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(state, args.exp_dir / f"checkpoint_epoch_{epoch + 1}.pth")

    # Save final weights (backbone)
    torch.save(model.backbone.state_dict(), args.exp_dir / "final_weights.pth")
    writer.close()
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print(f"Final Model Saved to: {args.exp_dir / 'model.pth'}")
    print(f"Best Model Saved to : {args.exp_dir / 'model_best.pth'} (Acc: {best_acc:.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
