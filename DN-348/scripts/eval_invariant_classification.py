#!/usr/bin/env python3
"""
Final Frozen Linear Invariant Evaluation for DN-348 SIE.
Evaluates the learned invariant representations using a linear vehicle-identity classification probe.
Follows the exact VeRi/SIE evaluation protocol:
- Frozen model (model.eval(), no backward pass to model)
- 200 training identities (disjoint from test identities)
- 80/20 train/validation split per identity with seed=42 (15,984 train, 4,000 val)
- Image resolution 224x224 (and 256x256), matching VeRi eval_classification.py
- Linear probe: nn.Linear(feature_dim, 200)
- Optimizer: Adam(lr=1e-3, weight_decay=0.0) for 50 epochs
- Tracks validation Top-1 and Top-5 accuracy

Evaluates:
1. Normalized 1024-D invariant projector output (Primary retrieval feature)
2. Raw (unnormalized) 1024-D invariant projector output
3. 256-D invariant backbone slice (matching VeRi --inv-part)
4. 512-D full ResNet-18 backbone
"""

import argparse
import csv
import json
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
DN_ROOT = ROOT / "DN348_SIE"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DN_ROOT) not in sys.path:
    sys.path.insert(0, str(DN_ROOT))

import src.models as m


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

        vehicle_dict = defaultdict(list)
        with open(day_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vid = line.split("/")[0]
                vehicle_dict[vid].append(self.day_root / line)

        with open(night_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vid = line.split("/")[0]
                vehicle_dict[vid].append(self.night_root / line)

        self.vehicle_ids = sorted(vehicle_dict.keys())
        assert len(self.vehicle_ids) == 200, f"Expected 200 training IDs, got {len(self.vehicle_ids)}"
        self.label_map = {vid: idx for idx, vid in enumerate(self.vehicle_ids)}

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


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


def train_probe_on_features(
    train_feats,
    train_targets,
    val_feats,
    val_targets,
    num_classes=200,
    epochs=50,
    batch_size=64,
    lr=1e-3,
    wd=0.0,
    device="cuda",
    desc="Probe",
    log_csv_path=None,
):
    feature_dim = train_feats.shape[1]
    probe = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=wd)

    train_ds = TensorDataset(train_feats, train_targets)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)

    val_feats_dev = val_feats.to(device)
    val_targets_cpu = val_targets.cpu()

    csv_writer = None
    csv_file = None
    if log_csv_path is not None:
        csv_file = open(log_csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Epoch", "Train_Loss", "Train_Top1", "Train_Top5", "Val_Top1", "Val_Top5"])

    best_val_top1 = 0.0
    best_val_top5 = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        probe.train()
        train_loss_sum = 0.0
        train_top1_sum = 0.0
        train_top5_sum = 0.0
        n_train_batches = 0

        for feats_b, labels_b in train_loader:
            feats_b = feats_b.to(device, non_blocking=True)
            labels_b = labels_b.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = probe(feats_b)
            loss = F.cross_entropy(logits, labels_b)
            top1, top5 = accuracy(logits, labels_b, topk=(1, 5))

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_top1_sum += top1.item()
            train_top5_sum += top5.item()
            n_train_batches += 1

        avg_loss = train_loss_sum / n_train_batches
        avg_train_top1 = (train_top1_sum / n_train_batches) * 100.0
        avg_train_top5 = (train_top5_sum / n_train_batches) * 100.0

        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_feats_dev)
            val_top1, val_top5 = accuracy(val_logits.cpu(), val_targets_cpu, topk=(1, 5))
            val_top1_pct = val_top1.item() * 100.0
            val_top5_pct = val_top5.item() * 100.0

        if val_top1_pct > best_val_top1:
            best_val_top1 = val_top1_pct
            best_val_top5 = val_top5_pct
            best_epoch = epoch

        if csv_writer is not None:
            csv_writer.writerow([
                epoch,
                f"{avg_loss:.6f}",
                f"{avg_train_top1:.4f}",
                f"{avg_train_top5:.4f}",
                f"{val_top1_pct:.4f}",
                f"{val_top5_pct:.4f}",
            ])
            csv_file.flush()

        if epoch % 10 == 0 or epoch == epochs or epoch == 1:
            print(
                f"[{desc} Epoch {epoch:2d}/{epochs}] "
                f"Train Loss: {avg_loss:.4f} | "
                f"Train Top-1: {avg_train_top1:5.2f}% | "
                f"Val Top-1: {val_top1_pct:5.2f}% | "
                f"Val Top-5: {val_top5_pct:5.2f}%"
            )

    if csv_file is not None:
        csv_file.close()

    return {
        "feature_dim": feature_dim,
        "best_epoch": best_epoch,
        "best_val_top1": float(best_val_top1),
        "best_val_top5": float(best_val_top5),
        "final_val_top1": float(val_top1_pct),
        "final_val_top5": float(val_top5_pct),
    }


def main():
    parser = argparse.ArgumentParser(description="DN-348 Final Invariant Linear Probe")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint")
    parser.add_argument("--dataset-root", type=Path, default=Path("./data/dn348"))
    parser.add_argument("--exp-dir", type=Path, required=True, help="Directory to save evaluation artifacts")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DN-348 FINAL FROZEN INVARIANT CLASSIFICATION EVALUATION")
    print("=" * 80)
    print(f"Checkpoint       : {args.checkpoint}")
    print(f"Dataset Root     : {args.dataset_root}")
    print(f"Resolution       : {args.resolution}x{args.resolution}")
    print(f"Probe Epochs     : {args.epochs}")
    print(f"Device           : {device}")
    print(f"Output Directory : {args.exp_dir}")

    # Build model and load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_state = ckpt["model"] if "model" in ckpt else ckpt
    clean_state = {k[7:] if k.startswith("module.") else k: v for k, v in model_state.items()}

    class Config:
        arch = "resnet18"
        equi = 256
        mlp = "2048-2048-2048"
        predictor_type = "hypernetwork"
        predictor = ""
        hypernetwork = "linear"
        trans_backbone = "1024-512-64"
        freeze_backbone = True
        detach_trans_input = True
        bias_hypernet = False
        bias_pred = False
        predictor_relu = False

    cfg = Config()
    if "args" in ckpt and isinstance(ckpt["args"], dict):
        for k, v in ckpt["args"].items():
            setattr(cfg, k.replace("-", "_"), v)

    model = m.SIEDayNightEncoder(cfg, num_classes=348)
    msg = model.load_state_dict(clean_state, strict=False)
    print("Loaded model state with msg:", msg)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)

    # Transforms (matching VeRi eval_classification.py)
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

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"\nTrain set: {len(ds_train)} images across 200 identities.")
    print(f"Val set  : {len(ds_val)} images across 200 identities.")

    # Feature extraction function
    def extract_all_representations(loader, desc="Extracting"):
        feats_norm_proj = []
        feats_raw_proj = []
        feats_inv_backbone = []
        feats_full_backbone = []
        targets = []

        with torch.no_grad():
            for imgs, lbls in tqdm(loader, desc=desc, leave=False):
                imgs = imgs.to(device, non_blocking=True)
                embs = model.backbone(imgs)                     # [B, 512]
                inv_raw = embs[..., :model.inv_repr_size]       # [B, 256]
                proj_raw = model.projector_inv(inv_raw)         # [B, 1024]
                proj_norm = F.normalize(proj_raw, dim=-1)       # [B, 1024]

                feats_norm_proj.append(proj_norm.cpu())
                feats_raw_proj.append(proj_raw.cpu())
                feats_inv_backbone.append(inv_raw.cpu())
                feats_full_backbone.append(embs.cpu())
                targets.append(lbls)

        return (
            torch.cat(feats_norm_proj, dim=0),
            torch.cat(feats_raw_proj, dim=0),
            torch.cat(feats_inv_backbone, dim=0),
            torch.cat(feats_full_backbone, dim=0),
            torch.cat(targets, dim=0),
        )

    print("\nExtracting frozen features for train set...")
    tr_norm_proj, tr_raw_proj, tr_inv_bb, tr_full_bb, tr_targets = extract_all_representations(train_loader, "Train")
    print("Extracting frozen features for val set...")
    va_norm_proj, va_raw_proj, va_inv_bb, va_full_bb, va_targets = extract_all_representations(val_loader, "Val")

    results = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(ckpt.get("step", -1)),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "num_classes": 200,
        "num_train_images": len(ds_train),
        "num_val_images": len(ds_val),
        "resolution": f"{args.resolution}x{args.resolution}",
        "epochs": args.epochs,
        "features": {},
    }

    feature_configs = [
        ("norm_proj_inv", "Normalized 1024-D Invariant Projector (Primary Retrieval Feature)", tr_norm_proj, va_norm_proj),
        ("raw_proj_inv", "Raw 1024-D Invariant Projector", tr_raw_proj, va_raw_proj),
        ("inv_backbone_256", "256-D Invariant Backbone Slice (VeRi --inv-part equivalent)", tr_inv_bb, va_inv_bb),
        ("full_backbone_512", "512-D Full Backbone", tr_full_bb, va_full_bb),
    ]

    for feat_key, feat_name, tr_f, va_f in feature_configs:
        print("\n" + "-" * 70)
        print(f"Training Linear Probe on: {feat_name} ({tr_f.shape[1]}-D)")
        print("-" * 70)
        csv_log = args.exp_dir / f"metrics_{feat_key}.csv"
        metrics = train_probe_on_features(
            tr_f,
            tr_targets,
            va_f,
            va_targets,
            num_classes=200,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            wd=args.wd,
            device=device,
            desc=feat_key,
            log_csv_path=csv_log,
        )
        metrics["feature_name"] = feat_name
        results["features"][feat_key] = metrics
        print(
            f"--> Best Val Top-1: {metrics['best_val_top1']:.2f}% | "
            f"Best Val Top-5: {metrics['best_val_top5']:.2f}% (Epoch {metrics['best_epoch']}) | "
            f"Final Val Top-1: {metrics['final_val_top1']:.2f}%"
        )

    summary_file = args.exp_dir / "invariant_classification_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved consolidated invariant classification summary to: {summary_file}")


if __name__ == "__main__":
    main()
