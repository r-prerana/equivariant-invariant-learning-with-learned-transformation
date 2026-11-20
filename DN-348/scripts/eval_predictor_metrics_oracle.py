#!/usr/bin/env python3
"""
DN-348 Predictor Metrics Evaluation for SIECategoricalOracle Baseline
Measures Cosine Similarity and L2 Distance when the Hypernetwork receives
the ground-truth one-hot transition vector directly.

Matches the exact validation protocol of the proposed model (3,200 pairs from val_pairs.txt).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

ROOT = Path(".")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "DN348_SIE") not in sys.path:
    sys.path.insert(0, str(ROOT / "DN348_SIE"))

import src.models as m

CLASS_NAMES = [
    "DAY_TO_DAY",
    "NIGHT_TO_NIGHT",
    "DAY_TO_NIGHT",
    "NIGHT_TO_DAY",
]


class DN348PairDataset(Dataset):
    def __init__(
        self,
        pair_file,
        root="./data/dn348",
        image_size=256,
    ):
        self.root = Path(root)
        self.pairs = []

        with open(pair_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 3:
                    continue
                source, target, label = parts
                self.pairs.append((source, target, int(label)))

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5016, 0.5037, 0.5060],
                std=[0.1030, 0.0999, 0.0969],
            ),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source_rel, target_rel, label = self.pairs[idx]
        img1 = Image.open(self.root / source_rel).convert("RGB")
        img2 = Image.open(self.root / target_rel).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(
        description="DN-348 SIECategoricalOracle Predictor Metrics Evaluation"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to SIECategoricalOracle checkpoint",
    )
    parser.add_argument(
        "--val-pairs",
        type=Path,
        default=ROOT / "DN348_SIE/val_pairs.txt",
        help="Path to validation pair file (3200 pairs)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "datasets/dn348-001",
        help="Root path to DN-348 dataset",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save evaluation results JSON",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("DN-348 SIECategoricalOracle PREDICTOR EVALUATION")
    print("=" * 70)
    print(f"Device        : {device}")
    print(f"Checkpoint    : {args.checkpoint}")
    print(f"Val Pairs     : {args.val_pairs}")
    print(f"Dataset Root  : {args.dataset_root}")

    # 1. Dataset & DataLoader
    dataset = DN348PairDataset(
        pair_file=str(args.val_pairs),
        root=str(args.dataset_root),
        image_size=256,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Loaded {len(dataset)} evaluation pairs.\n")

    # 2. Model Configuration & Loading
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_state = ckpt["model"] if "model" in ckpt else ckpt
    clean_state = {
        k[7:] if k.startswith("module.") else k: v
        for k, v in model_state.items()
    }

    class Config:
        arch = "resnet18"
        equi = 256
        mlp = "2048-2048-2048"
        predictor_type = "hypernetwork"
        predictor = ""
        hypernetwork = "linear"
        freeze_backbone = True
        bias_hypernet = False
        bias_pred = False
        predictor_relu = False

    cfg = Config()
    if "args" in ckpt and isinstance(ckpt["args"], dict):
        for k, v in ckpt["args"].items():
            setattr(cfg, k.replace("-", "_"), v)
    elif (args.checkpoint.parent / "params.json").exists():
        with open(args.checkpoint.parent / "params.json", "r") as pf:
            params = json.load(pf)
            for k, v in params.items():
                setattr(cfg, k.replace("-", "_"), v)

    if not hasattr(cfg, "hypernetwork"):
        cfg.hypernetwork = "linear"

    model = m.SIECategoricalOracle(cfg, num_classes=348)
    msg = model.load_state_dict(clean_state, strict=True)
    print("Model loaded successfully into SIECategoricalOracle with msg:", msg)
    model = model.to(device)
    model.eval()

    # 3. Evaluation Loop
    cosine_sum = 0.0
    l2_sum = 0.0
    num_samples = 0

    per_trans_cosine = defaultdict(list)
    per_trans_l2 = defaultdict(list)

    print("\nRunning Oracle predictor forward passes (GT one-hot transition -> Hypernetwork)...")
    with torch.no_grad():
        for batch_idx, (img1, img2, labels) in enumerate(loader):
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Extract backbone features
            feat1 = model.backbone(img1)
            feat2 = model.backbone(img2)

            # Equivariant branch: slice and project
            inv_size = model.inv_repr_size
            equi1 = feat1[:, inv_size:]
            equi2 = feat2[:, inv_size:]
            equi1 = model.projector_equi(equi1)
            equi2 = model.projector_equi(equi2)

            # Oracle conditioning: ground-truth one-hot vector [B, 4]
            z_one_hot = F.one_hot(labels.long(), num_classes=4).float()

            # Hypernetwork receives equi1 and one_hot(transition_target)
            if model.args.predictor_type == "hypernetwork":
                pred = model.predictor(equi1.unsqueeze(1), z_one_hot)
            else:
                pred = model.predictor(equi1, z_one_hot)

            # Normalize both predicted and true target embeddings to unit L2 norm
            pred_norm = pred / (torch.linalg.norm(pred, dim=-1, keepdim=True) + 1e-8)
            equi2_norm = equi2 / (torch.linalg.norm(equi2, dim=-1, keepdim=True) + 1e-8)

            # Cosine similarity and L2 distance
            cosine = F.cosine_similarity(pred_norm, equi2_norm, dim=1)
            l2 = torch.norm(pred_norm - equi2_norm, dim=1)

            # Accumulate
            b_size = pred.size(0)
            cosine_sum += cosine.sum().item()
            l2_sum += l2.sum().item()
            num_samples += b_size

            # Per-transition accumulation
            for c_val, l_val, label_val in zip(cosine.cpu(), l2.cpu(), labels.cpu()):
                lbl = label_val.item()
                per_trans_cosine[lbl].append(c_val.item())
                per_trans_l2[lbl].append(l_val.item())

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
                print(f"Processed {num_samples}/{len(dataset)} samples...")

    mean_cosine = cosine_sum / num_samples
    mean_l2 = l2_sum / num_samples

    print("\n" + "=" * 70)
    print("DN-348 ORACLE PREDICTOR EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total Evaluated Samples     : {num_samples}")
    print(f"Mean Cosine Similarity      : {mean_cosine:.6f}")
    print(f"Mean L2 Distance            : {mean_l2:.6f}")
    print(f"Expected L2 (sqrt(2-2*cos)) : {np.sqrt(max(0, 2 - 2 * mean_cosine)):.6f}")
    print(f"Inverse Consistency Error   : N/A (Oracle uses GT transition; no transformation encoder)")
    print()

    print("Per-Transition Predictor Performance:")
    print(f"{'Transition':<16} {'Mean Cosine':>14} {'Mean L2':>12} {'N Samples':>12}")
    print("-" * 58)

    per_trans_summary = {}
    for i, name in enumerate(CLASS_NAMES):
        c_list = per_trans_cosine[i]
        l_list = per_trans_l2[i]
        c_mean = float(np.mean(c_list)) if c_list else 0.0
        l_mean = float(np.mean(l_list)) if l_list else 0.0
        n_count = len(c_list)
        per_trans_summary[name] = {
            "mean_cosine": c_mean,
            "mean_l2": l_mean,
            "count": n_count,
        }
        print(f"{name:<16} {c_mean:>14.6f} {l_mean:>12.6f} {n_count:>12d}")
    print("=" * 70 + "\n")

    # Save JSON results
    if args.output is None:
        args.output = args.checkpoint.parent / f"predictor_metrics_{args.checkpoint.stem}.json"

    results = {
        "model_architecture": "SIECategoricalOracle",
        "conditioning": "one_hot(ground_truth_transition)",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_type": str(ckpt.get("checkpoint_type", "final_epoch")),
        "selection_criterion": "final epoch-200 checkpoint (no transformation-accuracy-based selection exists for Oracle)",
        "dataset_root": str(args.dataset_root),
        "total_samples": num_samples,
        "mean_cosine": float(mean_cosine),
        "mean_l2": float(mean_l2),
        "mean_inverse_consistency": "N/A",
        "per_transition": per_trans_summary,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
