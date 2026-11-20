#!/usr/bin/env python3
"""
Representation Diagnostics for DN-348 ResNet-18 Baseline.
Computes:
1. Embedding dimensionality & L2 norm statistics
2. Effective rank (via singular value spectrum) & feature collapse check
3. Mean positive-pair (same-vehicle) cosine similarity
4. Mean negative-pair (different-vehicle) cosine similarity
5. Cosine similarity difference: positive - negative
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(".")
DN_ROOT = ROOT / "DN348_SIE"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DN_ROOT) not in sys.path:
    sys.path.insert(0, str(DN_ROOT))

from src.resnet18_baseline import ResNet18IdentityBaseline
import src.dataset_dn348 as ds_dn


def main():
    parser = argparse.ArgumentParser(description="DN-348 ResNet-18 Baseline Representation Diagnostics")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint")
    parser.add_argument("--dataset-root", type=Path, default=Path("./data/dn348"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("RESNET-18 BASELINE REPRESENTATION DIAGNOSTICS")
    print("=" * 70)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_state = ckpt["model"] if "model" in ckpt else ckpt
    clean_state = {k[7:] if k.startswith("module.") else k: v for k, v in model_state.items()}

    model = ResNet18IdentityBaseline(num_classes=200, embedding_dim=1024, pretrained=False)
    msg = model.load_state_dict(clean_state, strict=True)
    print(f"Loaded checkpoint with msg: {msg}")
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5016, 0.5037, 0.5060], std=[0.1030, 0.0999, 0.0969]),
    ])

    dataset = ds_dn.DatasetDN348(dataset_root=args.dataset_root, transform=transform, samples_per_epoch=args.batch_size * args.num_batches)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_pos_sims = []
    all_neg_sims = []
    all_norms = []
    all_stds = []
    all_z = []

    with torch.no_grad():
        for batch_idx, (x, y, z_trans, labels) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            B = x.shape[0]

            z_x = model.extract_features(x)
            z_y = model.extract_features(y)

            norm_x = torch.norm(z_x, dim=-1)
            all_norms.extend(norm_x.cpu().numpy().tolist())

            std_x = torch.sqrt(z_x.var(dim=0) + 1e-4)
            all_stds.append(std_x.cpu().numpy())
            all_z.append(z_x.cpu())

            sim_matrix = torch.matmul(z_x, z_y.T)
            pos_sim = torch.diagonal(sim_matrix).cpu().numpy()
            all_pos_sims.extend(pos_sim.tolist())

            mask = ~torch.eye(B, dtype=torch.bool, device=device)
            neg_sim = sim_matrix[mask].cpu().numpy()
            all_neg_sims.extend(neg_sim.tolist())

    pos_arr = np.array(all_pos_sims)
    neg_arr = np.array(all_neg_sims)
    norms_arr = np.array(all_norms)
    stds_arr = np.concatenate(all_stds, axis=0)

    # Singular value spectrum check for effective rank
    z_all = torch.cat(all_z, dim=0).float()
    z_centered = z_all - z_all.mean(dim=0)
    _, s, _ = torch.svd(z_centered)
    singular_values = s.numpy()
    p = singular_values / (singular_values.sum() + 1e-12)
    effective_rank = np.exp(-np.sum(p * np.log(p + 1e-12)))

    results = {
        "checkpoint": str(args.checkpoint.name),
        "embedding_dimensionality": 1024,
        "l2_norm": {
            "mean": float(np.mean(norms_arr)),
            "std": float(np.std(norms_arr)),
            "min": float(np.min(norms_arr)),
            "max": float(np.max(norms_arr)),
        },
        "positive_similarity": {
            "mean": float(np.mean(pos_arr)),
            "std": float(np.std(pos_arr)),
            "min": float(np.min(pos_arr)),
            "max": float(np.max(pos_arr)),
            "median": float(np.median(pos_arr)),
        },
        "negative_similarity": {
            "mean": float(np.mean(neg_arr)),
            "std": float(np.std(neg_arr)),
            "min": float(np.min(neg_arr)),
            "max": float(np.max(neg_arr)),
            "median": float(np.median(neg_arr)),
        },
        "similarity_difference_pos_minus_neg": float(np.mean(pos_arr) - np.mean(neg_arr)),
        "effective_rank": float(effective_rank),
        "representation_variance": {
            "mean_dim_std": float(np.mean(stds_arr)),
            "min_dim_std": float(np.min(stds_arr)),
            "max_dim_std": float(np.max(stds_arr)),
            "has_dimensional_collapse": bool(np.mean(stds_arr) < 0.01 or (np.mean(pos_arr) - np.mean(neg_arr) < 0.01)),
        }
    }

    print("\n" + "=" * 60)
    print("REPRESENTATION DIAGNOSTICS")
    print("=" * 60)
    print(f"Embedding Dim             : {results['embedding_dimensionality']}")
    print(f"L2 Norm                   : {results['l2_norm']['mean']:.4f} ± {results['l2_norm']['std']:.4f}")
    print(f"Positive Pair Cosine Sim  : {results['positive_similarity']['mean']:.4f} ± {results['positive_similarity']['std']:.4f}")
    print(f"Negative Pair Cosine Sim  : {results['negative_similarity']['mean']:.4f} ± {results['negative_similarity']['std']:.4f}")
    print(f"Pos - Neg Cosine Diff     : {results['similarity_difference_pos_minus_neg']:.4f}")
    print(f"Effective Rank            : {results['effective_rank']:.2f} / 1024")
    print(f"Mean Per-Dim Std          : {results['representation_variance']['mean_dim_std']:.4f}")
    print(f"Dimensional Collapse Check: {'COLLAPSE' if results['representation_variance']['has_dimensional_collapse'] else 'NO COLLAPSE'}")
    print("=" * 60)

    out_dir = args.out_dir if args.out_dir else args.checkpoint.parent
    out_file = out_dir / f"representation_diagnostics_{args.checkpoint.stem}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved diagnostics to: {out_file}\n")


if __name__ == "__main__":
    main()
