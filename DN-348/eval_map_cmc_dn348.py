# eval_map_cmc_dn348.py
#
# Diagnostic 5C: Final cross-domain retrieval evaluation for DN-348.
#
# Computes Rank-1, Rank-5, Rank-10, and mAP for:
#   1. Day -> Night (Query: all test Day images; Gallery: all test Night images)
#   2. Night -> Day (Query: all test Night images; Gallery: all test Day images)
#
# Uses extracted invariant branch features only. Features are extracted ONCE
# and reused for both directions.

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

import src.models as m
import src.dataset_dn348 as ds_dn


def compute_cmc_map(query_feats, query_ids, gallery_feats, gallery_ids, desc="Evaluating"):
    num_queries = query_feats.size(0)
    cmc = np.zeros(gallery_feats.size(0))
    aps = []

    gallery_ids = np.asarray(gallery_ids)
    query_ids = np.asarray(query_ids)

    valid_queries = 0
    for i in tqdm(range(num_queries), desc=desc, leave=False):
        sim = torch.matmul(gallery_feats, query_feats[i]).cpu().numpy()
        ranked_idx = np.argsort(-sim)
        ranked_ids = gallery_ids[ranked_idx]

        matches = (ranked_ids == query_ids[i]).astype(int)
        if matches.sum() == 0:
            continue

        valid_queries += 1

        # CMC
        first_match = np.where(matches == 1)[0][0]
        cmc[first_match:] += 1

        # AP
        cumsum = np.cumsum(matches)
        ranks = np.arange(1, len(matches) + 1)
        precision = cumsum / ranks
        ap = (precision * matches).sum() / matches.sum()
        aps.append(ap)

    cmc = (cmc / valid_queries) * 100.0 if valid_queries > 0 else cmc
    mAP = np.mean(aps) * 100.0 if len(aps) > 0 else 0.0
    return cmc, mAP, valid_queries


def main():
    parser = argparse.ArgumentParser(description="DN-348 Cross-Domain Retrieval Evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained SIEDayNightEncoder checkpoint")
    parser.add_argument("--dataset-root", type=Path, default=Path("./data/dn348"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None, help="Path to save results JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading checkpoint:", args.checkpoint)
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

    # Overlay config from checkpoint args or params.json if available
    if "args" in ckpt and isinstance(ckpt["args"], dict):
        for k, v in ckpt["args"].items():
            setattr(cfg, k.replace("-", "_"), v)
    elif (args.checkpoint.parent / "params.json").exists():
        with open(args.checkpoint.parent / "params.json", "r") as pf:
            params = json.load(pf)
            for k, v in params.items():
                setattr(cfg, k.replace("-", "_"), v)

    # Ensure required attributes exist
    if not hasattr(cfg, "hypernetwork"):
        cfg.hypernetwork = "linear"

    if getattr(cfg, "experience", "") == "ResNet18IdentityBaseline" or "classifier.weight" in clean_state:
        from src.resnet18_baseline import ResNet18IdentityBaseline
        model = ResNet18IdentityBaseline(num_classes=200, embedding_dim=1024, pretrained=False)
        print("Instantiating ResNet18IdentityBaseline for evaluation.")
    elif getattr(cfg, "experience", "") == "SIECategoricalOracle" or "daynight_encoder.head.weight" not in clean_state:
        model = m.SIECategoricalOracle(cfg, num_classes=348)
        print("Instantiating SIECategoricalOracle for evaluation.")
    else:
        model = m.SIEDayNightEncoder(cfg, num_classes=348)
        print("Instantiating SIEDayNightEncoder for evaluation.")
    msg = model.load_state_dict(clean_state, strict=("classifier_id.weight" in clean_state))
    print("Checkpoint loaded successfully with msg:", msg)
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5016, 0.5037, 0.5060], std=[0.1030, 0.0999, 0.0969])
    ])

    test_dataset = ds_dn.DatasetDN348Eval(args.dataset_root, transform=transform, split="test")
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    day_feats, day_ids = [], []
    night_feats, night_ids = [], []

    print("\nExtracting invariant representations for test images...")
    with torch.no_grad():
        for imgs, vids, conds, _ in tqdm(loader, desc="Feature extraction"):
            embs = model.backbone(imgs.to(device))
            inv_embs = embs[..., :model.inv_repr_size]
            inv_proj = model.projector_inv(inv_embs)
            inv_norm = F.normalize(inv_proj, dim=-1)

            for feat, vid, cond in zip(inv_norm, vids, conds):
                if cond == "day":
                    day_feats.append(feat)
                    day_ids.append(vid)
                else:
                    night_feats.append(feat)
                    night_ids.append(vid)

    day_feats = torch.stack(day_feats, dim=0)
    night_feats = torch.stack(night_feats, dim=0)

    num_identities = len(set(day_ids) | set(night_ids))
    print(f"Test Identities   : {num_identities} (100% overlap between day and night)")
    print(f"Test Day Images   : {len(day_feats)}")
    print(f"Test Night Images : {len(night_feats)}")

    # 1. Day -> Night
    print("\nEvaluating Day -> Night retrieval...")
    cmc_d2n, map_d2n, valid_d2n = compute_cmc_map(day_feats, day_ids, night_feats, night_ids, desc="Day -> Night")

    # 2. Night -> Day
    print("Evaluating Night -> Day retrieval...")
    cmc_n2d, map_n2d, valid_n2d = compute_cmc_map(night_feats, night_ids, day_feats, day_ids, desc="Night -> Day")

    print("\n" + "=" * 60)
    print("DN-348 CROSS-DOMAIN RETRIEVAL RESULTS")
    print("=" * 60)
    print(f"Total Test Identities: {num_identities}")
    print("-" * 60)
    print("Direction: DAY -> NIGHT")
    print(f"  Valid Queries : {valid_d2n} / {len(day_feats)}")
    print(f"  Gallery Size  : {len(night_feats)}")
    print(f"  Rank-1        : {cmc_d2n[0]:6.2f}%")
    print(f"  Rank-5        : {cmc_d2n[4]:6.2f}%")
    print(f"  Rank-10       : {cmc_d2n[9]:6.2f}%")
    print(f"  mAP           : {map_d2n:6.2f}%")
    print("-" * 60)
    print("Direction: NIGHT -> DAY")
    print(f"  Valid Queries : {valid_n2d} / {len(night_feats)}")
    print(f"  Gallery Size  : {len(day_feats)}")
    print(f"  Rank-1        : {cmc_n2d[0]:6.2f}%")
    print(f"  Rank-5        : {cmc_n2d[4]:6.2f}%")
    print(f"  Rank-10       : {cmc_n2d[9]:6.2f}%")
    print(f"  mAP           : {map_n2d:6.2f}%")
    print("=" * 60 + "\n")

    # Save structured JSON
    if args.output is None:
        args.output = args.checkpoint.parent / f"map_cmc_results_{args.checkpoint.stem}.json"

    results = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_type": str(ckpt.get("checkpoint_type", "N/A")),
        "dataset_root": str(args.dataset_root),
        "num_identities": int(num_identities),
        "test_day_images": int(len(day_feats)),
        "test_night_images": int(len(night_feats)),
        "invariant_repr_size": int(getattr(model, "inv_repr_size", 512)),
        "invariant_embedding_size": int(getattr(model, "inv_emb_size", 1024)),
        "day_to_night": {
            "num_queries": int(len(day_feats)),
            "valid_queries": int(valid_d2n),
            "gallery_size": int(len(night_feats)),
            "rank1": float(cmc_d2n[0]),
            "rank5": float(cmc_d2n[4]),
            "rank10": float(cmc_d2n[9]),
            "mAP": float(map_d2n),
        },
        "night_to_day": {
            "num_queries": int(len(night_feats)),
            "valid_queries": int(valid_n2d),
            "gallery_size": int(len(day_feats)),
            "rank1": float(cmc_n2d[0]),
            "rank5": float(cmc_n2d[4]),
            "rank10": float(cmc_n2d[9]),
            "mAP": float(map_n2d),
        },
    }

    with open(args.output, "w") as jf:
        json.dump(results, jf, indent=4)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()

