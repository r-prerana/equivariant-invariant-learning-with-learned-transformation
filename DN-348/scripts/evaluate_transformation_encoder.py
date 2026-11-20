import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
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
from DN348_SIE.src.backbone import DN348Backbone
from DN348_SIE.src.dn_transform import DNTransformationEncoder


CLASS_NAMES = [
    "DAY_TO_DAY",
    "NIGHT_TO_NIGHT",
    "DAY_TO_NIGHT",
    "NIGHT_TO_DAY",
]


class DN348PairDataset(Dataset):
    """
    DN-348 image-pair dataset.
    Each line in the pair file has:
        source_path target_path transformation_label
    """

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
    parser = argparse.ArgumentParser(description="DN-348 Transformation Encoder Evaluation")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "DN348_SIE/checkpoints/dn348_v1/model_best.pth",
        help="Path to checkpoint (model_best.pth, model.pth, or legacy encoder checkpoint)",
    )
    parser.add_argument(
        "--val-pairs",
        type=Path,
        default=ROOT / "DN348_SIE/val_pairs.txt",
        help="Path to validation pair list",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("./data/dn348"),
        help="Path to dataset root",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path for results")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 70)
    print("DN-348 TRANSFORMATION ENCODER EVALUATION")
    print("=" * 70)
    print("Device     :", device)
    print("Checkpoint :", args.checkpoint)
    print("Val pairs  :", args.val_pairs)
    print("Dataset    :", args.dataset_root)

    # Dataset & Loader
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
    print(f"Loaded {len(dataset)} validation pairs.")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    is_sie_model = "model" in checkpoint or any(k.startswith("module.daynight_encoder") or k.startswith("daynight_encoder") for k in checkpoint.keys())

    if is_sie_model:
        print("\nLoading unified SIEDayNightEncoder model...")
        model_state = checkpoint["model"] if "model" in checkpoint else checkpoint
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
        if "args" in checkpoint and isinstance(checkpoint["args"], dict):
            for k, v in checkpoint["args"].items():
                setattr(cfg, k.replace("-", "_"), v)
        elif (args.checkpoint.parent / "params.json").exists():
            with open(args.checkpoint.parent / "params.json", "r") as pf:
                params = json.load(pf)
                for k, v in params.items():
                    setattr(cfg, k.replace("-", "_"), v)

        if not hasattr(cfg, "hypernetwork"):
            cfg.hypernetwork = "linear"

        sie_model = m.SIEDayNightEncoder(cfg, num_classes=348)
        msg = sie_model.load_state_dict(clean_state, strict=("classifier_id.weight" in clean_state))
        print("Model loaded successfully with msg:", msg)
        sie_model = sie_model.to(device)
        sie_model.eval()

        def forward_pairs(x_batch, y_batch):
            x_emb = sie_model.backbone(x_batch)
            y_emb = sie_model.backbone(y_batch)
            logits, probs = sie_model.daynight_encoder(x_emb, y_emb)
            return logits

    else:
        print("\nLoading legacy standalone transformation encoder...")
        backbone_ckpt_path = ROOT / "SIE_VeRi_V3Quat/checkpoints/veri_quatencoder_tw60_50ep_20260822_152234/checkpoint_epoch_700.pth"
        backbone = DN348Backbone(arch="resnet18")
        veri_checkpoint = torch.load(backbone_ckpt_path, map_location="cpu")
        veri_model_state = veri_checkpoint["model"]
        backbone_state = {k[len("module.backbone."):]: v for k, v in veri_model_state.items() if k.startswith("module.backbone.")}
        backbone.backbone.load_state_dict(backbone_state, strict=True)
        backbone = backbone.to(device).eval()
        for p in backbone.parameters():
            p.requires_grad = False

        encoder = DNTransformationEncoder(
            input_size=backbone.repr_size,
            hidden_sizes=(1024, 512, 64),
            num_classes=4,
        ).to(device)
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        encoder.eval()

        def forward_pairs(x_batch, y_batch):
            x_emb = backbone(x_batch)
            y_emb = backbone(y_batch)
            pair_feat = torch.cat([x_emb, y_emb, y_emb - x_emb], dim=1)
            logits = encoder.mlp(pair_feat)
            return logits

    # Evaluation
    confusion = torch.zeros(4, 4, dtype=torch.long)
    total = 0
    correct = 0
    total_ce_loss = 0.0

    print("\nRunning transformation evaluation...")
    with torch.no_grad():
        for x, y, labels in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = forward_pairs(x, y)
            ce = F.cross_entropy(logits, labels, reduction="sum")
            total_ce_loss += ce.item()

            preds = logits.argmax(dim=1)
            for target, pred in zip(labels.cpu(), preds.cpu()):
                confusion[target, pred] += 1

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = (correct / total) * 100.0
    mean_ce = total_ce_loss / total

    print("\n" + "=" * 70)
    print("DN-348 TRANSFORMATION ENCODER VALIDATION RESULTS")
    print("=" * 70)
    ckpt_epoch = int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1
    print(f"Epoch                  : {ckpt_epoch}")
    print(f"Total validation pairs : {total}")
    print(f"Overall Accuracy       : {accuracy:.2f}%")
    print(f"Cross-Entropy Loss     : {mean_ce:.6f}")
    print()
    print("Confusion Matrix:")
    print("Rows = Ground Truth | Cols = Prediction")
    print()
    print("                 ", end="")
    for name in CLASS_NAMES:
        print(f"{name:>16}", end="")
    print()

    per_class_results = {}
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:16}", end="")
        for j in range(4):
            print(f"{confusion[i, j].item():16d}", end="")
        print()

    print()
    print("Per-class accuracy / recall:")
    for i, name in enumerate(CLASS_NAMES):
        class_total = confusion[i].sum().item()
        class_correct = confusion[i, i].item()
        class_acc = (class_correct / class_total * 100.0) if class_total > 0 else 0.0
        per_class_results[name] = {
            "correct": class_correct,
            "total": class_total,
            "accuracy": class_acc,
        }
        print(f"  {name:16}: {class_acc:6.2f}% ({class_correct}/{class_total})")
    print("=" * 70 + "\n")

    if args.output is None:
        args.output = args.checkpoint.parent / f"transformation_eval_{args.checkpoint.stem}.json"

    results = {
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt_epoch,
        "total_pairs": total,
        "accuracy": float(accuracy),
        "cross_entropy": float(mean_ce),
        "confusion_matrix": confusion.tolist(),
        "class_names": CLASS_NAMES,
        "per_class": per_class_results,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to: {args.output}")


if __name__ == "__main__":
    main()
