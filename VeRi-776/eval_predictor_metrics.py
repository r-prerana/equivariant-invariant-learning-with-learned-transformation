import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


import src.dataset as ds
import src.models as models


# ---------------------------------------------------------
# Arguments
# ---------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Evaluate SIE Predictor on VeRi"
)

parser.add_argument(
    "--dataset-root",
    type=Path,
    required=True,
)

parser.add_argument(
    "--checkpoint",
    type=Path,
    required=True,
)

parser.add_argument(
    "--params",
    type=Path,
    default=None,
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
)

parser.add_argument(
    "--num-workers",
    type=int,
    default=4,
)

parser.add_argument(
    "--device",
    default="cuda",
)

args = parser.parse_args()


# ---------------------------------------------------------
# Read training configuration
# ---------------------------------------------------------

if args.params is None:
    args.params = args.checkpoint.parent / "params.json"

with open(args.params, "r") as f:
    train_args = json.load(f)


class Config:
    pass


cfg = Config()

for k, v in train_args.items():
    setattr(cfg, k.replace("-", "_"), v)


device = torch.device(
    args.device if torch.cuda.is_available() else "cpu"
)

print("Using device :", device)
print("Checkpoint   :", args.checkpoint)
print("Params       :", args.params)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

normalize = transforms.Normalize(
    mean=[0.5016, 0.5037, 0.5060],
    std=[0.1030, 0.0999, 0.0969],
)

transform = transforms.Compose([
    transforms.Resize(
        (cfg.resolution, cfg.resolution)
    ),
    transforms.ToTensor(),
    normalize,
])

dataset = ds.DatasetVeRiPredictorEval(
    args.dataset_root,
    transform=transform,
)

loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)

print(f"\nDataset Size : {len(dataset)} vehicles")


# ---------------------------------------------------------
# Build Model
# ---------------------------------------------------------

print("\nBuilding model...")

model = models.__dict__[cfg.experience](
    cfg,
    num_classes=575,
)

model = model.to(device)

print("Done.")


# ---------------------------------------------------------
# Load Checkpoint
# ---------------------------------------------------------

print("\nLoading checkpoint...")

checkpoint = torch.load(
    args.checkpoint,
    map_location="cpu",
)

if "model" in checkpoint:
    state_dict = checkpoint["model"]
else:
    state_dict = checkpoint


# Remove DistributedDataParallel prefix ("module.")
clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        clean_state_dict[key[7:]] = value
    else:
        clean_state_dict[key] = value


model.load_state_dict(
    clean_state_dict,
    strict=True,
)

print("\nCheckpoint Loaded Successfully.")

# ---- DIAGNOSTIC v2: train vs eval mode, on a HARD case specifically ----

with torch.no_grad():

    # VeRi orientation label -> canonical angle
    DIAG_LABEL_TO_ANGLE = {
        0: 0.0,
        6: 45.0,
        5: 90.0,
        7: 135.0,
        1: 180.0,
        4: 225.0,
        2: 270.0,
        3: 315.0,
    }

    # Find a batch with a genuinely large relative transformation.
    target_sample = None

    for img1, img2, quat, lbl, ori1, ori2 in loader:

        o1 = int(ori1[0].item())
        o2 = int(ori2[0].item())

        a1 = DIAG_LABEL_TO_ANGLE[o1]
        a2 = DIAG_LABEL_TO_ANGLE[o2]

        rel = a2 - a1

        while rel > 180:
            rel -= 360

        while rel < -180:
            rel += 360

        if abs(rel) >= 135:
            target_sample = (
                img1,
                img2,
                quat,
                o1,
                o2,
                rel,
            )
            break

    if target_sample is None:

        print("No large-angle batch found -- check transition grouping.")

    else:

        img1, img2, quat, o1, o2, rel = target_sample

        img1 = img1.to(device)
        img2 = img2.to(device)

        print(
            f"Testing on transition=({o1},{o2}), "
            f"true_relative_angle={rel:.1f}deg"
        )

        # -----------------------------
        # Evaluation mode
        # -----------------------------

        model.eval()

        feat1_eval = model.backbone(img1)
        feat2_eval = model.backbone(img2)

        q_pred_eval = model.quat_encoder(
            feat1_eval,
            feat2_eval,
        )

        # -----------------------------
        # Training mode
        # -----------------------------

        model.train()

        feat1_train = model.backbone(img1)
        feat2_train = model.backbone(img2)

        q_pred_train = model.quat_encoder(
            feat1_train,
            feat2_train,
        )

        # Restore evaluation mode
        model.eval()

        print("=" * 60)
        print(
            "DIAGNOSTIC v2: train-mode vs eval-mode "
            "q_pred, HARD case"
        )
        print("=" * 60)

        print("q_pred (eval mode, first 3 samples):")
        print(q_pred_eval[:3])

        print("q_pred (train mode, first 3 samples):")
        print(q_pred_train[:3])

        feat_diff = (
            feat1_eval - feat1_train
        ).abs().mean().item()

        q_diff = (
            q_pred_eval - q_pred_train
        ).abs().mean().item()

        print(
            f"Mean abs backbone feature difference "
            f"(eval vs train): {feat_diff:.4f}"
        )

        print(
            f"Mean abs q_pred difference "
            f"(eval vs train): {q_diff:.4f}"
        )

        # -----------------------------
        # Predicted rotation angle
        # -----------------------------

        import math

        w_eval = (
            q_pred_eval[..., 3]
            .abs()
            .clamp(-1 + 1e-7, 1 - 1e-7)
        )

        w_train = (
            q_pred_train[..., 3]
            .abs()
            .clamp(-1 + 1e-7, 1 - 1e-7)
        )

        angle_eval = (
            2
            * torch.acos(w_eval)
            * 57.29578
        )

        angle_train = (
            2
            * torch.acos(w_train)
            * 57.29578
        )

        print(
            "q_pred's OWN predicted angle "
            f"(eval mode): {angle_eval[:3].tolist()}"
        )

        print(
            "q_pred's OWN predicted angle "
            f"(train mode): {angle_train[:3].tolist()}"
        )

        print(
            f"(true target angle was {abs(rel):.1f} deg)"
        )

        print("=" * 60)

# ---- END DIAGNOSTIC v2 ----

model.eval()


# ---------------------------------------------------------
# VeRi orientation label -> canonical angle
# ---------------------------------------------------------

LABEL_TO_ANGLE = {
    0:   0.0,    # Front
    6:  45.0,    # Right Front
    5:  90.0,    # Right
    7: 135.0,    # Right Rear
    1: 180.0,    # Rear
    4: 225.0,    # Left Rear
    2: 270.0,    # Left
    3: 315.0,    # Left Front
}


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

print("\nStarting Predictor Evaluation...\n")

cosine_sum = 0.0
l2_sum = 0.0
num_samples = 0

from collections import defaultdict

transition_errors = defaultdict(list)


spread_q_preds = []

with torch.no_grad():

    for batch_idx, (
        img1,
        img2,
        quaternion,
        labels,
        ori1,
        ori2,
    ) in enumerate(loader):

        img1 = img1.to(device)
        img2 = img2.to(device)
        quaternion = quaternion.to(device)

        # -------------------------------------------------
        # Transition information
        # -------------------------------------------------

        ori1_rep = int(ori1[0].item())
        ori2_rep = int(ori2[0].item())


        # -------------------------------------------------
        # Backbone Features
        # -------------------------------------------------

        feat1 = model.backbone(img1)
        feat2 = model.backbone(img2)


        # -------------------------------------------------
        # Split into Invariant / Equivariant
        # -------------------------------------------------

        inv_size = model.inv_repr_size

        inv1 = feat1[:, :inv_size]
        equi1 = feat1[:, inv_size:]

        inv2 = feat2[:, :inv_size]
        equi2 = feat2[:, inv_size:]


        # -------------------------------------------------
        # Project to Embedding Space
        # -------------------------------------------------

        equi1 = model.projector_equi(equi1)
        equi2 = model.projector_equi(equi2)


        # -------------------------------------------------
        # Predictor
        # -------------------------------------------------

        # If this checkpoint has a learned transformation
        # encoder (SIEQuatEncoder), condition the predictor
        # on q_pred computed from the raw backbone features.
        #
        # The TRUE quaternion is NEVER used for prediction
        # for this model type.
        #
        # The true quaternion is used only below for
        # diagnostic evaluation of q_pred.

        if hasattr(model, "quat_encoder"):

            if model.args.detach_trans_input:

                feat1_trans = feat1.detach()
                feat2_trans = feat2.detach()

            else:

                feat1_trans = feat1
                feat2_trans = feat2


            # ---------------------------------------------
            # Predict quaternion
            # ---------------------------------------------

            q_pred = model.quat_encoder(
                feat1_trans,
                feat2_trans,
            )

            # ---------------------------------------------
            # Accumulate q_pred for collapse diagnostic
            # ---------------------------------------------

            current_count = sum(
                x.shape[0] for x in spread_q_preds
            )

            if current_count < 1000:

                remaining = 1000 - current_count

                spread_q_preds.append(
                    q_pred.detach()[:remaining]
                )


            # ---------------------------------------------
            # Quaternion diagnostic
            # ---------------------------------------------

            q_pred_n = F.normalize(
                q_pred,
                dim=-1,
            )

            quat_n = F.normalize(
                quaternion,
                dim=-1,
            )

            dot = (
                q_pred_n * quat_n
            ).sum(dim=-1)

            dot = dot.abs().clamp(
                -1 + 1e-7,
                1 - 1e-7,
            )


            # -------------------------------------------------
            # FIXED INDENTATION
            # -------------------------------------------------

            geodesic_err_deg = (
                2.0 * torch.acos(dot)
            ).mean().item() * 57.29578


            # ---------------------------------------------
            # True relative angle
            # ---------------------------------------------

            true_angle1 = LABEL_TO_ANGLE[
                ori1_rep
            ]

            true_angle2 = LABEL_TO_ANGLE[
                ori2_rep
            ]

            rel = true_angle2 - true_angle1

            while rel > 180:
                rel -= 360

            while rel < -180:
                rel += 360


            # ---------------------------------------------
            # Diagnostic print
            # ---------------------------------------------

            print(
                f"Mean q_pred geodesic error vs true "
                f"quaternion (deg): {geodesic_err_deg:.2f} "
                f"| transition=({ori1_rep},{ori2_rep}) "
                f"true_relative_angle={rel:.1f}deg"
            )


            transition_errors[
                (ori1_rep, ori2_rep)
            ].append(
                geodesic_err_deg
            )


            # Learned quaternion is used as predictor
            # condition.
            condition = q_pred


        else:

            # SIE / SIENoVar oracle checkpoints:
            # true quaternion is fed directly.

            condition = quaternion


        # -------------------------------------------------
        # Predictor Forward Pass
        # -------------------------------------------------

        if cfg.predictor_type == "hypernetwork":

            pred = model.predictor(
                equi1.unsqueeze(1),
                condition,
            )

        else:

            pred = model.predictor(
                equi1,
                condition,
            )


        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        pred = pred / (
            torch.linalg.norm(
                pred,
                dim=-1,
                keepdim=True,
            ) + 1e-8
        )

        equi2 = equi2 / (
            torch.linalg.norm(
                equi2,
                dim=-1,
                keepdim=True,
            ) + 1e-8
        )


        cosine = F.cosine_similarity(
            pred,
            equi2,
            dim=1,
        )

        l2 = torch.norm(
            pred - equi2,
            dim=1,
        )


        # -------------------------------------------------
        # Sanity Check
        # -------------------------------------------------

        print(
            "Mean Cosine          :",
            cosine.mean().item(),
        )

        print(
            "Mean L2              :",
            l2.mean().item(),
        )


        expected_mean_l2 = torch.sqrt(
            torch.clamp(
                2 - 2 * cosine,
                min=0,
            )
        )

        print(
            "Mean Expected L2     :",
            expected_mean_l2.mean().item(),
        )

        print(
            "Difference           :",
            abs(
                l2.mean().item()
                - expected_mean_l2.mean().item()
            ),
        )


        # -------------------------------------------------
        # Accumulate Metrics
        # -------------------------------------------------

        cosine_sum += cosine.sum().item()
        l2_sum += l2.sum().item()

        num_samples += pred.size(0)


        if (batch_idx + 1) % 20 == 0:

            print(
                f"Processed {num_samples} samples..."
            )


# ---------------------------------------------------------
# q_pred COLLAPSE / SPREAD DIAGNOSTIC
# ---------------------------------------------------------

if spread_q_preds:

    all_q = torch.cat(
        spread_q_preds,
        dim=0,
    )

    all_q = F.normalize(
        all_q,
        dim=-1,
    )

    # Safety limit
    all_q = all_q[:1000]

    dot_matrix = (
        all_q @ all_q.t()
    ).abs().clamp(
        -1 + 1e-7,
        1 - 1e-7,
    )

    n = dot_matrix.size(0)

    mask = ~torch.eye(
        n,
        dtype=torch.bool,
        device=dot_matrix.device,
    )

    off_diag = dot_matrix[mask]

    mean_spread_deg = (
        2.0
        * torch.acos(off_diag)
        * 57.29578
    ).mean().item()

    print("=" * 60)
    print(
        "q_pred COLLAPSE / SPREAD DIAGNOSTIC"
    )
    print("=" * 60)

    print(
        f"Samples analysed             : {n}"
    )

    print(
        f"Mean pairwise q_pred spread  : "
        f"{mean_spread_deg:.2f} deg"
    )

    print(
        "Interpretation: <30 deg = "
        "strong collapse; 90-120 deg = "
        "healthy variation."
    )

    print("=" * 60)


# ---------------------------------------------------------
# Final Results
# ---------------------------------------------------------

if num_samples == 0:

    print("\nNo samples were evaluated.")

else:

    mean_cosine = cosine_sum / num_samples
    mean_l2 = l2_sum / num_samples


    print("=" * 60)

    print(
        f"Total Samples           : {num_samples}"
    )

    print(
        f"Mean Cosine Similarity  : {mean_cosine:.6f}"
    )

    print(
        f"Mean L2 Distance        : {mean_l2:.6f}"
    )

    print("=" * 60)


    # -----------------------------------------------------
    # Per-Transition Geodesic Error
    # -----------------------------------------------------

    print(
        "Per-Transition Geodesic Error Summary"
    )

    print("=" * 60)

    print(
        f"{'(ori1,ori2)':>14} "
        f"{'rel_angle':>10} "
        f"{'mean_err_deg':>14} "
        f"{'n_batches':>10}"
    )


    summary_rows = []


    for (o1, o2), errs in transition_errors.items():

        a1 = LABEL_TO_ANGLE[o1]
        a2 = LABEL_TO_ANGLE[o2]

        rel = a2 - a1

        while rel > 180:
            rel -= 360

        while rel < -180:
            rel += 360


        mean_err = sum(errs) / len(errs)


        summary_rows.append(
            (
                o1,
                o2,
                rel,
                mean_err,
                len(errs),
            )
        )


    # -----------------------------------------------------
    # Sort worst-first
    # -----------------------------------------------------

    summary_rows.sort(
        key=lambda r: -r[3]
    )


    for o1, o2, rel, mean_err, n in summary_rows:

        print(
            f"{'(' + str(o1) + ',' + str(o2) + ')':>14} "
            f"{rel:>9.1f}deg "
            f"{mean_err:>13.2f}deg "
            f"{n:>10}"
        )


    print("=" * 60)


    print(
        "Read this top-to-bottom: if the worst rows all "
        "share |rel_angle| close to 180, that points to "
        "a genuine ~180-degree optimization difficulty "
        "(hypothesis 2). If the worst rows are scattered "
        "across different rel_angle values but share one "
        "specific ori1 or ori2 label, that points to a "
        "mapping bug for that one label (hypothesis 3). "
        "If the worst rows correspond to (ori1,ori2) pairs "
        "you separately find are rare in DatasetVeRi's "
        "actual training pairs (see the training-data "
        "coverage diagnostic), that points to data scarcity "
        "(hypothesis 1)."
    )
