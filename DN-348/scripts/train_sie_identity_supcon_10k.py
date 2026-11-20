#!/usr/bin/env python3
"""
DN-348 SIE Identity-Balanced Supervised Contrastive (SupCon) Continuation Script (Steps 5,000 -> 10,000).

Resumes from the completed Step-5,000 checkpoint:
  checkpoints/dn348_sie_identity_supcon/checkpoint_step_5000.pth

Executes exactly 5,000 additional optimizer.step() calls on the DN-348 dataset
under the identical configuration as the 5k run:
  1. Identity-balanced multi-image batches (16 identities x 4 images = 64 images)
  2. Supervised Contrastive Identity Loss (lambda_supcon = 0.50, tau = 0.1)
     applied to normalized 1024-D invariant projector output.
  3. Total Objective:
     L_total = L_existing_SIE + 0.50 * L_identity_CE + 0.50 * L_SupCon + L_eval

Checkpoints and evaluates at:
  - Step 6000
  - Step 7000
  - Step 8000
  - Step 9000
  - Step 10000

Strict isolation: All outputs stored under checkpoints/dn348_sie_identity_supcon/.
"""

import sys
import os
import time
import json
import argparse
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

DN_ROOT = Path(__file__).resolve().parent.parent
if str(DN_ROOT) not in sys.path:
    sys.path.insert(0, str(DN_ROOT))

import src.models as m
import src.daynight_transform as dn
from src.supcon_loss import SupConLoss
from src.dataset_dn348_supcon import (
    IdentityBalancedDN348Dataset,
    IdentityBalancedBatchSampler,
    build_sie_pairs_from_batch,
)


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


def run_evaluation(checkpoint_path, exp_dir, step_num):
    """
    Runs the official evaluation suite for a specific checkpoint step:
    1. Cross-illumination retrieval (eval_map_cmc_dn348.py)
    2. Transformation accuracy (evaluate_transformation_encoder.py)
    3. Predictor metrics (eval_predictor_metrics_dn348.py)
    4. Representation diagnostics (eval_identity_representation_diagnostics.py)
    """
    print(f"\n{'='*75}")
    print(f"RUNNING STEP {step_num} EVALUATION SUITE")
    print(f"{'='*75}")

    # 1. Retrieval
    cmd_retrieval = [
        "python3", str(DN_ROOT / "eval_map_cmc_dn348.py"),
        "--checkpoint", str(checkpoint_path),
        "--output", str(exp_dir / f"map_cmc_results_step_{step_num}.json")
    ]
    subprocess.run(cmd_retrieval, cwd=str(DN_ROOT), check=True)

    # 2. Transformation
    cmd_trans = [
        "python3", str(DN_ROOT / "scripts/evaluate_transformation_encoder.py"),
        "--checkpoint", str(checkpoint_path),
        "--output", str(exp_dir / f"transformation_eval_step_{step_num}.json")
    ]
    subprocess.run(cmd_trans, cwd=str(DN_ROOT), check=True)

    # 3. Predictor metrics
    cmd_pred = [
        "python3", str(DN_ROOT / "eval_predictor_metrics_dn348.py"),
        "--checkpoint", str(checkpoint_path),
        "--output", str(exp_dir / f"predictor_metrics_step_{step_num}.json")
    ]
    subprocess.run(cmd_pred, cwd=str(DN_ROOT), check=True)

    # 4. Representation diagnostics
    cmd_geom = [
        "python3", str(DN_ROOT / "scripts/eval_identity_representation_diagnostics.py"),
        "--checkpoint", str(checkpoint_path),
        "--num-batches", "20",
        "--out-dir", str(exp_dir)
    ]
    subprocess.run(cmd_geom, cwd=str(DN_ROOT), check=True)
    print(f"Completed evaluation suite for Step {step_num}.\n")


def forward_sie_supcon(model, supcon_fn, images, labels, domains, cfg):
    """
    Executes the proposed forward pass:
      1. Shared backbone forward on all 64 batch images
      2. Split into invariant (256-D) and equivariant (256-D) representations
      3. Invariant projector -> 1024-D; Equivariant projector -> 1024-D
      4. SupCon Loss on normalized 1024-D invariant projector output
      5. Identity CE Loss on unnormalized 1024-D invariant representation
      6. Existing SIE Losses on 64 constructed pairs (DD, NN, DN, ND)
         - Hypernetwork receives ONLY t_pred (predicted probabilities), never GT.
      7. Online evaluator loss on detached representations
    """
    # 1. Shared backbone forward
    embs = model.backbone(images)  # [64, 512]
    inv_repr = embs[:, :model.inv_repr_size]  # [64, 256]
    equi_repr = embs[:, model.inv_repr_size:]  # [64, 256]

    # 2. Projectors
    inv_emb = model.projector_inv(inv_repr)  # [64, 1024]
    equi_emb = model.projector_equi(equi_repr)  # [64, 1024]

    # 3. SupCon Loss on normalized invariant representation
    inv_emb_norm = F.normalize(inv_emb, dim=-1)
    loss_supcon = supcon_fn(inv_emb_norm, labels)
    weighted_loss_supcon = cfg.lambda_supcon * loss_supcon

    # 4. Identity CE Loss on unnormalized invariant representation
    logits_id = model.classifier_id(inv_emb)  # [64, 200]
    loss_id = F.cross_entropy(logits_id, labels)
    weighted_loss_id = cfg.lambda_id * loss_id
    acc_id = (logits_id.argmax(dim=-1) == labels).float().mean() * 100.0

    # 5. Build balanced SIE pairs (64 pairs: 16 DD, 16 NN, 16 DN, 16 ND)
    idx_x, idx_y, targets, pair_labels = build_sie_pairs_from_batch(labels, domains)

    x_emb, y_emb = embs[idx_x], embs[idx_y]
    x_inv, y_inv = inv_emb[idx_x], inv_emb[idx_y]
    x_equi, y_equi = equi_emb[idx_x], equi_emb[idx_y]

    # Invariant pair similarity loss
    repr_loss_inv = F.mse_loss(x_inv, y_inv)

    # Transformation branch
    if cfg.detach_trans_input:
        x_trans = x_emb.detach()
        y_trans = y_emb.detach()
    else:
        x_trans = x_emb
        y_trans = y_emb

    logits_fwd, probs_fwd = model.daynight_encoder(x_trans, y_trans)
    logits_rev, probs_rev = model.daynight_encoder(y_trans, x_trans)

    trans_loss = dn.daynight_transformation_loss(logits_fwd, targets)
    inverse_loss = dn.daynight_inverse_consistency_loss(probs_fwd, probs_rev)
    trans_acc = dn.daynight_transformation_accuracy(logits_fwd, targets)

    # Equivariant branch: Hypernetwork receives ONLY predicted transformation (probs_fwd)
    if cfg.predictor_type == "hypernetwork":
        y_equi_pred = model.predictor(x_equi.unsqueeze(1), probs_fwd)
    elif cfg.predictor_type in ["mlp"]:
        y_equi_pred = model.predictor(x_equi, probs_fwd)
    else:
        y_equi_pred = model.predictor(x_equi.unsqueeze(1), probs_fwd)

    repr_loss_equi = F.mse_loss(y_equi_pred, y_equi)

    # Std and Covariance regularizers
    x_comb = torch.cat((x_inv, x_equi), dim=1)
    y_comb = torch.cat((y_inv, y_equi), dim=1)

    y_equi_pred_g = torch.cat(m.FullGatherLayer.apply(y_equi_pred), dim=0)
    y_equi_pred_g = y_equi_pred_g - y_equi_pred_g.mean(dim=0)
    std_y_equi_pred = torch.sqrt(y_equi_pred_g.var(dim=0) + 0.0001)
    pred_std_loss = torch.mean(F.relu(1 - std_y_equi_pred)) / 2

    x_g = torch.cat(m.FullGatherLayer.apply(x_comb), dim=0)
    y_g = torch.cat(m.FullGatherLayer.apply(y_comb), dim=0)
    x_g = x_g - x_g.mean(dim=0)
    y_g = y_g - y_g.mean(dim=0)

    std_x = torch.sqrt(x_g.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y_g.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

    cov_x = (x_g.T @ x_g) / (x_comb.shape[0] - 1)
    cov_y = (y_g.T @ y_g) / (y_comb.shape[0] - 1)
    cov_loss = m.off_diagonal(cov_x).pow_(2).sum().div(cov_x.shape[0]) + m.off_diagonal(cov_y).pow_(2).sum().div(cov_x.shape[0])

    # Existing SIE SSL objective
    loss_ssl = (
        cfg.sim_coeff * repr_loss_inv
        + cfg.equi_factor * cfg.sim_coeff * repr_loss_equi
        + cfg.std_coeff * std_loss
        + cfg.std_coeff * pred_std_loss
        + cfg.cov_coeff * cov_loss
        + cfg.trans_weight * trans_loss
        + cfg.inverse_weight * inverse_loss
    )

    # Online evaluator for online representation monitoring
    loss_eval, stats_eval = model.evaluator(
        [x_emb.detach(), y_emb.detach()],
        [x_comb.detach(), y_comb.detach()],
        pair_labels,
        angles=None,
    )

    # Total conceptual objective: L_existing_SIE + 0.50 * L_identity_CE + 0.50 * L_SupCon + loss_eval
    total_loss = loss_ssl + weighted_loss_id + weighted_loss_supcon + loss_eval

    stats = {
        "loss_total": total_loss,
        "loss_ssl": loss_ssl,
        "loss_id": loss_id,
        "weighted_loss_id": weighted_loss_id,
        "loss_supcon": loss_supcon,
        "weighted_loss_supcon": weighted_loss_supcon,
        "loss_eval": loss_eval,
        "acc_id": acc_id,
        "repr_loss_inv": repr_loss_inv,
        "repr_loss_equi": repr_loss_equi,
        "std_loss": std_loss,
        "pred_std_loss": pred_std_loss,
        "cov_loss": cov_loss,
        "trans_loss": trans_loss,
        "inverse_loss": inverse_loss,
        "trans_accuracy": trans_acc,
    }

    return total_loss, stats, stats_eval


def main():
    parser = argparse.ArgumentParser(description="DN-348 SIE Identity SupCon Continuation (5,000 -> 10,000 Steps)")
    parser.add_argument("--resume", type=Path, default=DN_ROOT / "checkpoints/dn348_sie_identity_supcon/checkpoint_step_5000.pth")
    parser.add_argument("--dataset-root", type=Path, default=Path("./data/dn348"))
    parser.add_argument("--exp-dir", type=Path, default=DN_ROOT / "checkpoints/dn348_sie_identity_supcon")
    parser.add_argument("--target-steps", type=int, default=10000)
    parser.add_argument("--num-identities", type=int, default=16)
    parser.add_argument("--images-per-id", type=int, default=4)
    parser.add_argument("--base-lr", type=float, default=0.001)
    parser.add_argument("--backbone-lr", type=float, default=0.0001)
    parser.add_argument("--lambda-id", type=float, default=0.50)
    parser.add_argument("--lambda-supcon", type=float, default=0.50)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-eval", action="store_true", help="Skip periodic evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Execute 1 dry-run step and exit")
    args = parser.parse_args()

    args.exp_dir = Path(args.exp_dir)
    assert args.resume.exists(), f"Resume checkpoint not found: {args.resume}"

    # Setup logging (append mode)
    log_file = open(args.exp_dir / "training.log", "a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    device = torch.device(args.device)

    print("=" * 75)
    print("DN-348 SIE IDENTITY SUPCON CONTINUATION (5,000 -> 10,000 STEPS)")
    print("=" * 75)
    print(f"Resume Checkpoint      : {args.resume}")
    print(f"Target Optimizer Steps : {args.target_steps}")
    print(f"Checkpoint Milestones  : 6000, 7000, 8000, 9000, 10000")
    print(f"Batch Structure        : {args.num_identities} IDs x {args.images_per_id} images = {args.num_identities * args.images_per_id} images")
    print(f"Heads LR               : {args.base_lr}")
    print(f"Backbone LR            : {args.backbone_lr}")
    print(f"Identity CE Lambda     : {args.lambda_id}")
    print(f"SupCon Lambda          : {args.lambda_supcon} (tau = {args.temperature})")
    print(f"Output Directory       : {args.exp_dir}")
    print(f"Device                 : {device}")
    print("=" * 75)

    # Config object matching proposed SIE params
    class Config:
        def __init__(self):
            self.arch = "resnet18"
            self.equi = 256
            self.mlp = "2048-2048-2048"
            self.predictor_type = "hypernetwork"
            self.predictor = ""
            self.hypernetwork = "linear"
            self.tf_num_layers = 1
            self.trans_backbone = "1024-512-64"
            self.freeze_backbone = False
            self.detach_trans_input = True
            self.bias_hypernet = False
            self.bias_pred = False
            self.predictor_relu = False
            self.batch_size = args.num_identities * args.images_per_id
            self.num_identities = args.num_identities
            self.images_per_id = args.images_per_id
            self.epochs = 200
            self.lambda_id = args.lambda_id
            self.lambda_supcon = args.lambda_supcon
            self.temperature = args.temperature
            self.id_temp = 0.1
            self.sim_coeff = 5.0
            self.equi_factor = 4.5
            self.std_coeff = 35.0
            self.cov_coeff = 1.0
            self.trans_weight = 60.0
            self.inverse_weight = 1.0
            self.base_lr = args.base_lr
            self.backbone_lr = args.backbone_lr
            self.wd = args.wd
            self.weights_file = str(args.resume)
            self.exp_dir = str(args.exp_dir)
            self.dataset_root = str(args.dataset_root)
            self.experience = "SIEDayNightEncoder"
            self.resolution = 256
            self.num_workers = args.num_workers
            self.no_amp = False

    cfg = Config()

    # Load 5k checkpoint
    print(f"Loading checkpoint state from: {args.resume}")
    ckpt = torch.load(args.resume, map_location="cpu")
    start_step = ckpt.get("step", 5000)
    start_epoch = ckpt.get("epoch", 100)
    print(f"Resumed state: Step = {start_step}, Epoch = {start_epoch}")
    assert start_step == 5000, f"Expected step 5000, got {start_step}"

    # Build model and SupCon loss
    model = m.SIEDayNightEncoder(cfg, num_classes=348)
    supcon_fn = SupConLoss(temperature=args.temperature)

    clean_state = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt["model"].items()}
    msg = model.load_state_dict(clean_state, strict=True)
    print(f"Model state loaded with message: {msg}")

    model.to(device)
    supcon_fn.to(device)

    # Verify parameter counts
    num_total = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters     : {num_total:,}")
    print(f"Trainable Parameters : {num_trainable:,}")
    assert num_trainable == 23780412, f"Trainable parameter mismatch: {num_trainable} != 23,780,412"

    # Optimizer
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and "backbone." in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "backbone." not in n]
    optimizer = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": args.backbone_lr, "name": "backbone"},
            {"params": head_params, "lr": args.base_lr, "name": "heads"},
        ],
        weight_decay=args.wd,
    )
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        print("Optimizer state restored successfully from step-5000 checkpoint.")
    else:
        print("WARNING: Optimizer state not found in checkpoint; initializing fresh Adam.")

    scaler = torch.amp.GradScaler('cuda', init_scale=1024.0)

    # Dataset & Identity-Balanced Sampler
    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5016, 0.5037, 0.5060], std=[0.1030, 0.0999, 0.0969]),
    ])
    dataset = IdentityBalancedDN348Dataset(dataset_root=args.dataset_root, transform=transform)
    sampler = IdentityBalancedBatchSampler(
        dataset,
        num_identities=args.num_identities,
        images_per_id=args.images_per_id,
        batches_per_epoch=50,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Training identities  : {len(dataset.vehicle_ids)}")
    print(f"Total images pool    : {len(dataset)}")
    print(f"Batches per cycle    : {len(sampler)}")

    milestones = [6000, 7000, 8000, 9000, 10000]

    # Metrics CSV tracking (append mode)
    csv_file = args.exp_dir / "step_metrics.csv"
    if not csv_file.exists():
        with open(csv_file, "w", newline="") as f:
            f.write("step,epoch,total_loss,ssl_loss,id_loss,supcon_loss,trans_loss,trans_acc,repr_inv,repr_equi,time_sec\n")

    global_step = start_step
    epoch = start_epoch
    training_start = time.time()

    if args.dry_run:
        print("\n[DRY-RUN MODE] Executing 1 step test from Step 5,000...")
        model.train()
        images, labels, domains, paths = next(iter(loader))
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        domains = domains.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            total_loss, stats, stats_eval = forward_sie_supcon(model, supcon_fn, images, labels, domains, cfg)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print(f"Dry-run forward/backward SUCCESSFUL from step {global_step}!")
        print(f"  Batch shape   : {list(images.shape)}")
        print(f"  Total loss    : {total_loss.item():.4f}")
        print(f"  SSL loss      : {stats['loss_ssl'].item():.4f}")
        print(f"  ID CE loss    : {stats['loss_id'].item():.4f}")
        print(f"  SupCon loss   : {stats['loss_supcon'].item():.4f}")
        print(f"  Trans Acc     : {stats['trans_accuracy'].item() * 100:.1f}%")
        print("\nDry run completed cleanly. Exiting as requested.")
        return

    print(f"\nStarting step-budget continuation loop ({global_step} -> {args.target_steps} steps)...")
    while global_step < args.target_steps:
        epoch += 1
        model.train()
        for images, labels, domains, paths in loader:
            step_start = time.time()
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            domains = domains.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                total_loss, stats, stats_eval = forward_sie_supcon(model, supcon_fn, images, labels, domains, cfg)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            step_time = time.time() - step_start

            # Print progress every 10 steps
            if global_step % 10 == 0 or global_step in milestones or global_step == start_step + 1:
                t_loss = total_loss.item()
                ssl_l = stats["loss_ssl"].item()
                id_l = stats["loss_id"].item()
                sup_l = stats["loss_supcon"].item()
                t_acc = stats["trans_accuracy"].item() * 100.0
                print(
                    f"[Step {global_step:5d}/{args.target_steps:5d} | Epoch {epoch:3d}] "
                    f"Total: {t_loss:7.3f} | SSL: {ssl_l:7.3f} | ID: {id_l:6.3f} | "
                    f"SupCon: {sup_l:6.3f} | TransAcc: {t_acc:5.1f}% | Time: {step_time:4.2f}s"
                )

            # Log to CSV
            with open(csv_file, "a") as f:
                f.write(
                    f"{global_step},{epoch},{total_loss.item():.4f},"
                    f"{stats['loss_ssl'].item():.4f},"
                    f"{stats['loss_id'].item():.4f},"
                    f"{stats['loss_supcon'].item():.4f},"
                    f"{stats['trans_loss'].item():.4f},"
                    f"{stats['trans_accuracy'].item()*100.0:.2f},"
                    f"{stats['repr_loss_inv'].item():.4f},"
                    f"{stats['repr_loss_equi'].item():.4f},"
                    f"{step_time:.3f}\n"
                )

            # Checkpoint and evaluation at target milestones
            if global_step in milestones:
                ckpt_name = f"checkpoint_step_{global_step}.pth"
                ckpt_path = args.exp_dir / ckpt_name
                state = {
                    "epoch": epoch,
                    "step": global_step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
                }
                torch.save(state, ckpt_path)
                print(f"\n>>> [CHECKPOINT SAVED] Milestone Step {global_step}: {ckpt_path}")

                if global_step == args.target_steps:
                    torch.save(state, args.exp_dir / "model.pth")
                    torch.save(model.backbone.state_dict(), args.exp_dir / "final_weights.pth")
                    print(f">>> [FINAL CHECKPOINT SAVED] model.pth and final_weights.pth")

                if not args.skip_eval:
                    run_evaluation(ckpt_path, args.exp_dir, global_step)

            if global_step >= args.target_steps:
                break

    total_time = time.time() - training_start
    print("\n" + "=" * 75)
    print(f"CONTINUATION COMPLETE: Executed exactly {global_step} optimizer steps.")
    print(f"Total Epochs Completed : {epoch}")
    print(f"Additional Steps Run   : {global_step - start_step}")
    print(f"Continuation Wallclock : {total_time/3600:.2f} hours ({total_time:.1f} seconds)")
    print(f"Output Checkpoints     : {args.exp_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
