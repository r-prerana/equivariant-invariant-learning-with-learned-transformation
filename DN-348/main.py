# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from torchvision import transforms

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
#import augmentations as aug

import src.dataset as ds
import src.models as m
from copy import deepcopy

parser = argparse.ArgumentParser()

# Model
parser.add_argument("--arch", type=str, default="resnet18")
parser.add_argument("--equi", type=int, default=256)
parser.add_argument("--experience", type=str, choices=["SIENoVar","SIE","SIEOnlyEqui","VICReg","SimCLR","VICRegPartInv",
                                                        "SimCLROnlyEqui","SIERotColor","SimCLRAugSelf","SimCLRAugSelfRotColor",
                                                        "SimCLROnlyEquiRotColor","SimCLREquiModRotColor","SimCLREquiMod","VICRegEquiMod",
                                                        "SIEQuatEncoder", "SIEDayNightEncoder", "SIECategoricalOracle"],
                                                        default="SIE")
parser.add_argument("--hypernetwork", type=str, choices=["linear","deep"],default="linear")
# Only for when using an expander
parser.add_argument("--mlp", default="2048-2048-2048")
#Predictor architecture, in format "intermediate1-intermediate2-..."
parser.add_argument("--predictor", default="")
parser.add_argument("--pred-size-in",type=int, default=10)
parser.add_argument("--predictor-relu",  action="store_true")

# Predictor
parser.add_argument("--predictor-type",type=str,choices=["hypernetwork","mlp"],default="hypernetwork")
parser.add_argument("--bias-pred", action="store_true")
parser.add_argument("--bias-hypernet", action="store_true")
parser.add_argument("--simclr-temp",type=float,default=0.1)
parser.add_argument("--ec-weight",type=float,default=1)
parser.add_argument("--tf-num-layers",type=int,default=1)
parser.add_argument("--trans-backbone", type=str, default="1024-512-64",
    help="Hidden trunk of the quaternion encoder, everything before the final Linear-to-4 head.")
parser.add_argument("--trans-weight", type=float, default=10.0,
    help="Weight for the direct geodesic supervision of q_pred against z. This is the ONLY place z is used to train the encoder -- never fed to the predictor.")
parser.add_argument("--inverse-weight", type=float, default=1.0,
    help="Weight for the label-free multiplicative inverse-consistency loss: q_BA ~= conjugate(q_AB).")
parser.add_argument("--detach-trans-input", action="store_true", default=True,
    help="If set (default), x_emb/y_emb are detached before entering the quaternion encoder -- matches vanilla SIE's gradient structure (g was always external, never fed backbone gradient).")
parser.add_argument("--no-detach-trans-input", dest="detach_trans_input", action="store_false")
parser.add_argument("--freeze-backbone", action="store_true",
    help="Freeze backbone parameters and keep BatchNorm in eval mode (used for SIEDayNightEncoder).")
parser.add_argument("--weights-file", type=str, default="",
    help="Path to pretrained weights to initialize backbone before training.")
parser.add_argument("--lambda-id", type=float, default=0.0,
    help="Weight for identity InfoNCE loss on normalized invariant representations.")
parser.add_argument("--id-temp", type=float, default=0.1,
    help="Temperature for identity InfoNCE loss.")
parser.add_argument("--id-loss-type", type=str, choices=["ce", "arcface"], default="ce",
    help="Identity loss formulation: standard linear cross-entropy ('ce') or angular margin ArcFace ('arcface').")
parser.add_argument("--arcface-m", type=float, default=0.35,
    help="Angular margin m in radians for ArcFace identity loss.")
parser.add_argument("--arcface-s", type=float, default=32.0,
    help="Feature scale s for ArcFace identity loss.")



# Optim
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--base-lr", type=float, default=1e-3)
parser.add_argument("--backbone-lr", type=float, default=None, help="Differential learning rate for backbone")
parser.add_argument("--wd", type=float, default=1e-6)

parser.add_argument("--warmup-start",type=int, default=0)
parser.add_argument("--warmup-length",type=int, default=0)


# Data
parser.add_argument("--dataset-root", type=Path, default="DATA_FOLDER", required=True)
parser.add_argument("--images-file", type=Path, default="./data/train_images.npy", required=None)
parser.add_argument("--labels-file", type=Path, default="./data/val_images.npy", required=None)
parser.add_argument("--resolution", type=int, default=256)

# Checkpoints
parser.add_argument("--exp-dir", type=Path, default="")
parser.add_argument("--root-log-dir", type=Path,default="EXP_DIR/logs/")
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--eval-freq", type=int, default=10)
parser.add_argument("--log-freq-time", type=int, default=30)

# Loss
parser.add_argument("--sim-coeff", type=float, default=10.0)
parser.add_argument("--equi-factor", type=float, default=4.5)
parser.add_argument("--std-coeff", type=float, default=10.0)
parser.add_argument("--cov-coeff", type=float, default=1.0)

# Running
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--no-amp", action="store_true")
parser.add_argument("--port", type=int, default=52472)



class Tee:
    """Mirrors stdout/stderr to both console and a persistent file."""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if "SLURM_JOB_ID" in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = "scontrol show hostnames " + os.getenv("SLURM_JOB_NODELIST")
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv("SLURM_NODEID")) * args.ngpus_per_node
        args.world_size = int(os.getenv("SLURM_NNODES")) * args.ngpus_per_node
        args.dist_url = f"tcp://{host_name}:{args.port}"
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = f"tcp://localhost:{args.port}"
        args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        # Persistent log file mirroring all stdout/stderr
        log_file = open(args.exp_dir / "training.log", "a", buffering=1)
        sys.stdout = Tee(sys.__stdout__, log_file)
        sys.stderr = Tee(sys.__stderr__, log_file)

        # Config dumps (params.json and config.txt)
        dict_args = deepcopy(vars(args))
        for key, value in dict_args.items():
            if isinstance(value, Path):
                dict_args[key] = str(value)
        with open(args.exp_dir / "params.json", "w") as f:
            json.dump(dict_args, f, indent=2)
        with open(args.exp_dir / "config.txt", "w") as f:
            f.write("CMD: " + " ".join(sys.argv) + "\n\n")
            for k, v in sorted(dict_args.items()):
                f.write(f"{k}: {v}\n")

        # CSV metrics history header
        csv_file = args.exp_dir / "metrics.csv"
        if not csv_file.is_file():
            with open(csv_file, "w", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([
                    "epoch", "total_loss", "ssl_loss", "id_loss", "weighted_id_loss", "classif_loss",
                    "repr_loss_inv", "repr_loss_equi", "std_loss", "pred_std_loss",
                    "cov_loss", "trans_loss", "inverse_loss", "trans_accuracy",
                    "lr", "epoch_time_sec"
                ])

        # Dedicated TensorBoard dir
        tb_dir = args.exp_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)

        print("CMD: " + " ".join(sys.argv))

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    normalize = transforms.Normalize(
       mean=[0.5016, 0.5037, 0.5060], std=[0.1030, 0.0999, 0.0969]
    )
    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        normalize,
    ])

    if args.experience in ["SIEDayNightEncoder", "SIECategoricalOracle"]:
        import src.dataset_dn348 as ds_dn
        dataset = ds_dn.DatasetDN348(
            dataset_root=args.dataset_root,
            transform=transform,
        )
        num_classes = 348
    else:
        dataset = ds.DatasetVeRi(
            dataset_root=args.dataset_root,
            transform=transform,
        )
        num_classes = 575

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    print("per_device_batch_size",per_device_batch_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
    )

    model = m.__dict__[args.experience](
        args,
        num_classes=num_classes
    ).cuda(gpu)

    # Load pretrained backbone weights if specified
    if args.weights_file and Path(args.weights_file).is_file():
        if args.rank == 0:
            print(f"Loading pretrained backbone weights from: {args.weights_file}")
        ckpt = torch.load(args.weights_file, map_location="cpu")
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        backbone_state = {}
        for k, v in state_dict.items():
            if k.startswith("module.backbone."):
                backbone_state[k[len("module.backbone."):]] = v
            elif k.startswith("backbone."):
                backbone_state[k[len("backbone."):]] = v
            elif not any(k.startswith(p) for p in ["module.", "projector", "predictor", "quat_encoder", "daynight_encoder", "evaluator"]):
                backbone_state[k] = v
        msg = model.backbone.load_state_dict(backbone_state, strict=False)
        if args.rank == 0:
            print(f"Loaded backbone weights with msg: {msg}")

    if args.freeze_backbone and args.experience in ["SIEDayNightEncoder", "SIECategoricalOracle"]:
        for param in model.backbone.parameters():
            param.requires_grad = False
        model.backbone.eval()
        if args.rank == 0:
            print("Backbone frozen successfully.")

    if args.experience in ["SimCLR","SimCLROnlyEqui","SimCLROnlyEquiRotColor","SimCLREquiModRotColor","SimCLREquiMod"]:
        model.gpu = gpu
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=False)

    if getattr(args, "backbone_lr", None) is not None and not args.freeze_backbone:
        backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and "backbone." in n]
        head_params = [p for n, p in model.named_parameters() if p.requires_grad and "backbone." not in n]
        optimizer = torch.optim.Adam(
            [
                {"params": backbone_params, "lr": args.backbone_lr, "name": "backbone"},
                {"params": head_params, "lr": args.base_lr, "name": "heads"},
            ],
            weight_decay=args.wd,
        )
        if args.rank == 0:
            print("[OPTIMIZER] Differential Learning Rates Configured:")
            print(f"  Backbone LR : {args.backbone_lr:.2e} ({sum(p.numel() for p in backbone_params):,} params)")
            print(f"  Heads LR    : {args.base_lr:.2e} ({sum(p.numel() for p in head_params):,} params)")
    else:
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.base_lr,
            weight_decay=args.wd,
        )

    # Track the best transformation metric across ALL resumed runs.
    best_trans_err = float("inf")
    best_trans_acc = 0.0
    start_epoch = 0

    if (args.exp_dir / "model.pth").is_file():
        if args.rank == 0:
            print("[RESUME] Resuming from checkpoint:", args.exp_dir / "model.pth")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

        # Recover the best transformation metric from the best checkpoint.
        if (args.exp_dir / "model_best.pth").is_file():
            best_ckpt = torch.load(
                args.exp_dir / "model_best.pth",
                map_location="cpu"
            )
            best_trans_err = best_ckpt.get(
                "best_trans_err",
                float("inf")
            )
            best_trans_acc = best_ckpt.get(
                "best_trans_acc",
                0.0
            )

            if args.rank == 0:
                if args.experience == "SIEDayNightEncoder":
                    print(
                        f"Recovered best_trans_acc: {best_trans_acc:.2f}%"
                    )
                else:
                    print(
                        f"Recovered best_trans_err: "
                        f"{best_trans_err:.6f} rad "
                        f"({best_trans_err * 57.29578:.2f} deg)"
                    )

    if args.rank == 0:
        num_total_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_frozen_params = num_total_params - num_trainable_params
        print("\n" + "=" * 70)
        print("EXPERIMENT INITIALIZATION")
        print("=" * 70)
        print(f"Experience          : {args.experience}")
        print(f"Output Directory    : {args.exp_dir}")
        print(f"Pretrained Backbone : {args.weights_file if args.weights_file else 'None'}")
        print(f"Freeze Backbone     : {args.freeze_backbone}")
        print(f"Dataset             : {args.dataset_root}")
        print(f"Identities          : {num_classes}")
        print(f"Total Parameters    : {num_total_params:,}")
        print(f"Trainable Parameters: {num_trainable_params:,}")
        print(f"Frozen Parameters   : {num_frozen_params:,}")
        print(f"Hyperparameters     : lr={args.base_lr} (heads), lr={getattr(args, 'backbone_lr', args.base_lr)} (backbone)")
        print(f"                      batch_size={args.batch_size}, epochs={args.epochs}")
        print(f"                      trans_weight={args.trans_weight}, inverse_weight={args.inverse_weight}")
        print(f"                      sim_coeff={args.sim_coeff}, equi_factor={args.equi_factor}, std_coeff={args.std_coeff}")
        print("=" * 70 + "\n")

    start_time = last_logging = time.time()
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        sampler.set_epoch(epoch)
        epoch_total = 0.0
        epoch_ssl = 0.0
        epoch_cls = 0.0

        epoch_inv = 0.0
        epoch_equi = 0.0
        epoch_std = 0.0
        epoch_cov = 0.0
        epoch_predstd = 0.0
        epoch_trans = 0.0
        epoch_inverse = 0.0
        epoch_trans_acc = 0.0
        epoch_id = 0.0
        epoch_weighted_id = 0.0
        num_batches = 0
        progress = tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            disable=(args.rank != 0),
            leave=True
        )

        for step, (x, y, z, labels) in enumerate(
                progress,
                start=epoch * len(loader)):

            x = x.cuda(gpu, non_blocking=True)
            y = y.cuda(gpu, non_blocking=True)
            z = z.cuda(gpu, non_blocking=True)
            labels = labels.cuda(gpu, non_blocking=True)
            
            optimizer.zero_grad()

            # MAIN TRAINING PART
            with torch.cuda.amp.autocast(enabled=not args.no_amp):
                loss, classif_loss, stats, stats_eval = model.forward(x, y, z,labels)
                total_loss = loss + classif_loss

            scaler.scale(total_loss).backward()
            if step == 0 and args.rank == 0 and not args.freeze_backbone:
                bb_grads = [p.grad for n, p in model.named_parameters() if "backbone." in n and p.requires_grad and p.grad is not None]
                bb_grad_norms = [g.norm().item() for g in bb_grads]
                if len(bb_grad_norms) > 0 and sum(bb_grad_norms) > 0:
                    print(f"\n[BACKBONE GRADIENT CHECK] Step 0: Nonzero gradients verified across {len(bb_grads)} backbone parameters (mean norm = {sum(bb_grad_norms)/len(bb_grad_norms):.6f}).\n")
                else:
                    print("\n[BACKBONE GRADIENT CHECK] WARNING: Backbone did NOT receive nonzero gradients!\n")
            scaler.step(optimizer)
            scaler.update()
            postfix_dict = {
                "Total": f"{total_loss.item():.3f}",
                "SSL": f"{loss.item():.3f}",
                "Cls": f"{classif_loss.item():.3f}"
            }
            if "loss_id" in stats and stats["loss_id"].item() > 0:
                postfix_dict["ID"] = f"{stats['loss_id'].item():.3f}"
            progress.set_postfix(postfix_dict)
            epoch_total += total_loss.item()
            if "loss_ssl" in stats:
                epoch_ssl += stats["loss_ssl"].item()
            else:
                epoch_ssl += loss.item()
            if "loss_id" in stats:
                epoch_id += stats["loss_id"].item()
            if "weighted_loss_id" in stats:
                epoch_weighted_id += stats["weighted_loss_id"].item()
            epoch_cls += classif_loss.item()
            if "repr_loss_inv" in stats:
                epoch_inv += stats["repr_loss_inv"].item()

            if "repr_loss_equi" in stats:
                epoch_equi += stats["repr_loss_equi"].item()

            if "std_loss" in stats:
                epoch_std += stats["std_loss"].item()

            if "cov_loss" in stats:
                epoch_cov += stats["cov_loss"].item()

            if "pred_std_loss" in stats:
                epoch_predstd += stats["pred_std_loss"].item()
            if "trans_loss" in stats:
                epoch_trans += stats["trans_loss"].item()

            if "inverse_loss" in stats:
                epoch_inverse += stats["inverse_loss"].item()

            if "trans_accuracy" in stats:
                epoch_trans_acc += stats["trans_accuracy"].item()
            num_batches += 1
            current_time = time.time()
            if args.rank == 0 and (step == 0 or current_time - last_logging > args.log_freq_time):
                # General logs
                writer.add_scalar('General/epoch', epoch, step)
                writer.add_scalar('General/time_elapsed', int(current_time - start_time), step)
                #writer.add_scalar('General/lr', lr, step)
                writer.add_scalar('General/lr', args.base_lr, step)
                writer.add_scalar('General/Current GPU memory', torch.cuda.memory_allocated(torch.cuda.device('cuda:0'))/1e9, step)
                writer.add_scalar('General/Max GPU memory', torch.cuda.max_memory_allocated(torch.cuda.device('cuda:0'))/1e9, step)

                # Loss related logs
                writer.add_scalar('Loss/Total loss', stats["loss"].item(), step)
                if "loss_ssl" in stats:
                    writer.add_scalar('Loss/SSL loss', stats["loss_ssl"].item(), step)
                if "loss_id" in stats and stats["loss_id"].item() > 0:
                    writer.add_scalar('Loss/Identity InfoNCE loss', stats["loss_id"].item(), step)
                    writer.add_scalar('Loss/Weighted Identity loss', stats["weighted_loss_id"].item(), step)
                if "trans_loss" in stats:
                    writer.add_scalar(
                        'Loss/Transformation loss',
                        stats["trans_loss"].item(),
                        step
                    )

                if "inverse_loss" in stats:
                    writer.add_scalar(
                        'Loss/Inverse consistency loss',
                        stats["inverse_loss"].item(),
                        step
                    )
                if "trans_accuracy" in stats:
                    writer.add_scalar(
                        'Transformation/Accuracy',
                        stats["trans_accuracy"].item(),
                        step
                    )
                if args.experience in ["SimCLRAugSelf","SimCLRAugSelfFull","SimCLRAugSelfRotColor"]:
                    writer.add_scalar('Loss/Invariance loss', stats["repr_loss_inv"].item(), step)
                if not args.experience in ["SimCLR","SimCLRAugSelf","SimCLRAugSelfFull","SimCLRAugSelfRotColor","SimCLROnlyEqui","SimCLROnlyEquiRotColor","SimCLREquiModRotColor","SimCLREquiMod"]:
                    writer.add_scalar('Loss/Invariance loss', stats["repr_loss_inv"].item(), step)
                    writer.add_scalar('Loss/Std loss', stats["std_loss"].item(), step)
                    writer.add_scalar('Loss/Covariance loss', stats["cov_loss"].item(), step)
                if not args.experience in ["VICReg","VICRegNoCov","VICRegCos","VICRegL1","VICRegL1repr","FullEqui","VICRegPartInv","SimCLR","VICRegPartInv2Exps","SimCLROnlyEqui","SIERotColor","SimCLROnlyEquiRotColor"] :
                    writer.add_scalar('Loss/Equivariance loss', stats["repr_loss_equi"].item(), step)
                if args.experience in ["SIEOnlyEqui","SIE","SIEAll","SIERotColor","SIECategoricalOracle"]:
                    writer.add_scalar('Loss/Pred Std loss', stats["pred_std_loss"].item(), step)
                # Representations/embeddings stats
                writer.add_scalar('Stats/Corr. representations view1', stats["coremb_view1"].item(), step)
                writer.add_scalar('Stats/Corr. representations view2', stats["coremb_view2"].item(), step)
                writer.add_scalar('Stats/Std representations view1', stats["stdemb_view1"].item(), step)
                writer.add_scalar('Stats/Std representations view2', stats["stdemb_view2"].item(), step)
                writer.add_scalar('Stats/Corr. embeddings view1', stats["corhead_view1"].item(), step)
                writer.add_scalar('Stats/Corr. embeddings view2', stats["corhead_view2"].item(), step)
                writer.add_scalar('Stats/Std embeddings view1', stats["stdhead_view1"].item(), step)
                writer.add_scalar('Stats/Std embeddings view2', stats["stdhead_view2"].item(), step)
                if "stdemb_pred" in stats.keys():
                    writer.add_scalar('Stats/Corr. predictor output', stats["coremb_pred"].item(), step)
                    writer.add_scalar('Stats/Std predictor output', stats["stdemb_pred"].item(), step)

                
                # Online evaluation logs
                for key,value in stats_eval.items():
                    if "representations" in key:
                        writer.add_scalar(f'Online eval reprs/{key}', value, step)
                    elif "embeddings" in key:
                        writer.add_scalar(f'Online eval embs/{key}', value, step)
                for key,value in stats.items():
                    if "Latent/" in key:
                        writer.add_scalar(key, value, step)
                writer.flush()
                print("Logged, step :",step)
                last_logging = current_time
        if args.rank == 0:
            epoch_time = time.time() - epoch_start

            # Compact summary
            print(
                f"[Epoch {epoch + 1}/{args.epochs}] "
                f"total_loss={epoch_total / num_batches:.4f} | "
                f"ssl_loss={epoch_ssl / num_batches:.4f} | "
                f"id_loss={epoch_id / num_batches:.4f} | "
                f"eval_loss={epoch_cls / num_batches:.4f} | "
                f"trans_loss={epoch_trans / num_batches:.4f} | "
                f"trans_acc={epoch_trans_acc / num_batches:.2f}% | "
                f"repr_inv={epoch_inv / num_batches:.4f} | "
                f"repr_equi={epoch_equi / num_batches:.4f} | "
                f"inverse={epoch_inverse / num_batches:.4f} | "
                f"lr={args.base_lr:.2e} | "
                f"time={epoch_time:.1f}s"
            )

            # Detailed summary
            print(f"\n========== Epoch {epoch+1}/{args.epochs} ==========")
            print(f"Average Total Loss   : {epoch_total / num_batches:.4f}")
            print(f"Average SSL Loss     : {epoch_ssl / num_batches:.4f}")
            if epoch_id > 0:
                print(f"Identity InfoNCE Loss: {epoch_id / num_batches:.4f}")
                print(f"Weighted ID Loss     : {epoch_weighted_id / num_batches:.4f}")
            print(f"Average Class Loss   : {epoch_cls / num_batches:.4f}")

            print(f"Invariance Loss      : {epoch_inv / num_batches:.4f}")
            print(f"Equivariance Loss    : {epoch_equi / num_batches:.4f}")
            print(f"Std Loss             : {epoch_std / num_batches:.4f}")
            print(f"Covariance Loss      : {epoch_cov / num_batches:.4f}")
            if epoch_trans > 0:
                print(f"Transformation Loss  : {epoch_trans / num_batches:.4f}")

            if epoch_inverse > 0:
                print(f"Inverse Loss         : {epoch_inverse / num_batches:.4f}")
            if epoch_trans_acc > 0:
                print(f"Transformation Acc   : {epoch_trans_acc / num_batches:.2f}%")
            if epoch_predstd > 0:
                print(f"Predictor Std Loss   : {epoch_predstd / num_batches:.4f}")
            print(f"Epoch Time           : {epoch_time:.1f}s")
            print("=====================================\n")

            # CSV history row
            with open(args.exp_dir / "metrics.csv", "a", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([
                    epoch + 1,
                    f"{epoch_total / num_batches:.6f}",
                    f"{epoch_ssl / num_batches:.6f}",
                    f"{epoch_id / num_batches:.6f}",
                    f"{epoch_weighted_id / num_batches:.6f}",
                    f"{epoch_cls / num_batches:.6f}",
                    f"{epoch_inv / num_batches:.6f}",
                    f"{epoch_equi / num_batches:.6f}",
                    f"{epoch_std / num_batches:.6f}",
                    f"{epoch_predstd / num_batches:.6f}",
                    f"{epoch_cov / num_batches:.6f}",
                    f"{epoch_trans / num_batches:.6f}",
                    f"{epoch_inverse / num_batches:.6f}",
                    f"{epoch_trans_acc / num_batches:.4f}",
                    f"{args.base_lr:.6e}",
                    f"{epoch_time:.2f}",
                ])

            # TensorBoard epoch-level summaries
            writer.add_scalar('Epoch/Total_Loss', epoch_total / num_batches, epoch + 1)
            writer.add_scalar('Epoch/SSL_Loss', epoch_ssl / num_batches, epoch + 1)
            if epoch_id > 0:
                writer.add_scalar('Epoch/Identity_Loss', epoch_id / num_batches, epoch + 1)
                writer.add_scalar('Epoch/Weighted_Identity_Loss', epoch_weighted_id / num_batches, epoch + 1)
            writer.add_scalar('Epoch/Classif_Loss', epoch_cls / num_batches, epoch + 1)
            writer.add_scalar('Epoch/Invariance_Loss', epoch_inv / num_batches, epoch + 1)
            writer.add_scalar('Epoch/Equivariance_Loss', epoch_equi / num_batches, epoch + 1)
            writer.add_scalar('Epoch/Std_Loss', epoch_std / num_batches, epoch + 1)
            writer.add_scalar('Epoch/Cov_Loss', epoch_cov / num_batches, epoch + 1)
            if epoch_trans > 0:
                writer.add_scalar('Epoch/Transformation_Loss', epoch_trans / num_batches, epoch + 1)
            if epoch_inverse > 0:
                writer.add_scalar('Epoch/Inverse_Loss', epoch_inverse / num_batches, epoch + 1)
            if epoch_trans_acc > 0:
                writer.add_scalar('Epoch/Transformation_Accuracy', epoch_trans_acc / num_batches, epoch + 1)
            writer.flush()

            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                args=dict_args,
                best_trans_acc=best_trans_acc,
                best_trans_err=best_trans_err,
            )

            # Always save the latest checkpoint
            torch.save(
                state,
                args.exp_dir / "model.pth"
            )

            # Save the BEST checkpoint
            if num_batches > 0 and epoch_trans_acc > 0 and args.experience == "SIEDayNightEncoder":
                avg_trans_acc = epoch_trans_acc / num_batches
                avg_trans_loss = epoch_trans / num_batches
                if avg_trans_acc > best_trans_acc:
                    best_trans_acc = avg_trans_acc
                    best_state = dict(state)
                    best_state["best_trans_acc"] = best_trans_acc
                    best_state["checkpoint_type"] = "best_transformation_accuracy"
                    best_state["selection_criterion"] = "best transformation-accuracy checkpoint (online validation signal, not final retrieval checkpoint)"
                    torch.save(
                        best_state,
                        args.exp_dir / "model_best.pth"
                    )
                    print(
                        f"[BEST TRANSFORMATION-ACCURACY CHECKPOINT] "
                        f"Epoch {epoch + 1} | "
                        f"trans_acc = {avg_trans_acc:.2f}% | "
                        f"trans_loss = {avg_trans_loss:.6f}\n"
                        f"  (Note: Selected by online transformation accuracy, NOT as the final best SIE/retrieval checkpoint.)"
                    )
            elif num_batches > 0 and epoch_trans > 0:
                avg_trans_loss = epoch_trans / num_batches
                if avg_trans_loss < best_trans_err:
                    best_trans_err = avg_trans_loss
                    best_state = dict(state)
                    best_state["best_trans_err"] = best_trans_err
                    torch.save(
                        best_state,
                        args.exp_dir / "model_best.pth"
                    )
                    print(
                        f"[BEST CHECKPOINT] "
                        f"Epoch {epoch + 1} | "
                        f"trans_loss = {avg_trans_loss:.6f} | "
                        f"trans_err_deg = "
                        f"{avg_trans_loss * 57.29578:.2f}"
                    )

            # Save a snapshot every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save(
                    state,
                    args.exp_dir / f"checkpoint_epoch_{epoch + 1}.pth"
               )


    if args.rank == 0:
        writer.close()
        torch.save(model.module.backbone.state_dict(), args.exp_dir / "final_weights.pth")

    torch.distributed.destroy_process_group()


def exclude_bias_and_norm(p):
    return p.ndim == 1

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass


if __name__ == "__main__":
    main()


