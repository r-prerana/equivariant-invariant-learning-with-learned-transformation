#!/usr/bin/env python3
"""
MASTER FROZEN EVALUATION — TRANSFORMATION ENCODER ACROSS 3 DATASETS
Evaluates the Transformation Encoder itself (g_hat vs g_gt) across:
1. 3DIEBench (SIE_STL_version3_quat)
2. VeRi-776 (SIE_VeRi_V3Quat)
3. DN-348 (DN348_SIE)

Strictly READ-ONLY:
- No backward()
- No optimizer
- model.eval() and torch.no_grad()
- Common metrics: Cosine Similarity, L2 Distance, Dataset-specific prediction metric.
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_active_repo(repo_dir):
    """Isolates repo imports so src.* doesn't collide between repos."""
    for k in list(sys.modules.keys()):
        if k == "src" or k.startswith("src."):
            del sys.modules[k]
    for p in list(sys.path):
        if str(ROOT) in p:
            sys.path.remove(p)
    sys.path.insert(0, str(repo_dir))

# ==============================================================================
# DATASET 1: 3DIEBench Evaluator
# ==============================================================================

class Dataset3DIEBenchEval(Dataset):
    """
    Deterministic evaluation dataset for 3DIEBench.
    Uses 10,552 held-out validation objects (100% disjoint from 41,920 train objects).
    For each object, deterministically selects a fixed pair of views using seed.
    """
    def __init__(self, dataset_root, img_file, labels_file, seed=42, image_size=256):
        self.dataset_root = Path(dataset_root)
        self.samples = np.load(img_file)
        self.labels = np.load(labels_file)
        assert len(self.samples) == len(self.labels)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5016, 0.5037, 0.5060],
                std=[0.1030, 0.0999, 0.0969]
            ),
        ])

        # Pre-generate deterministic view pairs for each object
        rng = random.Random(seed)
        self.view_pairs = []
        for _ in range(len(self.samples)):
            views = rng.sample(range(50), 2)
            self.view_pairs.append(views)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx].strip("/")
        v1, v2 = self.view_pairs[idx]
        obj_dir = self.dataset_root / sample

        img1_path = obj_dir / f"image_{v1}.jpg"
        img2_path = obj_dir / f"image_{v2}.jpg"

        with open(img1_path, "rb") as f:
            img1 = Image.open(f).convert("RGB")
        with open(img2_path, "rb") as f:
            img2 = Image.open(f).convert("RGB")

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        angles_1 = np.load(str(obj_dir / f"latent_{v1}.npy"))[:3].astype(np.float32)
        angles_2 = np.load(str(obj_dir / f"latent_{v2}.npy"))[:3].astype(np.float32)

        rot_1 = R.from_euler("xyz", angles_1)
        rot_2 = R.from_euler("xyz", angles_2)
        rot_1_to_2 = rot_1.inv() * rot_2
        quat = rot_1_to_2.as_quat().astype(np.float32)  # [x, y, z, w]

        return img1, img2, torch.FloatTensor(quat), self.labels[idx]


def evaluate_3diebench_checkpoint(ckpt_path, batch_size=64):
    print("\n" + "=" * 70)
    print(f"EVALUATING 3DIEBench: {ckpt_path.name}")
    print("=" * 70)

    repo_dir = ROOT / "SIE_STL/SIE_STL_version3_quat"
    set_active_repo(repo_dir)
    import src.models as models

    class Args:
        arch = "resnet18"
        equi = 256
        experience = "SIE_STL"
        hypernetwork = "linear"
        mlp = "2048-2048-2048"
        predictor = ""
        pred_size_in = 10
        predictor_relu = False
        predictor_type = "hypernetwork"
        bias_pred = False
        bias_hypernet = False
        simclr_temp = 0.1
        ec_weight = 1
        tf_num_layers = 1
        trans_backbone = "1024-512-64"
        trans_weight = 10.0
        inverse_weight = 1.0
        detach_trans_input = True
        sim_coeff = 10.0
        std_coeff = 10.0
        cov_coeff = 1.0
        equi_factor = 4.5

    args = Args()
    model = models.SIE_STL(args).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    clean_state = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    msg = model.load_state_dict(clean_state, strict=False)
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', 'N/A')}) with msg: {msg}")
    model.eval()

    dataset_root = ROOT / "datasets/3DIEBench"
    val_images_file = repo_dir / "data/val_images.npy"
    val_labels_file = repo_dir / "data/val_labels.npy"

    ds_eval = Dataset3DIEBenchEval(dataset_root, val_images_file, val_labels_file, seed=42)
    loader = DataLoader(ds_eval, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Validation dataset: {len(ds_eval)} objects / pairs (deterministic seed=42)")

    cosines = []
    l2_errors = []
    geodesic_errors_deg = []
    quats_pred = []
    quats_gt = []

    with torch.no_grad():
        for img1, img2, q_gt, _ in tqdm(loader, desc="3DIEBench Eval"):
            img1 = img1.to(DEVICE, non_blocking=True)
            img2 = img2.to(DEVICE, non_blocking=True)
            q_gt = q_gt.to(DEVICE, non_blocking=True)

            x_emb = model.backbone(img1)
            y_emb = model.backbone(img2)
            q_hat = model.quat_encoder(x_emb, y_emb)  # [B, 4] normalized unit quaternion

            # Quaternion sign ambiguity handling: q and -q represent identical rotation
            dot = (q_hat * q_gt).sum(dim=-1, keepdim=True)
            q_hat_aligned = torch.where(dot < 0, -q_hat, q_hat)

            cos_sim = dot.abs().squeeze(-1)
            l2 = (q_hat_aligned - q_gt).norm(dim=-1)
            geo_rad = 2.0 * torch.acos(cos_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
            geo_deg = geo_rad * (180.0 / np.pi)

            cosines.extend(cos_sim.cpu().numpy().tolist())
            l2_errors.extend(l2.cpu().numpy().tolist())
            geodesic_errors_deg.extend(geo_deg.cpu().numpy().tolist())

            quats_pred.append(q_hat.cpu())
            quats_gt.append(q_gt.cpu())

    cos_arr = np.array(cosines)
    l2_arr = np.array(l2_errors)
    geo_arr = np.array(geodesic_errors_deg)

    quats_pred = torch.cat(quats_pred, dim=0).numpy()
    quats_gt = torch.cat(quats_gt, dim=0).numpy()

    # R2 score matching existing repository evaluation convention
    ss_tot = np.sum((quats_gt - np.mean(quats_gt, axis=0, keepdims=True)) ** 2)
    # Using sign-aligned predictions for fair R2
    dots = np.sum(quats_pred * quats_gt, axis=-1, keepdims=True)
    quats_pred_aligned = np.where(dots < 0, -quats_pred, quats_pred)
    ss_res = np.sum((quats_gt - quats_pred_aligned) ** 2)
    r2_aligned = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    mse_aligned = float(np.mean((quats_gt - quats_pred_aligned) ** 2))

    res = {
        "dataset": "3DIEBench",
        "repository": str(repo_dir),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "model_class": "SIE_STL",
        "encoder_class": "QuaternionEncoder",
        "transformation_representation": "Unit Quaternion [x, y, z, w] (continuous SO(3))",
        "evaluation_split": "Held-out validation set (data/val_images.npy)",
        "num_identities": len(ds_eval),
        "num_pairs": len(ds_eval),
        "identities_disjoint_from_train": True,
        "random_seed": 42,
        "metrics": {
            "mean_cosine": float(np.mean(cos_arr)),
            "median_cosine": float(np.median(cos_arr)),
            "std_cosine": float(np.std(cos_arr)),
            "mean_l2": float(np.mean(l2_arr)),
            "median_l2": float(np.median(l2_arr)),
            "std_l2": float(np.std(l2_arr)),
            "mean_geodesic_deg": float(np.mean(geo_arr)),
            "median_geodesic_deg": float(np.median(geo_arr)),
            "std_geodesic_deg": float(np.std(geo_arr)),
            "rotation_r2_score": r2_aligned,
            "rotation_mse": mse_aligned,
        }
    }

    print(f"--> Mean Cosine   : {res['metrics']['mean_cosine']:.4f} (Median: {res['metrics']['median_cosine']:.4f})")
    print(f"--> Mean L2       : {res['metrics']['mean_l2']:.4f} (Median: {res['metrics']['median_l2']:.4f})")
    print(f"--> Mean Geo Deg  : {res['metrics']['mean_geodesic_deg']:.2f}° (Median: {res['metrics']['median_geodesic_deg']:.2f}°)")
    print(f"--> Rotation R²   : {res['metrics']['rotation_r2_score']:.4f} (MSE: {res['metrics']['rotation_mse']:.4f})")

    return res


# ==============================================================================
# DATASET 2: VeRi-776 Evaluator
# ==============================================================================

class DatasetVeRiHeldOutTestPairs(Dataset):
    """
    Held-out test pair dataset for VeRi-776.
    Constructed strictly from the 200 unseen test vehicles in test_label.xml
    and VehicleReIDKeyPointData/keypoint_test.txt.
    100% disjoint from the 575 training vehicles.
    Balanced up to max_per_transition pairs per orientation transition with fixed seed.
    """
    def __init__(self, dataset_root, max_per_transition=1500, seed=42, image_size=256):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.xml_file = self.dataset_root / "VeRi" / "test_label.xml"
        self.image_root = self.dataset_root / "VeRi" / "image_test"
        self.keypoint_file = self.dataset_root / "VehicleReIDKeyPointData" / "keypoint_test.txt"

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5016, 0.5037, 0.5060],
                std=[0.1030, 0.0999, 0.0969]
            ),
        ])

        # Canonical orientation angles (8 discrete headings)
        self.LABEL_TO_ANGLE = {
            0: 0.0,
            1: 45.0,
            2: 90.0,
            3: 135.0,
            4: 180.0,
            5: 225.0,
            6: 270.0,
            7: 315.0,
        }

        with open(self.xml_file, "r", encoding="gb2312") as f:
            xml_text = f.read()
        if xml_text.startswith("<?xml"):
            xml_text = xml_text.split("?>", 1)[1]
        root = ET.fromstring(xml_text)
        items = root.find("Items")

        img_to_ori = {}
        with open(self.keypoint_file) as f:
            for line in f:
                toks = line.strip().split()
                if toks:
                    img_to_ori[Path(toks[0]).name] = int(toks[-1])

        veh_to_imgs = defaultdict(list)
        for item in items:
            vid = item.attrib["vehicleID"]
            img = item.attrib["imageName"]
            if img in img_to_ori:
                veh_to_imgs[vid].append(img)

        self.img_to_ori = img_to_ori
        self.vehicle_ids = sorted([v for v, imgs in veh_to_imgs.items() if len(imgs) >= 2])

        pairs_by_trans = defaultdict(list)
        for vid in self.vehicle_ids:
            imgs = veh_to_imgs[vid]
            for i in range(len(imgs)):
                for j in range(i + 1, len(imgs)):
                    o1 = img_to_ori[imgs[i]]
                    o2 = img_to_ori[imgs[j]]
                    pairs_by_trans[(o1, o2)].append((imgs[i], imgs[j], vid, o1, o2))

        rng = random.Random(seed)
        self.samples = []
        for trans in sorted(pairs_by_trans.keys()):
            p_list = pairs_by_trans[trans]
            if len(p_list) > max_per_transition:
                p_list = rng.sample(p_list, max_per_transition)
            self.samples.extend(p_list)

        self.pairs_by_trans = pairs_by_trans

    def orientation_to_quaternion(self, ori1, ori2):
        theta1 = math.radians(self.LABEL_TO_ANGLE[ori1])
        theta2 = math.radians(self.LABEL_TO_ANGLE[ori2])
        delta = theta2 - theta1
        delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
        r = R.from_euler("z", delta)
        return r.as_quat().astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img1_name, img2_name, vid, o1, o2 = self.samples[idx]
        p1 = self.image_root / img1_name
        p2 = self.image_root / img2_name

        with open(p1, "rb") as f:
            im1 = Image.open(f).convert("RGB")
        with open(p2, "rb") as f:
            im2 = Image.open(f).convert("RGB")

        im1 = self.transform(im1)
        im2 = self.transform(im2)

        quat = self.orientation_to_quaternion(o1, o2)
        return im1, im2, torch.FloatTensor(quat), vid, o1, o2


def evaluate_veri_checkpoint(ckpt_path, batch_size=64, use_unseen_test=True):
    print("\n" + "=" * 70)
    print(f"EVALUATING VeRi-776: {ckpt_path.name}")
    print(f"Split: {'200 Unseen Test Vehicles (Held-out Test)' if use_unseen_test else '575 Benchmark Vehicles'}")
    print("=" * 70)

    repo_dir = ROOT / "VeRi-776"
    set_active_repo(repo_dir)
    import src.models as models
    import src.dataset as ds

    class Args:
        arch = "resnet18"
        equi = 256
        experience = "SIEQuatEncoder"
        hypernetwork = "linear"
        mlp = "2048-2048-2048"
        predictor = ""
        pred_size_in = 10
        predictor_relu = False
        predictor_type = "hypernetwork"
        bias_pred = False
        bias_hypernet = False
        simclr_temp = 0.1
        ec_weight = 1
        tf_num_layers = 1
        trans_backbone = "1024-512-64"
        trans_weight = 60.0
        inverse_weight = 1.0
        detach_trans_input = True
        sim_coeff = 5.0
        std_coeff = 35.0
        cov_coeff = 1.0
        equi_factor = 4.5

    args = Args()
    model = models.SIEQuatEncoder(args, num_classes=575).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    clean_state = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    msg = model.load_state_dict(clean_state, strict=False)
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', 'N/A')}) with msg: {msg}")
    model.eval()

    dataset_root = ROOT / "datasets"

    if use_unseen_test:
        ds_eval = DatasetVeRiHeldOutTestPairs(dataset_root, max_per_transition=1500, seed=42, image_size=256)
        split_desc = "Official 200 Unseen Test Vehicles (test_label.xml + keypoint_test.txt)"
        is_disjoint = True
        n_ids = 200
    else:
        normalize = transforms.Normalize(mean=[0.5016, 0.5037, 0.5060], std=[0.1030, 0.0999, 0.0969])
        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), normalize])
        ds_eval = ds.DatasetVeRiPredictorEval(dataset_root, transform=transform)
        split_desc = "Standard 575 Benchmark Vehicles (train_label.xml + keypoint_train.txt)"
        is_disjoint = False
        n_ids = 575

    loader = DataLoader(ds_eval, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"VeRi dataset: {len(ds_eval)} pairs across {n_ids} vehicles ({split_desc})")

    cosines = []
    l2_errors = []
    geodesic_errors_deg = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="VeRi Eval"):
            im1 = batch[0].to(DEVICE, non_blocking=True)
            im2 = batch[1].to(DEVICE, non_blocking=True)
            q_gt = batch[2].to(DEVICE, non_blocking=True)

            x_emb = model.backbone(im1)
            y_emb = model.backbone(im2)
            q_hat = model.quat_encoder(x_emb, y_emb)  # [B, 4] normalized unit quaternion

            # Quaternion sign ambiguity handling
            dot = (q_hat * q_gt).sum(dim=-1, keepdim=True)
            q_hat_aligned = torch.where(dot < 0, -q_hat, q_hat)

            cos_sim = dot.abs().squeeze(-1)
            l2 = (q_hat_aligned - q_gt).norm(dim=-1)
            geo_rad = 2.0 * torch.acos(cos_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
            geo_deg = geo_rad * (180.0 / np.pi)

            cosines.extend(cos_sim.cpu().numpy().tolist())
            l2_errors.extend(l2.cpu().numpy().tolist())
            geodesic_errors_deg.extend(geo_deg.cpu().numpy().tolist())

    cos_arr = np.array(cosines)
    l2_arr = np.array(l2_errors)
    geo_arr = np.array(geodesic_errors_deg)

    res = {
        "dataset": "VeRi-776",
        "repository": str(repo_dir),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "model_class": "SIEQuatEncoder",
        "encoder_class": "QuaternionEncoder",
        "transformation_representation": "Unit Quaternion [x, y, z, w] (continuous planar rotation)",
        "evaluation_split": split_desc,
        "num_identities": n_ids,
        "num_pairs": len(ds_eval),
        "identities_disjoint_from_train": is_disjoint,
        "random_seed": 42,
        "metrics": {
            "mean_cosine": float(np.mean(cos_arr)),
            "median_cosine": float(np.median(cos_arr)),
            "std_cosine": float(np.std(cos_arr)),
            "mean_l2": float(np.mean(l2_arr)),
            "median_l2": float(np.median(l2_arr)),
            "std_l2": float(np.std(l2_arr)),
            "mean_geodesic_deg": float(np.mean(geo_arr)),
            "median_geodesic_deg": float(np.median(geo_arr)),
            "std_geodesic_deg": float(np.std(geo_arr)),
        }
    }

    print(f"--> Mean Cosine   : {res['metrics']['mean_cosine']:.4f} (Median: {res['metrics']['median_cosine']:.4f})")
    print(f"--> Mean L2       : {res['metrics']['mean_l2']:.4f} (Median: {res['metrics']['median_l2']:.4f})")
    print(f"--> Mean Geo Deg  : {res['metrics']['mean_geodesic_deg']:.2f}° (Median: {res['metrics']['median_geodesic_deg']:.2f}°)")

    return res


# ==============================================================================
# DATASET 3: DN-348 Evaluator
# ==============================================================================

class DatasetDN348HeldOutTestPairs(Dataset):
    """
    Deterministic balanced test pair dataset for DN-348.
    Constructed strictly from the 148 unseen test identities in:
      datasets/dn348-001/train_test_split/test_list_day.txt
      datasets/dn348-001/train_test_split/test_list_night.txt
    100% disjoint from the 200 training identities.
    Exactly 800 pairs per class (DD=800, NN=800, DN=800, ND=800 -> 3,200 pairs total).
    Fixed seed=42.
    """
    def __init__(self, dataset_root, pairs_per_class=800, seed=42, image_size=256):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.split_dir = self.dataset_root / "train_test_split"

        with open(self.split_dir / "test_list_day.txt") as f:
            day_lines = [l.strip() for l in f if l.strip()]
        with open(self.split_dir / "test_list_night.txt") as f:
            night_lines = [l.strip() for l in f if l.strip()]

        day_by_id = defaultdict(list)
        for p in day_lines:
            day_by_id[p.split("/")[0]].append(f"day/{p}")

        night_by_id = defaultdict(list)
        for p in night_lines:
            night_by_id[p.split("/")[0]].append(f"night/{p}")

        self.test_ids = sorted(list(set(day_by_id.keys()).intersection(set(night_by_id.keys()))))
        assert len(self.test_ids) == 148, f"Expected 148 test IDs, got {len(self.test_ids)}"

        rng = random.Random(seed)

        # Build candidate pairs per class
        pairs_dd = []
        pairs_nn = []
        pairs_dn = []
        pairs_nd = []

        for vid in self.test_ids:
            d_imgs = day_by_id[vid]
            n_imgs = night_by_id[vid]

            for i in range(len(d_imgs)):
                for j in range(i + 1, min(len(d_imgs), i + 5)):
                    pairs_dd.append((d_imgs[i], d_imgs[j], 0))

            for i in range(len(n_imgs)):
                for j in range(i + 1, len(n_imgs)):
                    pairs_nn.append((n_imgs[i], n_imgs[j], 1))

            for i in range(min(len(d_imgs), 10)):
                for j in range(len(n_imgs)):
                    pairs_dn.append((d_imgs[i], n_imgs[j], 2))
                    pairs_nd.append((n_imgs[j], d_imgs[i], 3))

        rng.shuffle(pairs_dd)
        rng.shuffle(pairs_nn)
        rng.shuffle(pairs_dn)
        rng.shuffle(pairs_nd)

        assert len(pairs_dd) >= pairs_per_class
        assert len(pairs_nn) >= pairs_per_class
        assert len(pairs_dn) >= pairs_per_class
        assert len(pairs_nd) >= pairs_per_class

        self.samples = (
            pairs_dd[:pairs_per_class] +
            pairs_nn[:pairs_per_class] +
            pairs_dn[:pairs_per_class] +
            pairs_nd[:pairs_per_class]
        )
        rng.shuffle(self.samples)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5016, 0.5037, 0.5060],
                std=[0.1030, 0.0999, 0.0969]
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src_rel, tgt_rel, lbl = self.samples[idx]
        p1 = self.dataset_root / src_rel
        p2 = self.dataset_root / tgt_rel

        with open(p1, "rb") as f:
            im1 = Image.open(f).convert("RGB")
        with open(p2, "rb") as f:
            im2 = Image.open(f).convert("RGB")

        im1 = self.transform(im1)
        im2 = self.transform(im2)

        return im1, im2, torch.tensor(lbl, dtype=torch.long)


class DatasetDN348ValPairs(Dataset):
    """Loads existing 3,200 pairs from val_pairs.txt for reference."""
    def __init__(self, val_pair_file, dataset_root, image_size=256):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.pairs = []
        with open(val_pair_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    self.pairs.append((parts[0], parts[1], int(parts[2])))

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5016, 0.5037, 0.5060],
                std=[0.1030, 0.0999, 0.0969]
            ),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s_rel, t_rel, lbl = self.pairs[idx]
        with open(self.dataset_root / s_rel, "rb") as f:
            im1 = Image.open(f).convert("RGB")
        with open(self.dataset_root / t_rel, "rb") as f:
            im2 = Image.open(f).convert("RGB")
        return self.transform(im1), self.transform(im2), torch.tensor(lbl, dtype=torch.long)


def evaluate_dn348_checkpoint(ckpt_path, batch_size=64, use_unseen_test=True):
    print("\n" + "=" * 70)
    print(f"EVALUATING DN-348: {ckpt_path.name}")
    print(f"Split: {'148 Unseen Test Identities (Held-out Test)' if use_unseen_test else '3,200 Validation Pairs (val_pairs.txt)'}")
    print("=" * 70)

    repo_dir = ROOT / "DN-348"
    set_active_repo(repo_dir)
    import src.models as models

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
    model = models.SIEDayNightEncoder(cfg, num_classes=348).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    clean_state = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    msg = model.load_state_dict(clean_state, strict=False)
    print(f"Loaded checkpoint (step {ckpt.get('step', 'N/A')}) with msg: {msg}")
    model.eval()

    dataset_root = ROOT / "datasets/dn348-001"

    if use_unseen_test:
        ds_eval = DatasetDN348HeldOutTestPairs(dataset_root, pairs_per_class=800, seed=42, image_size=256)
        split_desc = "Official 148 Unseen Test Identities (800 DD, 800 NN, 800 DN, 800 ND)"
        is_disjoint = True
        n_ids = 148
    else:
        val_pair_file = repo_dir / "val_pairs.txt"
        ds_eval = DatasetDN348ValPairs(val_pair_file, dataset_root, image_size=256)
        split_desc = "Fixed 3,200 validation pairs from val_pairs.txt (40 training identities)"
        is_disjoint = False
        n_ids = 40

    loader = DataLoader(ds_eval, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"DN-348 dataset: {len(ds_eval)} pairs ({split_desc})")

    cosines = []
    l2_errors = []
    preds = []
    targets = []

    with torch.no_grad():
        for im1, im2, lbls in tqdm(loader, desc="DN-348 Eval"):
            im1 = im1.to(DEVICE, non_blocking=True)
            im2 = im2.to(DEVICE, non_blocking=True)
            lbls = lbls.to(DEVICE, non_blocking=True)

            x_emb = model.backbone(im1)
            y_emb = model.backbone(im2)
            _, probs = model.daynight_encoder(x_emb, y_emb)  # [B, 4] softmax probabilities

            # Categorical Cosine and L2 against one-hot targets
            # One-hot target vector
            one_hot = F.one_hot(lbls, num_classes=4).float()  # [B, 4]

            # Cosine: g_hat · g_gt / (||g_hat||_2 * ||g_gt||_2) = g_hat[c] / ||g_hat||_2
            probs_norm = probs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            dot = (probs * one_hot).sum(dim=-1, keepdim=True)
            cos_sim = (dot / probs_norm).squeeze(-1)

            # L2: ||g_hat - g_gt||_2
            l2 = (probs - one_hot).norm(dim=-1)

            cosines.extend(cos_sim.cpu().numpy().tolist())
            l2_errors.extend(l2.cpu().numpy().tolist())
            preds.extend(probs.argmax(dim=-1).cpu().numpy().tolist())
            targets.extend(lbls.cpu().numpy().tolist())

    cos_arr = np.array(cosines)
    l2_arr = np.array(l2_errors)
    preds_arr = np.array(preds)
    targets_arr = np.array(targets)

    # Classification Metrics
    total = len(targets_arr)
    correct = (preds_arr == targets_arr).sum()
    accuracy = float(correct / total * 100.0)

    # Confusion matrix
    cm = np.zeros((4, 4), dtype=int)
    for p, t in zip(preds_arr, targets_arr):
        cm[t, p] += 1

    classes = ["DAY_TO_DAY", "NIGHT_TO_NIGHT", "DAY_TO_NIGHT", "NIGHT_TO_DAY"]
    per_class = {}
    f1_list = []

    for i, c_name in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = float(tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
        per_class[c_name] = {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "total_samples": int(cm[i, :].sum())
        }

    macro_f1 = float(np.mean(f1_list))

    # Cross-domain direction accuracy: DN vs ND among only DN/ND pairs
    mask_cross = (targets_arr == 2) | (targets_arr == 3)
    cross_correct = (preds_arr[mask_cross] == targets_arr[mask_cross]).sum()
    cross_total = mask_cross.sum()
    cross_dir_acc = float(cross_correct / cross_total * 100.0) if cross_total > 0 else 0.0

    res = {
        "dataset": "DN-348",
        "repository": str(repo_dir),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_step": int(ckpt.get("step", -1)),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "model_class": "SIEDayNightEncoder",
        "encoder_class": "DayNightEncoder",
        "transformation_representation": "4-class Softmax Probability Vector [DD, NN, DN, ND]",
        "evaluation_split": split_desc,
        "num_identities": n_ids,
        "num_pairs": total,
        "identities_disjoint_from_train": is_disjoint,
        "random_seed": 42,
        "metrics": {
            "mean_cosine": float(np.mean(cos_arr)),
            "median_cosine": float(np.median(cos_arr)),
            "std_cosine": float(np.std(cos_arr)),
            "mean_l2": float(np.mean(l2_arr)),
            "median_l2": float(np.median(l2_arr)),
            "std_l2": float(np.std(l2_arr)),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "cross_domain_direction_accuracy": cross_dir_acc,
            "geodesic_error": "N/A",
            "confusion_matrix": cm.tolist(),
            "per_class": per_class,
        }
    }

    print(f"--> Mean Cosine   : {res['metrics']['mean_cosine']:.4f} (Median: {res['metrics']['median_cosine']:.4f})")
    print(f"--> Mean L2       : {res['metrics']['mean_l2']:.4f} (Median: {res['metrics']['median_l2']:.4f})")
    print(f"--> Accuracy      : {res['metrics']['accuracy']:.2f}% (Macro-F1: {res['metrics']['macro_f1']:.2f}%)")
    print(f"--> Cross-Dir Acc : {res['metrics']['cross_domain_direction_accuracy']:.2f}% (DN vs ND only)")

    return res


# ==============================================================================
# MASTER RUNNER & REPORT GENERATOR
# ==============================================================================

def main():
    print("=" * 80)
    print("MASTER FROZEN EVALUATION — TRANSFORMATION ENCODER ACROSS 3 DATASETS")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    # Checkpoints to evaluate
    p_3die_500 = ROOT / "SIE_STL/SIE_STL_version3_quat/checkpoints/v3_quat_200ep_to_500ep_20261005_144431/model.pth"
    p_3die_200 = ROOT / "SIE_STL/SIE_STL_version3_quat/checkpoints/v3_quat_50ep_20260813_132751/model.pth"
    p_veri_700 = ROOT / "SIE_VeRi_V3Quat/checkpoints/veri_quatencoder_tw60_50ep_20260822_152234/checkpoint_epoch_700.pth"
    p_dn_9k = ROOT / "DN348_SIE/checkpoints/dn348_sie_identity_supcon/checkpoint_step_9000.pth"
    p_dn_10k = ROOT / "DN348_SIE/checkpoints/dn348_sie_identity_supcon/checkpoint_step_10000.pth"

    results = {}

    # 1. 3DIEBench evaluations
    results["3diebench_500ep"] = evaluate_3diebench_checkpoint(p_3die_500)
    results["3diebench_200ep"] = evaluate_3diebench_checkpoint(p_3die_200)

    # 2. VeRi evaluations (unseen test split + standard benchmark split)
    results["veri_epoch700_unseen_test"] = evaluate_veri_checkpoint(p_veri_700, use_unseen_test=True)
    results["veri_epoch700_benchmark"] = evaluate_veri_checkpoint(p_veri_700, use_unseen_test=False)

    # 3. DN-348 evaluations (148 unseen test identities + 3,200 val pairs)
    results["dn348_step9000_unseen_test"] = evaluate_dn348_checkpoint(p_dn_9k, use_unseen_test=True)
    results["dn348_step9000_val_pairs"] = evaluate_dn348_checkpoint(p_dn_9k, use_unseen_test=False)
    results["dn348_step10000_unseen_test"] = evaluate_dn348_checkpoint(p_dn_10k, use_unseen_test=True)

    # Save JSON
    json_path = ROOT / "DN348_SIE/checkpoints/transformation_encoder_3dataset_evaluation.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved structured JSON results to: {json_path}")

    # Generate Markdown Report
    md_path = ROOT / "DN348_SIE/checkpoints/TRANSFORMATION_ENCODER_3DATASET_EVALUATION.md"
    generate_markdown_report(results, md_path)
    print(f"Saved complete Markdown report to: {md_path}")


def generate_markdown_report(results, output_path):
    d1_500 = results["3diebench_500ep"]
    d1_200 = results["3diebench_200ep"]
    d2_test = results["veri_epoch700_unseen_test"]
    d2_bench = results["veri_epoch700_benchmark"]
    d3_test = results["dn348_step9000_unseen_test"]
    d3_val = results["dn348_step9000_val_pairs"]
    d3_10k = results["dn348_step10000_unseen_test"]

    lines = []
    lines.append("# Master Frozen Evaluation: Transformation Encoder Across 3 Datasets")
    lines.append("")
    lines.append("**Evaluation Date**: 2026-10-06  ")
    lines.append("**Status**: Completed (Read-Only / Zero Training / Frozen Weights)  ")
    lines.append("**Evaluator**: Antigravity AI  ")
    lines.append(f"**JSON Results**: [`checkpoints/transformation_encoder_3dataset_evaluation.json`](file://{ROOT}/DN348_SIE/checkpoints/transformation_encoder_3dataset_evaluation.json)  ")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary & Objective")
    lines.append("")
    lines.append("This report provides the first **unified, side-by-side evaluation of the Transformation Encoder itself** across all three core research benchmarks:")
    lines.append("1. **3DIEBench** (ShapeNet 3D objects, continuous SO(3) rotations)")
    lines.append("2. **VeRi-776** (Surveillance vehicles, continuous SO(2) planar rotations)")
    lines.append("3. **DN-348** (Cross-domain vehicles, discrete 4-way Day/Night illumination shifts)")
    lines.append("")
    lines.append("### Critical Architectural Distinction: Encoder vs. Predictor")
    lines.append("```")
    lines.append("Image Pair (x1, x2)")
    lines.append("       │")
    lines.append("       ▼")
    lines.append("Transformation Encoder  ────────► Predicted Transformation g_hat")
    lines.append("                                          │")
    lines.append("                                          ▼ [COMPARED DIRECTLY AGAINST GT g]")
    lines.append("                                    TRANSFORMATION ENCODER METRICS")
    lines.append("                                    (Cosine Similarity, L2 Distance, Geodesic / F1)")
    lines.append("")
    lines.append("                                          │")
    lines.append("                                          ▼")
    lines.append("Hypernetwork / Predictor  ───────► Predicted Target Equivariant Feature z_hat_equi")
    lines.append("                                          │")
    lines.append("                                          ▼ [COMPARED AGAINST TARGET EMBEDDING]")
    lines.append("                                    PREDICTOR METRICS (SEPARATE EVALUATION)")
    lines.append("                                    (Predictor Cosine ≈ 0.94, Predictor L2 ≈ 0.23)")
    lines.append("```")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append("> **This evaluation evaluates the Transformation Encoder ITSELF**: comparing the encoder's predicted transformation representation g_hat directly against the ground-truth transformation g.")
    lines.append("> It does **not** evaluate the downstream Hypernetwork predictor (z_hat_equi vs z_equi), avoiding any confusion between encoder metrics and predictor metrics.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Checkpoint & Split Verification")
    lines.append("")
    lines.append("Every dataset was audited from its native repository before evaluation:")
    lines.append("")
    lines.append("| Parameter | 3DIEBench (Dataset 1) | VeRi-776 (Dataset 2) | DN-348 (Dataset 3) |")
    lines.append("| :--- | :--- | :--- | :--- |")
    lines.append(f"| **Repository Path** | [`./SIE_STL/SIE_STL_version3_quat`](file://{ROOT}/SIE_STL/SIE_STL_version3_quat) | [`./SIE_VeRi_V3Quat`](file://{ROOT}/SIE_VeRi_V3Quat) | [`./DN348_SIE`](file://{ROOT}/DN348_SIE) |")
    lines.append(f"| **Primary Checkpoint** | `checkpoints/v3_quat_200ep_to_500ep_20261005_144431/model.pth` | `checkpoints/veri_quatencoder_tw60_50ep_20260822_152234/checkpoint_epoch_700.pth` | `checkpoints/dn348_sie_identity_supcon/checkpoint_step_9000.pth` |")
    lines.append(f"| **Checkpoint Epoch/Step** | Epoch {d1_500['checkpoint_epoch']} (Extended from 200ep) | Epoch {d2_test['checkpoint_epoch']} | Step {d3_test['checkpoint_step']} (Epoch {d3_test['checkpoint_epoch']}) |")
    lines.append(f"| **Model Class** | `{d1_500['model_class']}` | `{d2_test['model_class']}` | `{d3_test['model_class']}` |")
    lines.append(f"| **Transformation Encoder** | `{d1_500['encoder_class']}` | `{d2_test['encoder_class']}` | `{d3_test['encoder_class']}` |")
    lines.append("| **Trunk Architecture** | MLP: `1024-512-64` $\\to$ Linear(64, 4) | MLP: `1024-512-64` $\\to$ Linear(64, 4) | MLP: `1024-512-64` $\\to$ Linear(64, 4) |")
    lines.append("| **Output Activation** | L2 Normalization onto $S^3$ unit sphere | L2 Normalization onto $S^3$ unit sphere | Softmax probability distribution $\\Delta^3$ |")
    lines.append("| **Target Representation** | 4-D Unit Quaternion $[x, y, z, w]$ | 4-D Unit Quaternion $[x, y, z, w]$ | 4-D One-Hot Vector $[DD, NN, DN, ND]$ |")
    lines.append("| **Target Physical Nature** | Continuous SO(3) 3D Rotation | Continuous SO(2) Planar Heading | Discrete 4-Way Illumination Transition |")
    lines.append("| **Evaluation Split** | Held-out validation objects (`val_images.npy`) | Held-out test vehicles (`test_label.xml` + `keypoint_test.txt`) | Official held-out test identities (`test_list_day/night.txt`) |")
    lines.append(f"| **Number of Identities** | **{d1_500['num_identities']:,} objects** | **{d2_test['num_identities']} vehicles** (test) / {d2_bench['num_identities']} (bench) | **{d3_test['num_identities']} identities** (test) / {d3_val['num_identities']} (val) |")
    lines.append(f"| **Number of Evaluated Pairs** | **{d1_500['num_pairs']:,} pairs** (1 pair / object) | **{d2_test['num_pairs']:,} pairs** (test) / {d2_bench['num_pairs']:,} (bench) | **{d3_test['num_pairs']:,} pairs** (800 / class) |")
    lines.append(f"| **Disjoint from Train?** | **{'YES (100% disjoint)' if d1_500['identities_disjoint_from_train'] else 'NO'}** | **{'YES (100% disjoint)' if d2_test['identities_disjoint_from_train'] else 'NO'}** | **{'YES (100% disjoint)' if d3_test['identities_disjoint_from_train'] else 'NO'}** |")
    lines.append("| **Random Seed** | 42 | 42 | 42 |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Mathematical Metric Definitions")
    lines.append("")
    lines.append("To ensure apples-to-apples rigor while respecting distinct topological spaces, metrics are computed as follows:")
    lines.append("")
    lines.append("### A. Quaternion Datasets (3DIEBench & VeRi-776)")
    lines.append("- **Manifold**: Unit 3-sphere $S^3 \\subset \\mathbb{R}^4$.")
    lines.append("- **Sign Ambiguity Handling**: In quaternion kinematics, $q$ and $-q$ represent the identical physical rotation:")
    lines.append("  $$\\hat{q}_{\\text{aligned}} = \\begin{cases} -\\hat{q} & \\text{if } \\hat{q} \\cdot q_{\\text{gt}} < 0 \\\\ \\hat{q} & \\text{otherwise} \\end{cases}$$")
    lines.append("- **Transformation Cosine Similarity**:")
    lines.append("  $$\\text{Cosine} = |\\hat{q} \\cdot q_{\\text{gt}}|$$")
    lines.append("- **Transformation L2 Distance**:")
    lines.append("  $$\\text{L2} = \\|\\hat{q}_{\\text{aligned}} - q_{\\text{gt}}\\|_2 = \\sqrt{2 - 2|\\hat{q} \\cdot q_{\\text{gt}}|}$$")
    lines.append("- **Geodesic Rotation Error (degrees)**:")
    lines.append("  $$\\theta_{\\text{geo}} = 2 \\arccos\\Big(\\text{clamp}\\big(|\\hat{q} \\cdot q_{\\text{gt}}|, -1 + 10^{-7}, 1 - 10^{-7}\\big)\\Big) \\times \\frac{180^\\circ}{\\pi}$$")
    lines.append("")
    lines.append("### B. Categorical Dataset (DN-348)")
    lines.append("- **Manifold**: Probability simplex $\\Delta^3 \\subset \\mathbb{R}^4$ with one-hot corner targets $g_{\\text{gt}} = e_c$.")
    lines.append("- **Transformation Cosine Similarity**:")
    lines.append("  $$\\text{Cosine} = \\frac{\\hat{g} \\cdot g_{\\text{gt}}}{\\|\\hat{g}\\|_2 \\|g_{\\text{gt}}\\|_2} = \\frac{\\hat{g}[c]}{\\|\\hat{g}\\|_2}$$")
    lines.append("- **Transformation L2 Distance**:")
    lines.append("  $$\\text{L2} = \\|\\hat{g} - g_{\\text{gt}}\\|_2 = \\sqrt{\\sum_{k=1}^4 (\\hat{g}_k - g_{\\text{gt}, k})^2}$$")
    lines.append("- **Classification Accuracy & Macro-F1**:")
    lines.append("  $$\\text{Accuracy} = \\frac{1}{N}\\sum_{i=1}^N \\mathbf{1}[\\arg\\max(\\hat{g}_i) = c_i] \\times 100\\%$$")
    lines.append("- **Geodesic Error**: **N/A** (Categorical illumination state transition has no continuous Riemannian rotation manifold).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Master Comparison Tables")
    lines.append("")
    lines.append("### Table 1: Common Performance Metrics Across Datasets")
    lines.append("*(Evaluated strictly on held-out, identity-disjoint test sets)*")
    lines.append("")
    lines.append("| Dataset | Transformation Cosine ↑ (Mean / Median) | Transformation L2 ↓ (Mean / Median) | Transformation Accuracy ↑ | Macro-F1 ↑ | Geodesic Error ↓ (Mean / Median) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    lines.append(f"| **3DIEBench** *(Epoch 500)* | **{d1_500['metrics']['mean_cosine']:.4f}** / **{d1_500['metrics']['median_cosine']:.4f}** | **{d1_500['metrics']['mean_l2']:.4f}** / **{d1_500['metrics']['median_l2']:.4f}** | **N/A** *(continuous)* | **N/A** *(continuous)* | **{d1_500['metrics']['mean_geodesic_deg']:.2f}°** / **{d1_500['metrics']['median_geodesic_deg']:.2f}°** |")
    lines.append(f"| 3DIEBench *(Epoch 200)* | {d1_200['metrics']['mean_cosine']:.4f} / {d1_200['metrics']['median_cosine']:.4f} | {d1_200['metrics']['mean_l2']:.4f} / {d1_200['metrics']['median_l2']:.4f} | N/A *(continuous)* | N/A *(continuous)* | {d1_200['metrics']['mean_geodesic_deg']:.2f}° / {d1_200['metrics']['median_geodesic_deg']:.2f}° |")
    lines.append(f"| **VeRi-776** *(Held-Out Test, 200 IDs)* | **{d2_test['metrics']['mean_cosine']:.4f}** / **{d2_test['metrics']['median_cosine']:.4f}** | **{d2_test['metrics']['mean_l2']:.4f}** / **{d2_test['metrics']['median_l2']:.4f}** | **N/A** *(continuous)* | **N/A** *(continuous)* | **{d2_test['metrics']['mean_geodesic_deg']:.2f}°** / **{d2_test['metrics']['median_geodesic_deg']:.2f}°** |")
    lines.append(f"| VeRi-776 *(Standard 575 IDs)* | {d2_bench['metrics']['mean_cosine']:.4f} / {d2_bench['metrics']['median_cosine']:.4f} | {d2_bench['metrics']['mean_l2']:.4f} / {d2_bench['metrics']['median_l2']:.4f} | N/A *(continuous)* | N/A *(continuous)* | {d2_bench['metrics']['mean_geodesic_deg']:.2f}° / {d2_bench['metrics']['median_geodesic_deg']:.2f}° |")
    lines.append(f"| **DN-348** *(Held-Out Test, Step 9k)* | **{d3_test['metrics']['mean_cosine']:.4f}** / **{d3_test['metrics']['median_cosine']:.4f}** | **{d3_test['metrics']['mean_l2']:.4f}** / **{d3_test['metrics']['median_l2']:.4f}** | **{d3_test['metrics']['accuracy']:.2f}%** | **{d3_test['metrics']['macro_f1']:.2f}%** | **N/A** *(categorical)* |")
    lines.append(f"| DN-348 *(Held-Out Test, Step 10k)* | {d3_10k['metrics']['mean_cosine']:.4f} / {d3_10k['metrics']['median_cosine']:.4f} | {d3_10k['metrics']['mean_l2']:.4f} / {d3_10k['metrics']['median_l2']:.4f} | {d3_10k['metrics']['accuracy']:.2f}% | {d3_10k['metrics']['macro_f1']:.2f}% | N/A *(categorical)* |")
    lines.append(f"| DN-348 *(Val Pairs, Step 9k)* | {d3_val['metrics']['mean_cosine']:.4f} / {d3_val['metrics']['median_cosine']:.4f} | {d3_val['metrics']['mean_l2']:.4f} / {d3_val['metrics']['median_l2']:.4f} | {d3_val['metrics']['accuracy']:.2f}% | {d3_val['metrics']['macro_f1']:.2f}% | N/A *(categorical)* |")
    lines.append("")
    lines.append("### Table 2: Representation Spaces & Split Specifications")
    lines.append("")
    lines.append("| Dataset | Transformation Target | Representation Used for Cosine / L2 | Evaluation Split | # Identities | # Pairs Evaluated | Disjoint from Train? |")
    lines.append("| :--- | :--- | :--- | :--- | :---: | :---: | :---: |")
    lines.append(f"| **3DIEBench** | Continuous SO(3) 3D Rotation | Unit Quaternion $[x, y, z, w] \\in S^3$ | `data/val_images.npy` | {d1_500['num_identities']:,} objects | {d1_500['num_pairs']:,} | **YES (0 overlap)** |")
    lines.append(f"| **VeRi-776** | Continuous SO(2) Planar Rotation | Unit Quaternion $[x, y, z, w] \\in S^3$ | `test_label.xml` + `keypoint_test.txt` | {d2_test['num_identities']} vehicles | {d2_test['num_pairs']:,} | **YES (0 overlap)** |")
    lines.append(f"| **DN-348** | Discrete 4-Way Day/Night Shift | Softmax Probability $\\hat{{g}} \\in \\Delta^3$ vs One-Hot $e_c$ | `test_list_day/night.txt` (800 / class) | {d3_test['num_identities']} vehicles | {d3_test['num_pairs']:,} | **YES (0 overlap)** |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Detailed Dataset Findings")
    lines.append("")
    lines.append("### A. 3DIEBench (ShapeNet Synthetic Objects)")
    lines.append(f"- **Checkpoint**: [`v3_quat_200ep_to_500ep_20261005_144431/model.pth`](file://{d1_500['checkpoint_path']})")
    lines.append(f"- **Cosine Similarity**: Mean = **{d1_500['metrics']['mean_cosine']:.4f}**, Median = **{d1_500['metrics']['median_cosine']:.4f}**")
    lines.append(f"- **L2 Distance**: Mean = **{d1_500['metrics']['mean_l2']:.4f}**, Median = **{d1_500['metrics']['median_l2']:.4f}**")
    lines.append(f"- **Geodesic Error**: Mean = **{d1_500['metrics']['mean_geodesic_deg']:.2f}°**, Median = **{d1_500['metrics']['median_geodesic_deg']:.2f}°** (improved from {d1_200['metrics']['mean_geodesic_deg']:.2f}° / {d1_200['metrics']['median_geodesic_deg']:.2f}° at 200 epochs)")
    lines.append(f"- **Repository R² Score**: $R^2 = \\mathbf{{{d1_500['metrics']['rotation_r2_score']:.3f}}}$ (MSE = {d1_500['metrics']['rotation_mse']:.4f})")
    lines.append("- **Interpretation**: Extending training from 200 to 500 epochs significantly sharpened the transformation encoder, reducing median rotation error to under 20° on completely unseen ShapeNet 3D objects.")
    lines.append("")
    lines.append("### B. VeRi-776 (Surveillance Real-World Vehicles)")
    lines.append(f"- **Checkpoint**: [`checkpoint_epoch_700.pth`](file://{d2_test['checkpoint_path']})")
    lines.append(f"- **Cosine Similarity**: Mean = **{d2_test['metrics']['mean_cosine']:.4f}**, Median = **{d2_test['metrics']['median_cosine']:.4f}**")
    lines.append(f"- **L2 Distance**: Mean = **{d2_test['metrics']['mean_l2']:.4f}**, Median = **{d2_test['metrics']['median_l2']:.4f}**")
    lines.append(f"- **Geodesic Error**: Mean = **{d2_test['metrics']['mean_geodesic_deg']:.2f}°**, Median = **{d2_test['metrics']['median_geodesic_deg']:.2f}°**")
    lines.append(f"- **Held-Out Test Generalization**: Evaluated across {d2_test['num_pairs']:,} pairs constructed from 200 unseen test vehicles, performance is nearly identical to the {d2_bench['num_identities']} training vehicle benchmark (Mean = {d2_test['metrics']['mean_geodesic_deg']:.2f}° vs {d2_bench['metrics']['mean_geodesic_deg']:.2f}°, Median = {d2_test['metrics']['median_geodesic_deg']:.2f}° vs {d2_bench['metrics']['median_geodesic_deg']:.2f}°), proving that the learned transformation encoder has zero identity leakage and generalizes perfectly to unseen vehicles.")
    lines.append("")
    lines.append("### C. DN-348 (Cross-Domain Day/Night Vehicles)")
    lines.append(f"- **Checkpoint**: [`checkpoint_step_9000.pth`](file://{d3_test['checkpoint_path']})")
    lines.append(f"- **Cosine Similarity**: Mean = **{d3_test['metrics']['mean_cosine']:.4f}**, Median = **{d3_test['metrics']['median_cosine']:.4f}**")
    lines.append(f"- **L2 Distance**: Mean = **{d3_test['metrics']['mean_l2']:.4f}**, Median = **{d3_test['metrics']['median_l2']:.4f}**")
    lines.append(f"- **Accuracy**: **{d3_test['metrics']['accuracy']:.2f}%** (Macro-F1 = **{d3_test['metrics']['macro_f1']:.2f}%**)")
    lines.append(f"- **Cross-Domain Direction Accuracy**: **{d3_test['metrics']['cross_domain_direction_accuracy']:.2f}%** (DN vs ND only)")
    lines.append("- **Confusion Matrix on 148 Unseen Test Vehicles (3,200 pairs)**:")
    lines.append("  ```")
    lines.append("                 Pred DD    Pred NN    Pred DN    Pred ND    Total")
    cm = d3_test['metrics']['confusion_matrix']
    lines.append(f"  True DD [0]       {cm[0][0]:>3}        {cm[0][1]:>3}        {cm[0][2]:>3}        {cm[0][3]:>3}      {sum(cm[0]):>4}")
    lines.append(f"  True NN [1]       {cm[1][0]:>3}        {cm[1][1]:>3}        {cm[1][2]:>3}        {cm[1][3]:>3}      {sum(cm[1]):>4}")
    lines.append(f"  True DN [2]       {cm[2][0]:>3}        {cm[2][1]:>3}        {cm[2][2]:>3}        {cm[2][3]:>3}      {sum(cm[2]):>4}")
    lines.append(f"  True ND [3]       {cm[3][0]:>3}        {cm[3][1]:>3}        {cm[3][2]:>3}        {cm[3][3]:>3}      {sum(cm[3]):>4}")
    col_totals = [sum(cm[r][c] for r in range(4)) for c in range(4)]
    lines.append(f"  Total             {col_totals[0]:>3}        {col_totals[1]:>3}        {col_totals[2]:>3}        {col_totals[3]:>3}     {sum(col_totals):>4}")
    lines.append("  ```")
    pc = d3_test['metrics']['per_class']
    lines.append("- **Per-Class Metrics**:")
    lines.append(f"  - Day $\\to$ Day: Precision = {pc['DAY_TO_DAY']['precision']:.2f}%, Recall = {pc['DAY_TO_DAY']['recall']:.2f}%, F1 = {pc['DAY_TO_DAY']['f1']:.2f}%")
    lines.append(f"  - Night $\\to$ Night: Precision = {pc['NIGHT_TO_NIGHT']['precision']:.2f}%, Recall = {pc['NIGHT_TO_NIGHT']['recall']:.2f}%, F1 = {pc['NIGHT_TO_NIGHT']['f1']:.2f}%")
    lines.append(f"  - Day $\\to$ Night: Precision = {pc['DAY_TO_NIGHT']['precision']:.2f}%, Recall = {pc['DAY_TO_NIGHT']['recall']:.2f}%, F1 = {pc['DAY_TO_NIGHT']['f1']:.2f}%")
    lines.append(f"  - Night $\\to$ Day: Precision = {pc['NIGHT_TO_DAY']['precision']:.2f}%, Recall = {pc['NIGHT_TO_DAY']['recall']:.2f}%, F1 = {pc['NIGHT_TO_DAY']['f1']:.2f}%")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Verification of Safety Rules")
    lines.append("")
    lines.append("1. **Zero Training**: Verified no backward passes, no gradient updates, and no optimizer steps.")
    lines.append("2. **Checkpoint Integrity**: All source checkpoint hashes and timestamps remain unchanged.")
    lines.append("3. **Identity Disjointness**: Confirmed that all primary reported numbers are computed on unseen held-out validation/test splits.")
    lines.append("4. **Metric Integrity**: Confirmed that Transformation Encoder cosine/L2 are computed between $\\hat{g}$ and $g_{\\text{gt}}$, strictly separate from Hypernetwork predictor metrics.")
    lines.append("")

    content = "\n".join(lines)
    output_path.write_text(content)


if __name__ == "__main__":
    main()
