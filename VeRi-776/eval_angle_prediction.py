# eval_angle_prediction.py

import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision
import torch.nn.functional as F
from PIL import Image
import torch.nn as nn
from torchvision import transforms

import os.path
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from pathlib import Path
import json
import sys
import math
from scipy.spatial.transform import Rotation as R
import copy
import csv
import random

import argparse
import src.resnet as resnet


parser = argparse.ArgumentParser()

parser.add_argument("--arch", type=str, default="resnet18")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--experience", type=str,
                    choices=["quat", "euler"], default="quat")
parser.add_argument("--deep-end", action="store_true")
parser.add_argument("--equi-dims", type=int, default=512)
parser.add_argument("--inv-part", action="store_true")
parser.add_argument("--dataset-root", type=Path, required=True)

# Experience loading
parser.add_argument("--weights-file", type=str, default="./resnet50.pth")
parser.add_argument("--supervised", action="store_true")

# Optim
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=0)

# Checkpoints / logs
parser.add_argument("--exp-dir", type=Path, default="")
parser.add_argument("--root-log-dir", type=Path, default="eval_logs/tensorboard")
parser.add_argument("--log-freq-time", type=int, default=10)

# Running
parser.add_argument("--num-workers", type=int, default=10)

args = parser.parse_args()


# ============================================================
# MODEL
# ============================================================

class Model(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.backbone, _ = resnet.__dict__[args.arch](
            zero_init_residual=True
        )

        self.equi_dims = args.equi_dims
        self.out_dim = 4 if args.experience == "quat" else 3

        self.inv = args.inv_part

        self.in_dims = (
            self.equi_dims
            if not self.inv
            else 512 - self.equi_dims
        )

        if args.deep_end:

            self.head = nn.Sequential(
                nn.Linear(self.in_dims * 2, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, self.out_dim),
            )

        else:

            self.head = nn.Linear(
                self.in_dims * 2,
                self.out_dim
            )

    def forward(self, x, y):

        if not self.inv:

            out_x = self.backbone(x)[..., -self.equi_dims:]
            out_y = self.backbone(y)[..., -self.equi_dims:]

        else:

            out_x = self.backbone(x)[..., :-self.equi_dims]
            out_y = self.backbone(y)[..., :-self.equi_dims]

        concat = torch.cat([out_x, out_y], axis=1)

        out = self.head(concat)

        return out


# ============================================================
# FIXED VeRi PAIR DATASET
# ============================================================

class DatasetVeRiR2(Dataset):

    def __init__(
        self,
        dataset_root,
        transform=None,
        max_per_transition=1500,
        seed=42,
    ):

        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.transform = transform

        self.xml_file = (
            self.dataset_root /
            "VeRi" /
            "train_label.xml"
        )

        self.image_root = (
            self.dataset_root /
            "VeRi" /
            "image_train"
        )

        self.keypoint_file = (
            self.dataset_root /
            "VehicleReIDKeyPointData" /
            "keypoint_train.txt"
        )

        # --------------------------------------------------------
        # Read XML
        # --------------------------------------------------------

        with open(
            self.xml_file,
            "r",
            encoding="gb2312"
        ) as f:

            xml_text = f.read()

        if xml_text.startswith("<?xml"):

            xml_text = xml_text.split("?>", 1)[1]

        root = __import__("xml.etree.ElementTree",
                          fromlist=["ElementTree"]).fromstring(
            xml_text
        )

        items = root.find("Items")

        if items is None:

            raise RuntimeError(
                "Items section not found in train_label.xml"
            )

        # --------------------------------------------------------
        # Orientation annotations
        # --------------------------------------------------------

        self.image_to_orientation = {}

        with open(self.keypoint_file, "r") as f:

            for line in f:

                tokens = line.strip().split()

                if not tokens:
                    continue

                image_name = Path(tokens[0]).name
                orientation = int(tokens[-1])

                self.image_to_orientation[image_name] = orientation

        # --------------------------------------------------------
        # Vehicle -> images
        # --------------------------------------------------------

        vehicle_to_images = {}

        for item in items:

            vehicle_id = item.attrib["vehicleID"]
            image_name = item.attrib["imageName"]

            if image_name not in self.image_to_orientation:
                continue

            if vehicle_id not in vehicle_to_images:
                vehicle_to_images[vehicle_id] = []

            vehicle_to_images[vehicle_id].append(image_name)

        # Keep vehicles with >= 2 annotated images

        self.vehicle_to_images = {

            vehicle_id: images

            for vehicle_id, images
            in vehicle_to_images.items()

            if len(images) >= 2
        }

        self.vehicle_ids = sorted(
            self.vehicle_to_images.keys()
        )

        print(
            "\nR2 Dataset"
        )

        print(
            f"Vehicles : {len(self.vehicle_ids)}"
        )

        print(
            f"Annotated images : "
            f"{sum(len(v) for v in self.vehicle_to_images.values())}"
        )

        # --------------------------------------------------------
        # Build ALL pairs grouped by orientation transition
        # --------------------------------------------------------

        pairs_by_transition = {}

        for vehicle_id in self.vehicle_ids:

            images = self.vehicle_to_images[vehicle_id]

            for i in range(len(images)):

                for j in range(i + 1, len(images)):

                    img1 = images[i]
                    img2 = images[j]

                    ori1 = self.image_to_orientation[img1]
                    ori2 = self.image_to_orientation[img2]

                    transition = (ori1, ori2)

                    if transition not in pairs_by_transition:
                        pairs_by_transition[transition] = []

                    pairs_by_transition[transition].append(
                        (
                            img1,
                            img2,
                            vehicle_id
                        )
                    )

        # --------------------------------------------------------
        # Balanced fixed evaluation set
        # --------------------------------------------------------

        rng = random.Random(seed)

        self.samples = []

        print("\nTransition Counts")

        for transition in sorted(pairs_by_transition.keys()):

            pairs = pairs_by_transition[transition]

            original_count = len(pairs)

            if len(pairs) > max_per_transition:

                pairs = rng.sample(
                    pairs,
                    max_per_transition
                )

            self.samples.extend(pairs)

            print(
                f"{transition}: "
                f"{original_count} -> {len(pairs)}"
            )

        print(
            f"\nFinal fixed R2 pairs : {len(self.samples)}"
        )

    # --------------------------------------------------------
    # Image loader
    # --------------------------------------------------------

    def get_img(self, image_name):

        image_path = self.image_root / image_name

        with open(image_path, "rb") as f:

            img = Image.open(f)
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img

    # --------------------------------------------------------
    # Orientation -> quaternion
    # --------------------------------------------------------

    def orientation_to_quaternion(self, ori1, ori2):

        LABEL_TO_ANGLE = {

            0: 0.0,
            6: 45.0,
            5: 90.0,
            7: 135.0,
            1: 180.0,
            4: 225.0,
            2: 270.0,
            3: 315.0,
        }

        angle1 = LABEL_TO_ANGLE[int(ori1)]
        angle2 = LABEL_TO_ANGLE[int(ori2)]

        relative_angle = angle2 - angle1

        while relative_angle > 180:
            relative_angle -= 360

        while relative_angle < -180:
            relative_angle += 360

        rotation = R.from_euler(
            "z",
            relative_angle,
            degrees=True,
        )

        quaternion = rotation.as_quat().astype(
            np.float32
        )

        return quaternion

    # --------------------------------------------------------
    # Item
    # --------------------------------------------------------

    def __getitem__(self, index):

        img1_name, img2_name, vehicle_id = self.samples[index]

        img1 = self.get_img(img1_name)
        img2 = self.get_img(img2_name)

        ori1 = self.image_to_orientation[img1_name]
        ori2 = self.image_to_orientation[img2_name]

        quaternion = self.orientation_to_quaternion(
            ori1,
            ori2
        )

        return (
            img1,
            img2,
            torch.FloatTensor(quaternion)
        )

    def __len__(self):

        return len(self.samples)


# ============================================================
# NORMALIZATION
# ============================================================

normalize = transforms.Normalize(
    mean=[0.5016, 0.5037, 0.5060],
    std=[0.1030, 0.0999, 0.0969]
)


# ============================================================
# R2
# ============================================================

def r2_score(output, target):

    target_mean = torch.mean(target)

    ss_tot = torch.sum(
        (target - target_mean) ** 2
    )

    ss_res = torch.sum(
        (target - output) ** 2
    )

    return 1 - ss_res / ss_tot


# ============================================================
# LOAD BACKBONE
# ============================================================

def load_from_state_dict(
    model,
    state_dict,
    prefix,
    new_suffix
):

    state_dict = copy.deepcopy(state_dict)

    state_dict = {

        k.replace(prefix, new_suffix): v

        for k, v in state_dict.items()

        if k.startswith(prefix)
    }

    for k, v in model.state_dict().items():

        if k not in state_dict:

            print(
                f'key "{k}" could not be found'
            )

        elif state_dict[k].shape != v.shape:

            print(
                f'key "{k}" shape mismatch'
            )

            state_dict[k] = v

    msg = model.load_state_dict(
        state_dict,
        strict=False
    )

    print(
        "Load pretrained model:",
        msg
    )


# ============================================================
# INIT
# ============================================================

args.exp_dir.mkdir(
    parents=True,
    exist_ok=True
)

args.root_log_dir.mkdir(
    parents=True,
    exist_ok=True
)

dict_args = deepcopy(vars(args))

for key, value in dict_args.items():

    if isinstance(value, Path):

        dict_args[key] = str(value)

with open(
    args.exp_dir / "params.json",
    "w"
) as f:

    json.dump(
        dict_args,
        f
    )

if str(args.exp_dir)[-1] == "/":

    exp_name = str(args.exp_dir)[:-1].split("/")[-1]

else:

    exp_name = str(args.exp_dir).split("/")[-1]

logdir = args.root_log_dir / exp_name

writer = SummaryWriter(
    log_dir=logdir
)

csv_file = args.exp_dir / "metrics.csv"

with open(
    csv_file,
    "w",
    newline=""
) as f:

    csv_writer = csv.writer(f)

    csv_writer.writerow([
        "Epoch",
        "Train_Loss",
        "Train_R2",
        "Val_Loss",
        "Val_R2"
    ])

print(" ".join(sys.argv))


# ============================================================
# DATA
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

ds_train = DatasetVeRiR2(
    args.dataset_root,
    transform=transform
)

ds_val = DatasetVeRiR2(
    args.dataset_root,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    ds_train,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers
)

val_loader = torch.utils.data.DataLoader(
    ds_val,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers
)

print(
    f"\nTrain batches per epoch : {len(train_loader)}"
)

print(
    f"Validation batches       : {len(val_loader)}"
)


# ============================================================
# MODEL
# ============================================================

net = Model(args)

net = net.to(args.device)

optimizer = torch.optim.Adam(
    net.parameters(),
    lr=args.lr,
    weight_decay=args.wd
)

# ------------------------------------------------------------
# Load frozen backbone
# ------------------------------------------------------------

ckpt = torch.load(
    args.weights_file,
    map_location="cpu"
)

if args.supervised:

    load_from_state_dict(
        net.backbone,
        ckpt,
        prefix="backbone.",
        new_suffix=""
    )

else:

    if "final_weights" in args.weights_file:

        load_from_state_dict(
            net.backbone,
            ckpt,
            prefix="",
            new_suffix=""
        )

    else:

        load_from_state_dict(
            net.backbone,
            ckpt["model"],
            prefix="module.backbone.",
            new_suffix=""
        )

for param in net.backbone.parameters():

    param.requires_grad = False

print("\nBackbone frozen.")
print("Training only R2 probe.")


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(args.epochs):

    net.train()

    train_loss_sum = 0.0
    train_r2_sum = 0.0
    train_count = 0

    for step, (
        inputs_1,
        inputs_2,
        latents
    ) in enumerate(train_loader):

        inputs_1 = inputs_1.to(args.device)
        inputs_2 = inputs_2.to(args.device)
        latents = latents.to(args.device)

        optimizer.zero_grad()

        outputs = net(
            inputs_1,
            inputs_2
        )

        loss = F.mse_loss(
            outputs,
            latents
        )

        r2 = r2_score(
            outputs,
            latents
        )

        loss.backward()

        optimizer.step()

        batch_size = inputs_1.shape[0]

        train_loss_sum += loss.item() * batch_size
        train_r2_sum += r2.item() * batch_size
        train_count += batch_size

    train_loss = train_loss_sum / train_count
    train_r2 = train_r2_sum / train_count

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    net.eval()

    total_latents = []
    total_preds = []

    val_loss_sum = 0.0
    val_count = 0

    with torch.no_grad():

        for (
            inputs_1,
            inputs_2,
            latents
        ) in val_loader:

            inputs_1 = inputs_1.to(args.device)
            inputs_2 = inputs_2.to(args.device)
            latents = latents.to(args.device)

            outputs = net(
                inputs_1,
                inputs_2
            )

            loss = F.mse_loss(
                outputs,
                latents
            )

            batch_size = inputs_1.shape[0]

            val_loss_sum += loss.item() * batch_size
            val_count += batch_size

            total_latents.append(
                latents.cpu()
            )

            total_preds.append(
                outputs.cpu()
            )

    total_latents = torch.cat(
        total_latents,
        dim=0
    )

    total_preds = torch.cat(
        total_preds,
        dim=0
    )

    val_loss = val_loss_sum / val_count

    val_r2 = r2_score(
        total_preds,
        total_latents
    ).item()

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    writer.add_scalar(
        "Loss/train",
        train_loss,
        epoch
    )

    writer.add_scalar(
        "Metrics/train_R2",
        train_r2,
        epoch
    )

    writer.add_scalar(
        "Loss/val",
        val_loss,
        epoch
    )

    writer.add_scalar(
        "Metrics/val_R2",
        val_r2,
        epoch
    )

    writer.flush()

    print(
        f"[Epoch {epoch}] "
        f"Train Loss: {train_loss:.6f}, "
        f"Train R2: {train_r2:.4f}, "
        f"Val Loss: {val_loss:.6f}, "
        f"Val R2: {val_r2:.4f}"
    )

    with open(
        csv_file,
        "a",
        newline=""
    ) as f:

        csv_writer = csv.writer(f)

        csv_writer.writerow([
            epoch,
            train_loss,
            train_r2,
            val_loss,
            val_r2
        ])


print("\nR2 evaluation finished.")
print(
    f"Logs saved in: {args.exp_dir}"
)
