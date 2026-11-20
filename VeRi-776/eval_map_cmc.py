import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import src.models as models


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="VeRi-776 mAP / CMC evaluation using frozen SIE invariant features"
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
    type=str,
    default="cuda",
)

parser.add_argument(
    "--output",
    type=Path,
    default=None,
)

args = parser.parse_args()


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    args.device if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("VeRi-776 mAP / CMC Evaluation")
print("=" * 70)
print("Device     :", device)
print("Dataset    :", args.dataset_root)
print("Checkpoint :", args.checkpoint)


# ============================================================
# LOAD TRAINING PARAMS
# ============================================================

if args.params is None:
    args.params = args.checkpoint.parent / "params.json"

if not args.params.exists():
    raise FileNotFoundError(
        f"Could not find params.json at {args.params}"
    )

with open(args.params, "r") as f:
    train_args = json.load(f)


class Config:
    pass


cfg = Config()

for k, v in train_args.items():
    setattr(cfg, k.replace("-", "_"), v)

print("\nModel configuration:")
print("  experience       :", cfg.experience)
print("  arch             :", cfg.arch)
print("  equi             :", cfg.equi)
print("  mlp              :", cfg.mlp)
print("  predictor_type   :", cfg.predictor_type)
print("  resolution       :", cfg.resolution)


# ============================================================
# NORMALIZATION
# ============================================================

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


# ============================================================
# VEri TEST METADATA
# ============================================================

class VeRiTestMetadata:

    def __init__(self, dataset_root):

        self.dataset_root = Path(dataset_root)

        self.veri_root = self.dataset_root / "VeRi"

        self.image_root = self.veri_root / "image_test"

        self.xml_file = self.veri_root / "test_label.xml"

        if not self.xml_file.exists():
            raise FileNotFoundError(
                f"Missing: {self.xml_file}"
            )

        if not self.image_root.exists():
            raise FileNotFoundError(
                f"Missing: {self.image_root}"
            )

        # ----------------------------------------------------
        # Read XML
        # ----------------------------------------------------

        print("\nReading test_label.xml ...")

        with open(
            self.xml_file,
            "r",
            encoding="gb2312"
        ) as f:

            xml_text = f.read()

        if xml_text.startswith("<?xml"):

            xml_text = xml_text.split(
                "?>",
                1
            )[1]

        root = ET.fromstring(xml_text)

        items = root.find("Items")

        if items is None:

            raise RuntimeError(
                "Items section not found in test_label.xml"
            )

        self.image_to_vehicle = {}

        for item in items:

            image_name = item.attrib["imageName"]

            vehicle_id = item.attrib["vehicleID"]

            self.image_to_vehicle[image_name] = vehicle_id

        print(
            "Images found in XML:",
            len(self.image_to_vehicle)
        )

        # ----------------------------------------------------
        # Locate query list
        # ----------------------------------------------------

        possible_query_files = [

            self.veri_root / "name_query.txt",

            self.dataset_root / "name_query.txt",

            self.veri_root / "image_query.txt",

            self.dataset_root / "image_query.txt",

        ]

        self.query_file = None

        for candidate in possible_query_files:

            if candidate.exists():

                self.query_file = candidate

                break

        if self.query_file is None:

            raise FileNotFoundError(
                "\nCould not find the official VeRi query list.\n"
                "Expected one of:\n"
                + "\n".join(
                    str(x)
                    for x in possible_query_files
                )
            )

        print(
            "Query list:",
            self.query_file
        )

        # ----------------------------------------------------
        # Read query names
        # ----------------------------------------------------

        self.query_names = []

        with open(
            self.query_file,
            "r"
        ) as f:

            for line in f:

                name = line.strip()

                if not name:
                    continue

                # Some versions of name_query.txt may contain
                # paths rather than only filenames.
                name = Path(name).name

                self.query_names.append(name)

        self.query_names = list(
            dict.fromkeys(self.query_names)
        )

        print(
            "Queries:",
            len(self.query_names)
        )

        # ----------------------------------------------------
        # Build gallery
        # ----------------------------------------------------

        test_images = sorted(
            self.image_to_vehicle.keys()
        )

        query_set = set(
            self.query_names
        )

        self.gallery_names = [

            name

            for name in test_images

            if name not in query_set

        ]

        print(
            "Gallery images:",
            len(self.gallery_names)
        )

        # ----------------------------------------------------
        # Verify query metadata
        # ----------------------------------------------------

        missing_queries = [

            name

            for name in self.query_names

            if name not in self.image_to_vehicle
        ]

        if missing_queries:

            print(
                "\nWARNING:"
            )

            print(
                "Queries missing from test_label.xml:",
                len(missing_queries)
            )

            print(
                "First few:",
                missing_queries[:10]
            )

        # ----------------------------------------------------
        # Camera IDs
        # ----------------------------------------------------

        self.image_to_camera = {}

        for name in test_images:

            self.image_to_camera[name] = (
                self.extract_camera_id(name)
            )

        camera_values = [

            x

            for x in self.image_to_camera.values()

            if x is not None
        ]

        print(
            "Images with parsed camera IDs:",
            len(camera_values),
            "/",
            len(test_images)
        )

        if len(camera_values) != len(test_images):

            bad = [

                name

                for name in test_images

                if self.image_to_camera[name] is None
            ]

            print(
                "WARNING: Could not parse camera ID from:"
            )

            print(
                bad[:10]
            )

    # --------------------------------------------------------
    # Camera parser
    # --------------------------------------------------------

    @staticmethod
    def extract_camera_id(image_name):

        """
        Standard VeRi filenames contain camera information
        in the form '_cXXX_'.

        Example:
            0002_c005_000000.jpg

        -> camera ID = 5
        """

        name = Path(image_name).name

        match = re.search(
            r"_c(\d+)_",
            name
        )

        if match is not None:

            return int(
                match.group(1)
            )

        # More permissive fallback
        match = re.search(
            r"_c(\d+)",
            name
        )

        if match is not None:

            return int(
                match.group(1)
            )

        return None


# ============================================================
# IMAGE DATASET
# ============================================================

class VeRiImageDataset(Dataset):

    def __init__(
        self,
        image_names,
        image_root,
        image_to_vehicle,
        transform,
    ):

        self.image_names = image_names

        self.image_root = Path(
            image_root
        )

        self.image_to_vehicle = (
            image_to_vehicle
        )

        self.transform = transform

    def __len__(self):

        return len(
            self.image_names
        )

    def __getitem__(self, index):

        image_name = self.image_names[index]

        image_path = (
            self.image_root /
            image_name
        )

        with open(
            image_path,
            "rb"
        ) as f:

            image = Image.open(f).convert(
                "RGB"
            )

        if self.transform:

            image = self.transform(
                image
            )

        vehicle_id = (
            self.image_to_vehicle[
                image_name
            ]
        )

        return image, image_name, vehicle_id


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding model...")

model = models.__dict__[cfg.experience](
    cfg,
    num_classes=575,
)

model = model.to(device)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\nLoading checkpoint...")

checkpoint = torch.load(
    args.checkpoint,
    map_location="cpu",
)

if "model" in checkpoint:

    state_dict = checkpoint["model"]

else:

    state_dict = checkpoint


clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):

        clean_state_dict[
            key[7:]
        ] = value

    else:

        clean_state_dict[key] = value


missing, unexpected = model.load_state_dict(
    clean_state_dict,
    strict=False,
)

print(
    "Missing keys:",
    len(missing)
)

print(
    "Unexpected keys:",
    len(unexpected)
)

if missing:

    print(
        "First missing keys:",
        missing[:10]
    )

if unexpected:

    print(
        "First unexpected keys:",
        unexpected[:10]
    )


# ============================================================
# FREEZE MODEL
# ============================================================

model.eval()

for param in model.parameters():

    param.requires_grad = False


print(
    "\nModel loaded and frozen."
)

print(
    "Invariant representation size:",
    model.inv_repr_size
)

print(
    "Invariant embedding size:",
    model.inv_emb_size
)


# ============================================================
# METADATA
# ============================================================

metadata = VeRiTestMetadata(
    args.dataset_root
)


# ============================================================
# DATA LOADERS
# ============================================================

query_dataset = VeRiImageDataset(
    metadata.query_names,
    metadata.image_root,
    metadata.image_to_vehicle,
    transform,
)

gallery_dataset = VeRiImageDataset(
    metadata.gallery_names,
    metadata.image_root,
    metadata.image_to_vehicle,
    transform,
)

query_loader = DataLoader(
    query_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)

gallery_loader = DataLoader(
    gallery_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_features(loader):

    all_features = []

    all_names = []

    all_vehicle_ids = []

    for batch_idx, batch in enumerate(loader):

        images, names, vehicle_ids = batch

        images = images.to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Frozen SIE backbone
        # ----------------------------------------------------

        backbone_features = model.backbone(
            images
        )

        # ----------------------------------------------------
        # Invariant branch
        # ----------------------------------------------------

        inv_features = backbone_features[
            ...,
            :model.inv_repr_size
        ]

        # ----------------------------------------------------
        # Invariant projector
        # ----------------------------------------------------

        inv_features = model.projector_inv(
            inv_features
        )

        # ----------------------------------------------------
        # Normalize for cosine similarity
        # ----------------------------------------------------

        inv_features = F.normalize(
            inv_features,
            dim=1
        )

        all_features.append(
            inv_features.cpu()
        )

        all_names.extend(
            list(names)
        )

        all_vehicle_ids.extend(
            list(vehicle_ids)
        )

        if batch_idx % 20 == 0:

            print(
                f"Extracted batch "
                f"{batch_idx + 1}/"
                f"{len(loader)}"
            )

    all_features = torch.cat(
        all_features,
        dim=0
    )

    return (
        all_features,
        all_names,
        all_vehicle_ids,
    )


print("\nExtracting QUERY features...")

query_features, query_names, query_vehicle_ids = (
    extract_features(query_loader)
)

print(
    "Query feature shape:",
    tuple(query_features.shape)
)


print("\nExtracting GALLERY features...")

gallery_features, gallery_names, gallery_vehicle_ids = (
    extract_features(gallery_loader)
)

print(
    "Gallery feature shape:",
    tuple(gallery_features.shape)
)


# ============================================================
# RETRIEVAL EVALUATION
# ============================================================

print("\nComputing VeRi retrieval metrics...")

gallery_features = gallery_features.float()
query_features = query_features.float()


# ------------------------------------------------------------
# Convert metadata to numpy-friendly form
# ------------------------------------------------------------

gallery_vehicle_ids = np.asarray(
    gallery_vehicle_ids
)

query_vehicle_ids = np.asarray(
    query_vehicle_ids
)

gallery_camera_ids = np.asarray([
    metadata.image_to_camera[name]
    for name in gallery_names
])

query_camera_ids = np.asarray([
    metadata.image_to_camera[name]
    for name in query_names
])


# ============================================================
# CMC / mAP
# ============================================================

cmc_hits = []

average_precisions = []

valid_queries = 0

num_queries = len(
    query_names
)

for q_idx in range(num_queries):

    q_feat = query_features[
        q_idx
    ]

    q_id = query_vehicle_ids[
        q_idx
    ]

    q_cam = query_camera_ids[
        q_idx
    ]

    # --------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------

    similarities = torch.matmul(
        gallery_features,
        q_feat
    )

    ranking = torch.argsort(
        similarities,
        descending=True
    ).cpu().numpy()

    ranked_ids = gallery_vehicle_ids[
        ranking
    ]

    ranked_cams = gallery_camera_ids[
        ranking
    ]

    # --------------------------------------------------------
    # Remove same-ID + same-camera junk
    # --------------------------------------------------------

    junk = (
        (ranked_ids == q_id)
        &
        (ranked_cams == q_cam)
    )

    keep = ~junk

    ranked_ids = ranked_ids[
        keep
    ]

    # --------------------------------------------------------
    # Correct matches
    # --------------------------------------------------------

    matches = (
        ranked_ids == q_id
    )

    num_good = int(
        matches.sum()
    )

    if num_good == 0:

        # No valid cross-camera gallery match
        continue

    valid_queries += 1

    # --------------------------------------------------------
    # CMC
    # --------------------------------------------------------

    first_correct = np.where(
        matches
    )[0][0]

    cmc_hits.append(
        first_correct
    )

    # --------------------------------------------------------
    # Average Precision
    # --------------------------------------------------------

    cumulative_hits = np.cumsum(
        matches
    )

    ranks = np.arange(
        1,
        len(matches) + 1
    )

    precision_at_k = (
        cumulative_hits
        / ranks
    )

    ap = (
        precision_at_k[matches].sum()
        / num_good
    )

    average_precisions.append(
        ap
    )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        (q_idx + 1) % 200 == 0
        or q_idx == num_queries - 1
    ):

        print(
            f"Processed "
            f"{q_idx + 1}/"
            f"{num_queries} queries"
        )


# ============================================================
# FINAL METRICS
# ============================================================

if valid_queries == 0:

    raise RuntimeError(
        "No valid queries remained after "
        "same-ID/same-camera filtering."
    )


cmc_hits = np.asarray(
    cmc_hits
)

average_precisions = np.asarray(
    average_precisions
)


cmc1 = np.mean(
    cmc_hits < 1
)

cmc5 = np.mean(
    cmc_hits < 5
)

cmc10 = np.mean(
    cmc_hits < 10
)

map_score = np.mean(
    average_precisions
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("VeRi-776 RETRIEVAL RESULTS")
print("=" * 70)

print(
    f"Total queries              : {num_queries}"
)

print(
    f"Valid queries              : {valid_queries}"
)

print(
    f"Gallery images             : {len(gallery_names)}"
)

print(
    f"CMC / Rank-1               : {cmc1 * 100:.2f}%"
)

print(
    f"CMC / Rank-5               : {cmc5 * 100:.2f}%"
)

print(
    f"CMC / Rank-10              : {cmc10 * 100:.2f}%"
)

print(
    f"mAP                        : {map_score * 100:.2f}%"
)

print("=" * 70)


# ============================================================
# SAVE RESULTS
# ============================================================

if args.output is None:

    args.output = (
        args.checkpoint.parent
        / "map_cmc_results.json"
    )

results = {

    "checkpoint": str(
        args.checkpoint
    ),

    "dataset_root": str(
        args.dataset_root
    ),

    "num_queries": int(
        num_queries
    ),

    "valid_queries": int(
        valid_queries
    ),

    "gallery_images": int(
        len(gallery_names)
    ),

    "invariant_repr_size": int(
        model.inv_repr_size
    ),

    "invariant_embedding_size": int(
        model.inv_emb_size
    ),

    "cmc_rank1": float(
        cmc1
    ),

    "cmc_rank5": float(
        cmc5
    ),

    "cmc_rank10": float(
        cmc10
    ),

    "mAP": float(
        map_score
    ),
}


with open(
    args.output,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print(
    "\nResults saved to:",
    args.output
)
