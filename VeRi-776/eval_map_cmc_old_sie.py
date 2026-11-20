# ============================================================
# VeRi-776 mAP / CMC evaluation
# Original SIE 2000-epoch checkpoint
#
# Checkpoint:
# ./SIE_VeRi/checkpoints/veri_2000ep/model.pth
#
# This evaluates the INVARIANT embedding only.
# No training is performed.
# ============================================================

import os
import re
import json
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm import tqdm

import src.models as models


# ============================================================
# PATHS
# ============================================================

CHECKPOINT_DIR = Path(
    "./checkpoints/veri_2000ep"
)

CHECKPOINT_PATH = CHECKPOINT_DIR / "model.pth"

DATASET_ROOT = Path(
    "./data/VeRi"
)

TEST_XML = DATASET_ROOT / "test_label.xml"
QUERY_FILE = DATASET_ROOT / "name_query.txt"

IMAGE_DIR = DATASET_ROOT / "image_test"

LOG_ROOT = Path(
    "./logs/map_cmc"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)
print("Checkpoint:", CHECKPOINT_PATH)
print("Dataset root:", DATASET_ROOT)


# ============================================================
# IMAGE NORMALIZATION
# Same normalization used in the original evaluation code.
# ============================================================

NORMALIZE = transforms.Normalize(
    mean=[0.5016, 0.5037, 0.5060],
    std=[0.1030, 0.0999, 0.0969]
)

TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    NORMALIZE,
])


# ============================================================
# READ VERI XML
# ============================================================

def read_test_xml(xml_path):

    print()
    print("Reading:", xml_path)

    with open(xml_path, "r", encoding="gb2312") as f:
        xml_text = f.read()

    if xml_text.startswith("<?xml"):
        xml_text = xml_text.split("?>", 1)[1]

    root = ET.fromstring(xml_text)

    records = []

    for elem in root.iter():

        attrs = elem.attrib

        # Try several possible attribute names because
        # VeRi releases can use slightly different XML naming.
        image_name = (
            attrs.get("imageName")
            or attrs.get("image_name")
            or attrs.get("name")
            or attrs.get("imgName")
            or attrs.get("img_name")
        )

        vehicle_id = (
            attrs.get("vehicleID")
            or attrs.get("vehicle_id")
            or attrs.get("id")
            or attrs.get("carID")
        )

        camera_id = (
            attrs.get("cameraID")
            or attrs.get("camera_id")
            or attrs.get("camID")
            or attrs.get("cam_id")
        )

        if image_name is None:
            continue

        # ----------------------------------------------------
        # If vehicle/camera IDs are absent from XML,
        # try extracting them from the filename.
        #
        # Typical VeRi name:
        # 0001_c001_00016400_0.jpg
        #          ^^^
        #          camera
        # ----------------------------------------------------

        if vehicle_id is None or camera_id is None:

            base = os.path.basename(image_name)

            match = re.match(
                r"(\d+)_c(\d+)_",
                base
            )

            if match:

                if vehicle_id is None:
                    vehicle_id = match.group(1)

                if camera_id is None:
                    camera_id = match.group(2)

        if vehicle_id is None:
            continue

        if camera_id is None:
            continue

        records.append({
            "name": os.path.basename(image_name),
            "vehicle_id": str(vehicle_id),
            "camera_id": str(camera_id),
        })

    print("Images parsed:", len(records))

    return records


# ============================================================
# READ QUERY LIST
# ============================================================

def read_query_file(query_path):

    print()
    print("Reading query list:", query_path)

    with open(query_path, "r") as f:
        queries = [
            os.path.basename(line.strip())
            for line in f
            if line.strip()
        ]

    # Remove duplicates while preserving order
    queries = list(dict.fromkeys(queries))

    print("Queries:", len(queries))

    return queries


# ============================================================
# DATASET
# ============================================================

class VeRiImageDataset(Dataset):

    def __init__(self, records, image_dir, transform):

        self.records = records
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        image_name = record["name"]

        image_path = self.image_dir / image_name

        # ----------------------------------------------------
        # Some VeRi setups may store images directly under
        # image_test. If not found, search recursively.
        # ----------------------------------------------------

        if not image_path.exists():

            matches = list(
                self.image_dir.rglob(image_name)
            )

            if len(matches) == 0:
                raise FileNotFoundError(
                    f"Could not find image: {image_name}"
                )

            image_path = matches[0]

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        return image, index


# ============================================================
# BUILD ORIGINAL SIE MODEL
# ============================================================

def build_original_sie():

    # Parameters taken from:
    #
    # ./SIE_VeRi/checkpoints/veri_2000ep/params.json
    #
    # {
    #   "arch": "resnet18",
    #   "equi": 256,
    #   "mlp": "2048-2048-2048",
    #   "predictor_type": "hypernetwork",
    #   ...
    # }

    args = SimpleNamespace()

    args.arch = "resnet18"
    args.equi = 256
    args.mlp = "2048-2048-2048"

    args.predictor_type = "hypernetwork"
    args.predictor = ""
    args.hypernetwork = "linear"
    args.tf_num_layers = 1
    
    args.bias_pred = False
    args.bias_hypernet = False
    args.predictor_relu = False



    args.equi_factor = 4.5
    args.sim_coeff = 10.0
    args.std_coeff = 10.0
    args.cov_coeff = 1.0

    # Needed by model construction
    args.batch_size = 8

    print()
    print("Building ORIGINAL SIE model...")
    print("Architecture :", args.arch)
    print("Equivariant   :", args.equi)
    print("MLP           :", args.mlp)
    print("Predictor     :", args.predictor_type)

    model = models.SIE(args)

    return model


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(model, checkpoint_path):

    print()
    print("Loading checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    print("Checkpoint type:", type(checkpoint))

    # --------------------------------------------------------
    # The original training code normally stores:
    #
    # checkpoint["model"]
    #
    # --------------------------------------------------------

    if isinstance(checkpoint, dict) and "model" in checkpoint:

        state_dict = checkpoint["model"]

    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Remove DistributedDataParallel "module." prefix
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key

        cleaned_state_dict[new_key] = value

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    print("Missing keys   :", len(missing))
    print("Unexpected keys:", len(unexpected))

    if len(missing) > 0:
        print()
        print("Missing keys:")
        for k in missing:
            print("  ", k)

    if len(unexpected) > 0:
        print()
        print("Unexpected keys:")
        for k in unexpected:
            print("  ", k)

    return model


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_features(
    model,
    records,
    image_dir,
    batch_size=64,
    num_workers=4
):

    dataset = VeRiImageDataset(
        records,
        image_dir,
        TRANSFORM
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model.eval()

    all_features = []

    print()
    print(
        f"Extracting features from {len(records)} images..."
    )

    for batch_idx, (images, indices) in enumerate(
        tqdm(loader)
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # ORIGINAL SIE:
        #
        # backbone output:
        # [ invariant | equivariant ]
        #
        # equi = 256
        # total repr size = 512 for ResNet18
        #
        # invariant part is the FIRST 256 dimensions.
        # ----------------------------------------------------

        backbone_output = model.backbone(images)

        invariant_repr = backbone_output[
            ..., :model.inv_repr_size
        ]

        # ----------------------------------------------------
        # Apply invariant projector
        # ----------------------------------------------------

        invariant_embedding = model.projector_inv(
            invariant_repr
        )

        # ----------------------------------------------------
        # Normalize for cosine similarity
        # ----------------------------------------------------

        invariant_embedding = F.normalize(
            invariant_embedding,
            p=2,
            dim=1
        )

        all_features.append(
            invariant_embedding.cpu()
        )

    features = torch.cat(
        all_features,
        dim=0
    ).numpy()

    print("Feature shape:", features.shape)

    return features


# ============================================================
# CMC + mAP
# ============================================================

def evaluate_retrieval(
    query_records,
    gallery_records,
    query_features,
    gallery_features
):

    print()
    print("Computing VeRi retrieval metrics...")

    # Cosine similarity because features are normalized.
    similarity = np.matmul(
        query_features,
        gallery_features.T
    )

    cmc_rank1 = []
    cmc_rank5 = []
    cmc_rank10 = []

    average_precisions = []

    valid_queries = 0

    for q_idx in tqdm(
        range(len(query_records))
    ):

        query = query_records[q_idx]

        query_id = query["vehicle_id"]
        query_cam = query["camera_id"]

        scores = similarity[q_idx]

        ranking = np.argsort(
            -scores
        )

        # ----------------------------------------------------
        # Remove same vehicle + same camera.
        #
        # These are junk images under the VeRi protocol.
        # ----------------------------------------------------

        keep = []

        for gallery_idx in ranking:

            gallery = gallery_records[
                gallery_idx
            ]

            if (
                gallery["vehicle_id"] == query_id
                and
                gallery["camera_id"] == query_cam
            ):
                continue

            keep.append(gallery_idx)

        ranking = np.array(
            keep,
            dtype=np.int64
        )

        # ----------------------------------------------------
        # Determine correct matches.
        # ----------------------------------------------------

        matches = np.array([
            gallery_records[idx]["vehicle_id"] == query_id
            for idx in ranking
        ])

        num_matches = matches.sum()

        if num_matches == 0:
            continue

        valid_queries += 1

        # ----------------------------------------------------
        # CMC
        # ----------------------------------------------------

        if matches[:1].any():
            cmc_rank1.append(1)
        else:
            cmc_rank1.append(0)

        if matches[:5].any():
            cmc_rank5.append(1)
        else:
            cmc_rank5.append(0)

        if matches[:10].any():
            cmc_rank10.append(1)
        else:
            cmc_rank10.append(0)

        # ----------------------------------------------------
        # Average Precision
        # ----------------------------------------------------

        positive_positions = np.where(matches)[0]

        precisions = []

        for rank_position in positive_positions:

            num_correct = matches[
                :rank_position + 1
            ].sum()

            precision = (
                num_correct
                / float(rank_position + 1)
            )

            precisions.append(precision)

        ap = np.mean(precisions)

        average_precisions.append(ap)

    # --------------------------------------------------------
    # Final values
    # --------------------------------------------------------

    rank1 = (
        np.mean(cmc_rank1)
        if len(cmc_rank1) > 0
        else 0.0
    )

    rank5 = (
        np.mean(cmc_rank5)
        if len(cmc_rank5) > 0
        else 0.0
    )

    rank10 = (
        np.mean(cmc_rank10)
        if len(cmc_rank10) > 0
        else 0.0
    )

    mAP = (
        np.mean(average_precisions)
        if len(average_precisions) > 0
        else 0.0
    )

    return {
        "total_queries": len(query_records),
        "valid_queries": valid_queries,
        "gallery_images": len(gallery_records),
        "rank1": float(rank1),
        "rank5": float(rank5),
        "rank10": float(rank10),
        "mAP": float(mAP),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    for path in [
        CHECKPOINT_PATH,
        DATASET_ROOT,
        TEST_XML,
        QUERY_FILE,
        IMAGE_DIR,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required path does not exist:\n{path}"
            )

    # --------------------------------------------------------
    # Read test metadata
    # --------------------------------------------------------

    all_records = read_test_xml(
        TEST_XML
    )

    query_names = read_query_file(
        QUERY_FILE
    )

    query_name_set = set(
        query_names
    )

    # --------------------------------------------------------
    # Build query set
    # --------------------------------------------------------

    record_by_name = {
        r["name"]: r
        for r in all_records
    }

    query_records = []

    for name in query_names:

        if name not in record_by_name:

            print(
                "WARNING: query image not found:",
                name
            )

            continue

        query_records.append(
            record_by_name[name]
        )

    # --------------------------------------------------------
    # Gallery = all test images NOT in query list
    # --------------------------------------------------------

    gallery_records = [
        r
        for r in all_records
        if r["name"] not in query_name_set
    ]

    print()
    print("=" * 70)
    print("VeRi DATA SPLIT")
    print("=" * 70)

    print(
        "Total test images :",
        len(all_records)
    )

    print(
        "Query images      :",
        len(query_records)
    )

    print(
        "Gallery images    :",
        len(gallery_records)
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_original_sie()

    model = load_checkpoint(
        model,
        CHECKPOINT_PATH
    )

    model = model.to(DEVICE)

    # Freeze
    for parameter in model.parameters():
        parameter.requires_grad = False

    print()
    print("Model loaded and frozen.")

    print(
        "Invariant representation size:",
        model.inv_repr_size
    )

    print(
        "Invariant embedding size:",
        model.inv_emb_size
    )

    # --------------------------------------------------------
    # Extract query features
    # --------------------------------------------------------

    print()
    print("EXTRACTING QUERY FEATURES")

    query_features = extract_features(
        model,
        query_records,
        IMAGE_DIR,
        batch_size=64,
        num_workers=4
    )

    # --------------------------------------------------------
    # Extract gallery features
    # --------------------------------------------------------

    print()
    print("EXTRACTING GALLERY FEATURES")

    gallery_features = extract_features(
        model,
        gallery_records,
        IMAGE_DIR,
        batch_size=64,
        num_workers=4
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = evaluate_retrieval(
        query_records,
        gallery_records,
        query_features,
        gallery_features
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ORIGINAL SIE — VeRi-776 RETRIEVAL RESULTS")
    print("=" * 70)

    print(
        f"Total queries   : {results['total_queries']}"
    )

    print(
        f"Valid queries   : {results['valid_queries']}"
    )

    print(
        f"Gallery images  : {results['gallery_images']}"
    )

    print(
        f"CMC / Rank-1    : {results['rank1'] * 100:.2f}%"
    )

    print(
        f"CMC / Rank-5    : {results['rank5'] * 100:.2f}%"
    )

    print(
        f"CMC / Rank-10   : {results['rank10'] * 100:.2f}%"
    )

    print(
        f"mAP             : {results['mAP'] * 100:.2f}%"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    output_dir = LOG_ROOT / (
        "veri_original_sie_2000ep"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = output_dir / (
        "map_cmc_results.json"
    )

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print()
    print(
        "Results saved to:",
        output_file
    )


if __name__ == "__main__":
    main()
