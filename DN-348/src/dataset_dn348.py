# src/dataset_dn348.py
#
# DN-348 dataset implementation following DatasetVeRi conventions.
# Uses official disjoint train/test split files.

from pathlib import Path
from collections import defaultdict
import random
from PIL import Image

import torch
from torch.utils.data import Dataset


# Transformation label mapping:
# 0 = DD (Day -> Day)
# 1 = NN (Night -> Night)
# 2 = DN (Day -> Night)
# 3 = ND (Night -> Day)
TRANSITION_TO_LABEL = {
    ("day", "day"): 0,
    ("night", "night"): 1,
    ("day", "night"): 2,
    ("night", "day"): 3,
}


class DatasetDN348(Dataset):
    """
    Pairwise training dataset for DN-348.
    Samples pairs from the same vehicle identity on the fly.
    Returns: (img1, img2, transition_label, vehicle_label)
    """

    def __init__(
        self,
        dataset_root,
        transform=None,
        samples_per_epoch=None,
    ):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.transform = transform

        self.split_root = self.dataset_root / "train_test_split"
        self.day_root = self.dataset_root / "day"
        self.night_root = self.dataset_root / "night"

        self.day_file = self.split_root / "train_list_day.txt"
        self.night_file = self.split_root / "train_list_night.txt"

        if not self.day_file.exists() or not self.night_file.exists():
            raise FileNotFoundError(f"Missing official split files under {self.split_root}")

        print("\nInitializing DatasetDN348...")

        # Parse official splits
        self.vehicle_to_images = defaultdict(list)
        day_count = 0
        with open(self.day_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vehicle_id = line.split("/")[0]
                self.vehicle_to_images[vehicle_id].append((self.day_root / line, "day"))
                day_count += 1

        night_count = 0
        with open(self.night_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vehicle_id = line.split("/")[0]
                self.vehicle_to_images[vehicle_id].append((self.night_root / line, "night"))
                night_count += 1

        self.vehicle_ids = sorted(self.vehicle_to_images.keys())
        self.label_map = {vid: idx for idx, vid in enumerate(self.vehicle_ids)}
        self.samples_per_epoch = samples_per_epoch if samples_per_epoch is not None else len(self.vehicle_ids)

        print(f"Training Vehicles   : {len(self.vehicle_ids)}")
        print(f"Day Images          : {day_count}")
        print(f"Night Images        : {night_count}")
        print(f"Total Images        : {day_count + night_count}")
        print(f"Samples per Epoch   : {self.samples_per_epoch}\n")

    def _load_image(self, path):
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        vehicle_id = self.vehicle_ids[index % len(self.vehicle_ids)]
        vehicle_label = self.label_map[vehicle_id]
        images = self.vehicle_to_images[vehicle_id]

        if len(images) < 2:
            raise RuntimeError(f"Vehicle {vehicle_id} has fewer than 2 images.")

        (path1, cond1), (path2, cond2) = random.sample(images, 2)
        img1 = self._load_image(path1)
        img2 = self._load_image(path2)
        trans_label = TRANSITION_TO_LABEL[(cond1, cond2)]

        return (
            img1,
            img2,
            torch.tensor(trans_label, dtype=torch.long),
            torch.tensor(vehicle_label, dtype=torch.long),
        )


class DatasetDN348Eval(Dataset):
    """
    Individual test image dataset for cross-domain retrieval evaluation.
    Returns: (image, vehicle_id, condition, relative_path)
    """

    def __init__(self, dataset_root, transform=None, split="test"):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        self.split_root = self.dataset_root / "train_test_split"

        day_file = self.split_root / f"{split}_list_day.txt"
        night_file = self.split_root / f"{split}_list_night.txt"

        self.samples = []
        with open(day_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append((self.dataset_root / "day" / line, line.split("/")[0], "day", line))
        with open(night_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append((self.dataset_root / "night" / line, line.split("/")[0], "night", line))

        print(f"DatasetDN348Eval ({split}): {len(self.samples)} images loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, vehicle_id, condition, rel_path = self.samples[index]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img, vehicle_id, condition, rel_path
