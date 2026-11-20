# src/dataset_dn348_supcon.py
#
# Identity-balanced multi-image dataset and sampler for DN-348.
# Target structure: 16 identities x 4 images = 64 images per batch.
# Each identity attempts to sample: Day1, Day2, Night1, Night2 independently.
# Zero filename correspondence assumed between Day and Night images.

from pathlib import Path
from collections import defaultdict
import random
from PIL import Image

import torch
from torch.utils.data import Dataset, Sampler


# Transition mapping:
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

DOMAIN_INT_TO_STR = {0: "day", 1: "night"}
DOMAIN_STR_TO_INT = {"day": 0, "night": 1}


class IdentityBalancedDN348Dataset(Dataset):
    """
    Dataset that loads individual images from DN-348 official splits.
    Returns: (img_tensor, vehicle_label, domain_int, rel_path)
    """

    def __init__(self, dataset_root, transform=None):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        self.split_root = self.dataset_root / "train_test_split"
        self.day_root = self.dataset_root / "day"
        self.night_root = self.dataset_root / "night"

        self.samples = []
        self.vehicle_to_indices = defaultdict(lambda: {"day": [], "night": []})

        with open(self.split_root / "train_list_day.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                v_id = line.split("/")[0]
                idx = len(self.samples)
                self.samples.append((self.day_root / line, v_id, "day", line))
                self.vehicle_to_indices[v_id]["day"].append(idx)

        with open(self.split_root / "train_list_night.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                v_id = line.split("/")[0]
                idx = len(self.samples)
                self.samples.append((self.night_root / line, v_id, "night", line))
                self.vehicle_to_indices[v_id]["night"].append(idx)

        self.vehicle_ids = sorted(self.vehicle_to_indices.keys())
        self.label_map = {vid: idx for idx, vid in enumerate(self.vehicle_ids)}

        day_total = sum(len(self.vehicle_to_indices[v]["day"]) for v in self.vehicle_ids)
        night_total = sum(len(self.vehicle_to_indices[v]["night"]) for v in self.vehicle_ids)
        print(f"IdentityBalancedDN348Dataset initialized:")
        print(f"  Training Vehicles : {len(self.vehicle_ids)}")
        print(f"  Day Images        : {day_total}")
        print(f"  Night Images      : {night_total}")
        print(f"  Total Images      : {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, v_id, domain, rel_path = self.samples[idx]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.label_map[v_id], DOMAIN_STR_TO_INT[domain], rel_path


class IdentityBalancedBatchSampler(Sampler):
    """
    BatchSampler that constructs identity-balanced batches:
      - 16 vehicle identities per batch
      - 4 images per identity
      - Total = 64 images per batch (shape [64, 3, 256, 256])
    
    For each identity, independently samples 2 Day images and 2 Night images
    without replacement from their respective image pools.
    Zero filename correspondence is assumed between Day and Night images.
    """

    def __init__(
        self,
        dataset: IdentityBalancedDN348Dataset,
        num_identities: int = 16,
        images_per_id: int = 4,
        batches_per_epoch: int = 50,
    ):
        self.dataset = dataset
        self.num_identities = num_identities
        self.images_per_id = images_per_id
        self.batches_per_epoch = batches_per_epoch

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        v_ids = self.dataset.vehicle_ids
        for _ in range(self.batches_per_epoch):
            selected_ids = random.sample(v_ids, self.num_identities)
            batch = []
            for vid in selected_ids:
                day_pool = self.dataset.vehicle_to_indices[vid]["day"]
                night_pool = self.dataset.vehicle_to_indices[vid]["night"]

                # Target: 2 Day images, 2 Night images sampled independently
                if len(day_pool) >= 2 and len(night_pool) >= 2:
                    d_sampled = random.sample(day_pool, 2)
                    n_sampled = random.sample(night_pool, 2)
                    id_indices = [d_sampled[0], d_sampled[1], n_sampled[0], n_sampled[1]]
                else:
                    # Fallback preserving identity grouping
                    all_pool = day_pool + night_pool
                    if len(all_pool) >= 4:
                        id_indices = random.sample(all_pool, 4)
                    else:
                        id_indices = [random.choice(all_pool) for _ in range(4)]
                batch.extend(id_indices)
            yield batch


def build_sie_pairs_from_batch(labels: torch.Tensor, domains: torch.Tensor):
    """
    Constructs balanced SIE pairs from a 64-image batch (16 identities x 4 images).
    
    For each identity k (images 4k+0, 4k+1, 4k+2, 4k+3):
      - Pair 1: (4k+0, 4k+1) -> DD (if D1, D2)
      - Pair 2: (4k+2, 4k+3) -> NN (if N1, N2)
      - Pair 3: (4k+0, 4k+2) -> DN (if D1, N1)
      - Pair 4: (4k+3, 4k+1) -> ND (if N2, D2)
    
    Transition targets are dynamically computed from actual domains:
      TRANSITION_TO_LABEL[(domain[i], domain[j])]
    
    Returns:
      idx_x: Tensor of shape [64] (first image indices)
      idx_y: Tensor of shape [64] (second image indices)
      transition_targets: Tensor of shape [64] (transition class labels in {0, 1, 2, 3})
      pair_labels: Tensor of shape [64] (identity labels for each pair)
    """
    device = labels.device
    idx_x = []
    idx_y = []
    transition_targets = []
    pair_labels = []

    domains_list = domains.cpu().tolist()
    labels_list = labels.cpu().tolist()

    num_identities = len(labels) // 4
    for k in range(num_identities):
        base = 4 * k
        pairs = [
            (base + 0, base + 1),  # Typically Day1 -> Day2 (DD)
            (base + 2, base + 3),  # Typically Night1 -> Night2 (NN)
            (base + 0, base + 2),  # Typically Day1 -> Night1 (DN)
            (base + 3, base + 1),  # Typically Night2 -> Day2 (ND)
        ]
        for i, j in pairs:
            idx_x.append(i)
            idx_y.append(j)
            cond_i = DOMAIN_INT_TO_STR[domains_list[i]]
            cond_j = DOMAIN_INT_TO_STR[domains_list[j]]
            t_label = TRANSITION_TO_LABEL[(cond_i, cond_j)]
            transition_targets.append(t_label)
            pair_labels.append(labels_list[base])

    return (
        torch.tensor(idx_x, dtype=torch.long, device=device),
        torch.tensor(idx_y, dtype=torch.long, device=device),
        torch.tensor(transition_targets, dtype=torch.long, device=device),
        torch.tensor(pair_labels, dtype=torch.long, device=device),
    )
