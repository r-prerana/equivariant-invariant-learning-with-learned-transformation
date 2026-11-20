# dataset_veri.py
from pathlib import Path
from collections import defaultdict
import random
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torchvision

from torch.utils.data import Dataset
from PIL import Image
from scipy.spatial.transform import Rotation as R


class DatasetVeRi(Dataset):

    def __init__(
        self,
        dataset_root,
        transform=None,
        size_dataset=-1,
    ):

        super().__init__()

        self.dataset_root = Path(dataset_root)

        self.transform = transform

        # ----------------------------------------------------------
        # Dataset paths
        # ----------------------------------------------------------

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

        print("\nInitializing DatasetVeRi...\n")

        # ----------------------------------------------------------
        # Read XML
        # ----------------------------------------------------------

        with open(self.xml_file, "r", encoding="gb2312") as f:

            xml_text = f.read()

        if xml_text.startswith("<?xml"):

            xml_text = xml_text.split("?>", 1)[1]

        root = ET.fromstring(xml_text)

        items = root.find("Items")
        if items is None:
            raise RuntimeError("Items section not found in train_label.xml")

        print(
            f"Training Images : {items.attrib['number']}"
        )

        # ----------------------------------------------------------
        # Read Orientation Annotations
        # ----------------------------------------------------------


        self.image_to_orientation = {}

        with open(self.keypoint_file, "r") as f:

            for line in f:

                tokens = line.strip().split()

                if not tokens:
                    continue

                image_name = Path(tokens[0]).name

                orientation = int(tokens[-1])

                self.image_to_orientation[image_name] = orientation

        print(
            f"Orientation Annotations : {len(self.image_to_orientation)}"
        )

        # ----------------------------------------------------------
        # Vehicle -> Images (keep only annotated images)
        # ----------------------------------------------------------

        self.vehicle_to_images = defaultdict(list)

        missing_images = 0

        for item in items:

            vehicle_id = item.attrib["vehicleID"]

            image_name = item.attrib["imageName"]

            if image_name not in self.image_to_orientation:
                missing_images += 1
                continue

            self.vehicle_to_images[vehicle_id].append(image_name)

        print(f"Skipped images without orientation : {missing_images}")

        # ----------------------------------------------------------
        # Remove vehicles with fewer than 2 annotated images
        # ----------------------------------------------------------

        filtered = {}

        for vehicle_id, imgs in self.vehicle_to_images.items():

            if len(imgs) >= 2:

                filtered[vehicle_id] = imgs

        self.vehicle_to_images = filtered

        self.vehicle_ids = sorted(
            self.vehicle_to_images.keys()
        )

        print(
            f"Training Vehicles : {len(self.vehicle_ids)}"
        )

        print(
            f"Training Images After Filtering : {sum(len(v) for v in self.vehicle_to_images.values())}"
        )

        # ----------------------------------------------------------
        # Create Integer Labels
        # ----------------------------------------------------------

        self.label_map = {}

        for idx, vehicle in enumerate(self.vehicle_ids):

            self.label_map[vehicle] = idx

        # ----------------------------------------------------------
        # Optional Dataset Reduction
        # ----------------------------------------------------------

        if size_dataset > 0:

            self.vehicle_ids = self.vehicle_ids[:size_dataset]

    # ----------------------------------------------------------
    # Image Loader
    # ----------------------------------------------------------

    def get_img(self, image_name):

        image_path = self.image_root / image_name

        with open(image_path, "rb") as f:

            img = Image.open(f)

            img = img.convert("RGB")

            if self.transform:

                img = self.transform(img)

        return img

    # ----------------------------------------------------------
    # Relative Orientation -> Quaternion
    # ----------------------------------------------------------

    def orientation_to_quaternion(self, ori1, ori2):

        # Convert VeRi orientation labels to canonical angles.
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

        angle1 = LABEL_TO_ANGLE[int(ori1)]
        angle2 = LABEL_TO_ANGLE[int(ori2)]

        # Relative rotation
        relative_angle = angle2 - angle1

        # Normalize to [-180, 180]
        while relative_angle > 180:
            relative_angle -= 360

        while relative_angle < -180:
            relative_angle += 360

        # Create rotation about Z-axis
        rotation = R.from_euler(
            "z",
            relative_angle,
            degrees=True,
        )

        quaternion = rotation.as_quat().astype(np.float32)

        return quaternion

    # ----------------------------------------------------------
    # Get One Training Sample
    # ----------------------------------------------------------

    def __getitem__(self, index):

        # ------------------------------------------------------
        # Vehicle
        # ------------------------------------------------------

        vehicle_id = self.vehicle_ids[index]

        label = self.label_map[vehicle_id]

        images = self.vehicle_to_images[vehicle_id]

        # ------------------------------------------------------
        # Need at least two images
        # ------------------------------------------------------

        if len(images) < 2:

            raise RuntimeError(
                f"Vehicle {vehicle_id} has fewer than 2 images."
            )

        # ------------------------------------------------------
        # Randomly sample two different images
        # ------------------------------------------------------

        img1_name, img2_name = random.sample(images, 2)

        # ------------------------------------------------------
        # Load images
        # ------------------------------------------------------

        img1 = self.get_img(img1_name)

        img2 = self.get_img(img2_name)

        # ------------------------------------------------------
        # Read orientations
        # ------------------------------------------------------

        ori1 = self.image_to_orientation[img1_name]

        ori2 = self.image_to_orientation[img2_name]

        # ------------------------------------------------------
        # Relative transformation
        # ------------------------------------------------------

        quaternion = self.orientation_to_quaternion(
            ori1,
            ori2,
        )

        return img1, img2, torch.FloatTensor(quaternion), label
    # ----------------------------------------------------------
    # Dataset Length
    # ----------------------------------------------------------

    def __len__(self):

        return len(self.vehicle_ids)

class DatasetVeRiEval(Dataset):

    def __init__(
        self,
        dataset_root,
        split="train",
        transform=None,
    ):

        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.transform = transform

        if split == "train":
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

        else:
            self.xml_file = (
                self.dataset_root /
                "VeRi" /
                "test_label.xml"
            )

            self.image_root = (
                self.dataset_root /
                "VeRi" /
                "image_test"
            )

        print(f"\nInitializing DatasetVeRiEval ({split})...\n")

        with open(self.xml_file, "r", encoding="gb2312") as f:

            xml_text = f.read()

        if xml_text.startswith("<?xml"):

            xml_text = xml_text.split("?>", 1)[1]

        root = ET.fromstring(xml_text)

        items = root.find("Items")

        if items is None:
            raise RuntimeError("Items section not found.")

        vehicle_ids = sorted(
            list(
                set(
                    item.attrib["vehicleID"]
                    for item in items
                )
            )
        )

        self.label_map = {}

        for idx, vehicle in enumerate(vehicle_ids):

            self.label_map[vehicle] = idx

        self.samples = []

        for item in items:

            image_name = item.attrib["imageName"]

            vehicle_id = item.attrib["vehicleID"]

            self.samples.append(
                (
                    image_name,
                    self.label_map[vehicle_id],
                )
            )

        print(f"Images : {len(self.samples)}")

        print(f"Vehicles : {len(vehicle_ids)}")

    def get_img(self, image_name):

        image_path = self.image_root / image_name

        with open(image_path, "rb") as f:

            img = Image.open(f)

            img = img.convert("RGB")

        if self.transform:

            img = self.transform(img)

        return img

    def __getitem__(self, index):

        image_name, label = self.samples[index]

        img = self.get_img(image_name)

        return img, label

    def __len__(self):

        return len(self.samples)


class DatasetVeRiLinear(Dataset):

    def __init__(
        self,
        dataset_root,
        split="train",
        transform=None,
        train_ratio=0.8,
        seed=42,
    ):

        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.transform = transform

        self.image_root = (
            self.dataset_root /
            "VeRi" /
            "image_train"
        )

        self.xml_file = (
            self.dataset_root /
            "VeRi" /
            "train_label.xml"
        )

        print(f"\nInitializing DatasetVeRiLinear ({split})...\n")

        with open(self.xml_file, "r", encoding="gb2312") as f:
            xml_text = f.read()

        if xml_text.startswith("<?xml"):
            xml_text = xml_text.split("?>", 1)[1]

        root = ET.fromstring(xml_text)

        items = root.find("Items")

        if items is None:
            raise RuntimeError("Items section not found.")

        # ---------------------------------------------------
        # Group images by vehicle ID
        # ---------------------------------------------------

        vehicle_dict = {}

        for item in items:

            vehicle_id = item.attrib["vehicleID"]
            image_name = item.attrib["imageName"]

            if vehicle_id not in vehicle_dict:
                vehicle_dict[vehicle_id] = []

            vehicle_dict[vehicle_id].append(image_name)

        # ---------------------------------------------------
        # Create label mapping
        # ---------------------------------------------------

        vehicle_ids = sorted(vehicle_dict.keys())

        self.label_map = {
            vehicle_id: idx
            for idx, vehicle_id in enumerate(vehicle_ids)
        }

        # ---------------------------------------------------
        # Split each vehicle independently
        # ---------------------------------------------------

        random.seed(seed)

        self.samples = []

        for vehicle_id in vehicle_ids:

            images = vehicle_dict[vehicle_id][:]

            random.shuffle(images)

            split_idx = max(
                1,
                int(len(images) * train_ratio)
            )

            if split == "train":
                selected = images[:split_idx]
            else:
                selected = images[split_idx:]

                # Handle vehicles having only one image
                if len(selected) == 0:
                    selected = images[-1:]

            label = self.label_map[vehicle_id]

            for image_name in selected:
                self.samples.append(
                    (
                        image_name,
                        label
                    )
                )

        print(f"Images : {len(self.samples)}")
        print(f"Vehicles : {len(vehicle_ids)}")

    def get_img(self, image_name):

        image_path = self.image_root / image_name

        with open(image_path, "rb") as f:

            img = Image.open(f)
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img

    def __getitem__(self, index):

        image_name, label = self.samples[index]

        img = self.get_img(image_name)

        return img, label

    def __len__(self):

        return len(self.samples)

class DatasetVeRiPredictorEval(DatasetVeRi):

    def __init__(self, dataset_root, transform=None):

        super().__init__(
            dataset_root,
            transform,
            size_dataset=-1,
        )

        from collections import defaultdict

        # -------------------------------------------------
        # Build all pairs grouped by transition
        # -------------------------------------------------

        pairs_by_transition = defaultdict(list)

        for vehicle_id in self.vehicle_ids:

            images = self.vehicle_to_images[vehicle_id]

            if len(images) < 2:
                continue

            for i in range(len(images)):
                for j in range(i + 1, len(images)):

                    img1_name = images[i]
                    img2_name = images[j]

                    ori1 = self.image_to_orientation[img1_name]
                    ori2 = self.image_to_orientation[img2_name]

                    transition = (ori1, ori2)

                    pairs_by_transition[transition].append(
                        (
                            img1_name,
                            img2_name,
                            vehicle_id,
                        )
                    )

        # -------------------------------------------------
        # Print transition statistics
        # -------------------------------------------------

        print("\nTransition Counts")
        print("-" * 50)

        total_pairs = 0

        for transition, pairs in sorted(
            pairs_by_transition.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        ):

            print(f"{transition} : {len(pairs)}")
            total_pairs += len(pairs)

        print("-" * 50)
        print(f"Total pairs : {total_pairs}")

        # -------------------------------------------------
        # Build balanced evaluation list
        # -------------------------------------------------

        import random

        MAX_PER_TRANSITION = 1500

        random.seed(42)

        self.samples = []

        for transition, pairs in pairs_by_transition.items():

            if len(pairs) > MAX_PER_TRANSITION:
                pairs = random.sample(pairs, MAX_PER_TRANSITION)

            self.samples.extend(pairs)

        print("-" * 50)
        print(f"Final Predictor Evaluation Pairs : {len(self.samples)}")


    def __getitem__(self, index):

        img1_name, img2_name, vehicle_id = self.samples[index]

        label = self.label_map[vehicle_id]

        img1 = self.get_img(img1_name)
        img2 = self.get_img(img2_name)

        ori1 = self.image_to_orientation[img1_name]
        ori2 = self.image_to_orientation[img2_name]

        quaternion = self.orientation_to_quaternion(
            ori1,
            ori2,
        )

        return img1, img2, torch.FloatTensor(quaternion), label, ori1, ori2
    def __len__(self):

        return len(self.samples)
