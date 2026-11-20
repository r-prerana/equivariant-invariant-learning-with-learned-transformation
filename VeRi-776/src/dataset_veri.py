from pathlib import Path
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from PIL import Image


class DatasetVeRi:

    def __init__(self):

        print("Initializing DatasetVeRi...")

        # --------------------------------------------------
        # Dataset paths
        # --------------------------------------------------

        self.xml_file = Path("./data/VeRi/train_label.xml")

        self.dataset_root = Path("./data/VeRi/image_train")

        self.keypoint_file = Path(
            "./data/VehicleReIDKeyPointData/keypoint_train.txt"
        )

        # --------------------------------------------------
        # Read XML
        # --------------------------------------------------

        with open(self.xml_file, "r", encoding="gb2312") as f:
            xml_text = f.read()

        if xml_text.startswith("<?xml"):
            xml_text = xml_text.split("?>", 1)[1]

        self.root = ET.fromstring(xml_text)

        self.items = self.root.find("Items")

        print(f"Training images : {self.items.attrib['number']}")

        # --------------------------------------------------
        # Vehicle -> Images
        # --------------------------------------------------

        self.vehicle_to_images = defaultdict(list)

        for item in self.items:

            vehicle_id = item.attrib["vehicleID"]

            image_name = item.attrib["imageName"]

            self.vehicle_to_images[vehicle_id].append(image_name)

        print(f"Training vehicles : {len(self.vehicle_to_images)}")

        # --------------------------------------------------
        # Image -> Orientation
        # --------------------------------------------------

        self.image_to_orientation = {}

        with open(self.keypoint_file) as f:

            for line in f:

                tokens = line.strip().split()

                image_name = Path(tokens[0]).name

                orientation = int(tokens[-1])

                self.image_to_orientation[image_name] = orientation

        print(
            f"Orientation annotations : {len(self.image_to_orientation)}"
        )

    # ------------------------------------------------------
    # Dataset Length
    # ------------------------------------------------------

    def __len__(self):

        return len(self.vehicle_to_images)

    # ------------------------------------------------------
    # Load Image
    # ------------------------------------------------------

    def get_image(self, image_name):

        image_path = self.dataset_root / image_name

        img = Image.open(image_path)

        img = img.convert("RGB")

        return img

    # ------------------------------------------------------
    # Sample a random pair
    # ------------------------------------------------------

    def sample_pair(self):

        vehicle = random.choice(list(self.vehicle_to_images.keys()))

        images = self.vehicle_to_images[vehicle]

        img1_name, img2_name = random.sample(images, 2)

        img1 = self.get_image(img1_name)

        img2 = self.get_image(img2_name)

        ori1 = self.image_to_orientation.get(img1_name)

        ori2 = self.image_to_orientation.get(img2_name)

        print("\n-----------------------------------")
        print("Sampled Vehicle :", vehicle)
        print("-----------------------------------")

        print("\nImage 1")
        print("Name :", img1_name)
        print("Size :", img1.size)
        print("Orientation :", ori1)

        print()

        print("Image 2")
        print("Name :", img2_name)
        print("Size :", img2.size)
        print("Orientation :", ori2)

        print()

        if ori1 == ori2:
            print("Pair Type : SAME ORIENTATION")
        else:
            print("Pair Type : DIFFERENT ORIENTATION")

        return img1, img2, ori1, ori2, vehicle


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

if __name__ == "__main__":

    dataset = DatasetVeRi()

    print("\nDataset Length :", len(dataset))

    dataset.sample_pair()
