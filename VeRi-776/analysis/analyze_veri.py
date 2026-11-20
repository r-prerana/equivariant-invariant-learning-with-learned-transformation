import xml.etree.ElementTree as ET
from pathlib import Path

xml_file = Path("./data/VeRi/train_label.xml")

# Read the XML as text using the correct encoding
with open(xml_file, "r", encoding="gb2312") as f:
    xml_text = f.read()

# Remove the XML declaration
if xml_text.startswith("<?xml"):
    xml_text = xml_text.split("?>", 1)[1]

# Parse the XML string
root = ET.fromstring(xml_text)

items = root.find("Items")

print("Total training images :", items.attrib["number"])

# Count unique vehicle IDs
vehicle_ids = set()

for item in items:
    vehicle_ids.add(item.attrib["vehicleID"])

print("Total training images :", items.attrib["number"])
print("Unique vehicles :", len(vehicle_ids))

from collections import defaultdict

# ------------------------------------
# Count images per vehicle
# ------------------------------------
vehicle_image_count = defaultdict(int)

for item in items:
    vid = item.attrib["vehicleID"]
    vehicle_image_count[vid] += 1

print("\nFirst 10 vehicles and their image counts:\n")

for i, (vid, count) in enumerate(vehicle_image_count.items()):
    print(f"Vehicle {vid} : {count} images")

    if i == 9:
        break
from collections import defaultdict

# ------------------------------------
# Group images by vehicle
# ------------------------------------

vehicle_to_images = defaultdict(list)

for item in items:

    vehicle_id = item.attrib["vehicleID"]

    image_name = item.attrib["imageName"]

    vehicle_to_images[vehicle_id].append(image_name)

print("\nExample:\n")

example_vehicle = "0001"

print(f"Vehicle {example_vehicle}")

for img in vehicle_to_images[example_vehicle][:10]:
    print("   ", img)
