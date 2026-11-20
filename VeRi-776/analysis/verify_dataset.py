from pathlib import Path

# Dataset paths
veri_root = Path("./data/VeRi")
anno_file = Path("./data/VehicleReIDKeyPointData/keypoint_test.txt")

print("Dataset exists :", veri_root.exists())
print("Annotation exists :", anno_file.exists())

print("\nReading first annotation...\n")

with open(anno_file, "r") as f:
    first_line = f.readline().strip()

print(first_line)

print("\nChecking image path...\n")

tokens = first_line.split()

relative_path = tokens[0]

print("Relative Path :", relative_path)

# Remove "VeRi/" because veri_root already points to the VeRi folder
image_path = veri_root / relative_path.replace("VeRi/", "")

print("Full Path :", image_path)
print("Image Exists :", image_path.exists())

from PIL import Image

print("\nLoading image...\n")

img = Image.open(image_path)

print("Image Size :", img.size)
print("Image Mode :", img.mode)

print("\nDecoding annotation...\n")

coords = list(map(int, tokens[1:-1]))

orientation = int(tokens[-1])

print("Orientation :", orientation)
print("Number of coordinate values :", len(coords))
