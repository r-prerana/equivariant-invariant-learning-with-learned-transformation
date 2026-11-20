from pathlib import Path
from PIL import Image, ImageDraw

# -----------------------------
# Dataset paths
# -----------------------------
veri_root = Path("./data/VeRi")
anno_file = Path("./data/VehicleReIDKeyPointData/keypoint_test.txt")

# -----------------------------
# Read first annotation
# -----------------------------
with open(anno_file, "r") as f:
    line = f.readline().strip()

tokens = line.split()

relative_path = tokens[0]

image_path = veri_root / relative_path.replace("VeRi/", "")

coords = list(map(int, tokens[1:-1]))

orientation = int(tokens[-1])

print("Image :", image_path.name)
print("Orientation :", orientation)

# -----------------------------
# Load image
# -----------------------------
img = Image.open(image_path)

draw = ImageDraw.Draw(img)
# -----------------------------
# Draw the 20 keypoints
# -----------------------------
for i in range(0, len(coords), 2):

    x = coords[i]
    y = coords[i + 1]

    # Skip invisible keypoints
    if x == -1 or y == -1:
        continue

    r = 3
    draw.ellipse(
        (x-r, y-r, x+r, y+r),
        fill="red"
    )

# -----------------------------
# Save the output image
# -----------------------------
output_path = Path("./SIE_VeRi/outputs/keypoints_demo.jpg")

img.save(output_path)

print("\nSaved to:")
print(output_path)
