from pathlib import Path
from collections import defaultdict

anno_file = Path("./data/VehicleReIDKeyPointData/keypoint_train.txt")

# Dictionary:
# vehicle_id -> set of orientations
vehicle_orientations = defaultdict(set)

with open(anno_file) as f:
    for line in f:

        tokens = line.strip().split()

        image_path = tokens[0]
        orientation = int(tokens[-1])

        # Example filename:
        # 0190_c014_00051260_1.jpg
        filename = Path(image_path).name

        vehicle_id = filename.split("_")[0]

        vehicle_orientations[vehicle_id].add(orientation)

print("First 10 vehicles:\n")

for i, (vid, orientations) in enumerate(vehicle_orientations.items()):

    print(f"Vehicle {vid} -> {sorted(list(orientations))}")

    if i == 9:
        break
