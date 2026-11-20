from pathlib import Path
import random

ROOT = Path("./data/dn348")
SPLIT = ROOT / "train_test_split"
OUT = Path("./data/splits")

SEED = 348

# Number of image pairs generated per identity
# for each of the four transformation types.
PAIRS_PER_ID_PER_TYPE = 20

# 200 official training identities:
# 160 -> encoder training
# 40  -> encoder validation
TRAIN_ID_RATIO = 0.8

LABELS = {
    ("day", "day"): 0,
    ("night", "night"): 1,
    ("day", "night"): 2,
    ("night", "day"): 3,
}

LABEL_NAMES = {
    0: "DAY_TO_DAY",
    1: "NIGHT_TO_NIGHT",
    2: "DAY_TO_NIGHT",
    3: "NIGHT_TO_DAY",
}


def read_split(filename):
    path = SPLIT / filename

    with open(path, "r") as f:
        return [x.strip() for x in f if x.strip()]


def get_identity(path_string):
    return path_string.split("/")[0]


def group_by_identity(items):
    groups = {}

    for item in items:
        vid = get_identity(item)
        groups.setdefault(vid, []).append(item)

    return groups


def generate_pairs(
    day_items,
    night_items,
    selected_ids,
):
    day_groups = group_by_identity(day_items)
    night_groups = group_by_identity(night_items)

    pairs = []

    for vid in sorted(selected_ids):

        day_images = day_groups[vid]
        night_images = night_groups[vid]

        pair_types = [
            ("day", day_images, "day", day_images),
            ("night", night_images, "night", night_images),
            ("day", day_images, "night", night_images),
            ("night", night_images, "day", day_images),
        ]

        for src_domain, src_images, tgt_domain, tgt_images in pair_types:

            possible_pairs = [
                (a, b)
                for a in src_images
                for b in tgt_images
            ]

            random.shuffle(possible_pairs)

            n = min(
                PAIRS_PER_ID_PER_TYPE,
                len(possible_pairs),
            )

            label = LABELS[(src_domain, tgt_domain)]

            for a, b in possible_pairs[:n]:

                pairs.append(
                    (
                        f"{src_domain}/{a}",
                        f"{tgt_domain}/{b}",
                        label,
                    )
                )

    random.shuffle(pairs)

    return pairs


def save_pairs(path, pairs):

    with open(path, "w") as f:
        for a, b, label in pairs:
            f.write(f"{a} {b} {label}\n")


random.seed(SEED)

train_day = read_split("train_list_day.txt")
train_night = read_split("train_list_night.txt")

day_groups = group_by_identity(train_day)
night_groups = group_by_identity(train_night)

all_ids = sorted(
    set(day_groups.keys()) &
    set(night_groups.keys())
)

print("=" * 70)
print("DN-348 TRANSFORMATION ENCODER TRAIN / VALIDATION SPLIT")
print("=" * 70)

print(f"Total available training identities : {len(all_ids)}")

# Shuffle identities once using the fixed seed.
shuffled_ids = all_ids.copy()
random.shuffle(shuffled_ids)

num_train_ids = int(
    len(shuffled_ids) * TRAIN_ID_RATIO
)

train_ids = sorted(
    shuffled_ids[:num_train_ids]
)

val_ids = sorted(
    shuffled_ids[num_train_ids:]
)

print(f"Transformation Encoder train IDs   : {len(train_ids)}")
print(f"Transformation Encoder validation IDs: {len(val_ids)}")

print()
print("Identity overlap:")
print("Train ∩ Val =", len(set(train_ids) & set(val_ids)))

print()

# Generate pairs.
train_pairs = generate_pairs(
    train_day,
    train_night,
    train_ids,
)

val_pairs = generate_pairs(
    train_day,
    train_night,
    val_ids,
)

# Save.
train_path = OUT / "train_pairs.txt"
val_path = OUT / "val_pairs.txt"

save_pairs(train_path, train_pairs)
save_pairs(val_path, val_pairs)

print("=" * 70)
print("PAIR COUNTS")
print("=" * 70)

print(f"Training pairs   : {len(train_pairs)}")
print(f"Validation pairs : {len(val_pairs)}")

print()
print("Training label distribution:")

train_counts = {0: 0, 1: 0, 2: 0, 3: 0}

for _, _, label in train_pairs:
    train_counts[label] += 1

for label in range(4):
    print(
        f"{label}: {LABEL_NAMES[label]:15s} "
        f"{train_counts[label]}"
    )

print()
print("Validation label distribution:")

val_counts = {0: 0, 1: 0, 2: 0, 3: 0}

for _, _, label in val_pairs:
    val_counts[label] += 1

for label in range(4):
    print(
        f"{label}: {LABEL_NAMES[label]:15s} "
        f"{val_counts[label]}"
    )

print()
print(f"Saved training pairs   : {train_path}")
print(f"Saved validation pairs : {val_path}")

print()
print("=" * 70)
print("IMPORTANT")
print("=" * 70)
print("Train and validation identities are disjoint.")
print("The official 148 DN-348 test identities remain untouched.")
print("=" * 70)
