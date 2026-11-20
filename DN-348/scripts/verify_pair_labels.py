from pathlib import Path
from collections import Counter

ROOT = Path("./data/dn348")
SPLIT = ROOT / "train_test_split"

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
    # Official DN-348 split format:
    # identity/image.jpg
    return path_string.split("/")[0]


train_day = read_split("train_list_day.txt")
train_night = read_split("train_list_night.txt")

print("=" * 60)
print("DN-348 TRANSFORMATION LABEL VERIFICATION")
print("=" * 60)

print(f"Train Day images   : {len(train_day)}")
print(f"Train Night images : {len(train_night)}")

day_ids = {get_identity(x) for x in train_day}
night_ids = {get_identity(x) for x in train_night}

print(f"Train Day IDs      : {len(day_ids)}")
print(f"Train Night IDs    : {len(night_ids)}")

print()
print("Same training identities:", day_ids == night_ids)

# Count images per identity
day_counts = Counter(get_identity(x) for x in train_day)
night_counts = Counter(get_identity(x) for x in train_night)

print()
print("Images per identity:")
print(f"Day   : min={min(day_counts.values())}, "
      f"max={max(day_counts.values())}, "
      f"avg={sum(day_counts.values()) / len(day_counts):.2f}")

print(f"Night : min={min(night_counts.values())}, "
      f"max={max(night_counts.values())}, "
      f"avg={sum(night_counts.values()) / len(night_counts):.2f}")

print()
print("=" * 60)
print("PAIR TRANSFORMATION LABELS")
print("=" * 60)

for (src, tgt), label in LABELS.items():
    print(f"{label}: {LABEL_NAMES[label]:15s}  ({src} -> {tgt})")

print()
print("=" * 60)
print("EXAMPLE PAIRS")
print("=" * 60)

examples = [
    ("day", train_day[0], "day", train_day[1]),
    ("night", train_night[0], "night", train_night[1]),
    ("day", train_day[0], "night", train_night[0]),
    ("night", train_night[0], "day", train_day[0]),
]

for src_domain, src, tgt_domain, tgt in examples:
    label = LABELS[(src_domain, tgt_domain)]

    print()
    print(f"Source : {src_domain}/{src}")
    print(f"Target : {tgt_domain}/{tgt}")
    print(f"Label  : {label} ({LABEL_NAMES[label]})")

print()
print("=" * 60)
print("IMPORTANT")
print("=" * 60)
print("These labels represent pairwise Day/Night domain relationships.")
print("They are NOT exact physical illumination ground truth.")
print("The GT label will supervise the Transformation Encoder only.")
print("The Hypernetwork will receive t_pred, not the GT label.")
print("=" * 60)

