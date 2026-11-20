from pathlib import Path
import random

ROOT = Path("./data/dn348")
SPLIT = ROOT / "train_test_split"
OUT = Path("./data/splits")

SEED = 348
PAIRS_PER_ID_PER_TYPE = 20

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
    with open(SPLIT / filename, "r") as f:
        return [x.strip() for x in f if x.strip()]


def identity(x):
    return x.split("/")[0]


def group_by_identity(items):
    groups = {}

    for item in items:
        groups.setdefault(identity(item), []).append(item)

    return groups


def sample_pairs(source_items, target_items, source_domain, target_domain):
    source_groups = group_by_identity(source_items)
    target_groups = group_by_identity(target_items)

    pairs = []

    common_ids = sorted(set(source_groups) & set(target_groups))

    for vid in common_ids:
        src = source_groups[vid]
        tgt = target_groups[vid]

        possible = [(a, b) for a in src for b in tgt]

        random.shuffle(possible)

        n = min(PAIRS_PER_ID_PER_TYPE, len(possible))

        label = LABELS[(source_domain, target_domain)]

        for a, b in possible[:n]:
            pairs.append(
                (
                    f"{source_domain}/{a}",
                    f"{target_domain}/{b}",
                    label,
                )
            )

    return pairs


random.seed(SEED)

train_day = read_split("train_list_day.txt")
train_night = read_split("train_list_night.txt")

all_pairs = []

all_pairs += sample_pairs(
    train_day,
    train_day,
    "day",
    "day",
)

all_pairs += sample_pairs(
    train_night,
    train_night,
    "night",
    "night",
)

all_pairs += sample_pairs(
    train_day,
    train_night,
    "day",
    "night",
)

all_pairs += sample_pairs(
    train_night,
    train_day,
    "night",
    "day",
)

random.shuffle(all_pairs)

OUT.mkdir(parents=True, exist_ok=True)

output = OUT / "train_pairs.txt"

with open(output, "w") as f:
    for a, b, label in all_pairs:
        f.write(f"{a} {b} {label}\n")

print("=" * 60)
print("DN-348 TRAIN PAIRS")
print("=" * 60)

print(f"Total pairs: {len(all_pairs)}")

counts = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
}

for _, _, label in all_pairs:
    counts[label] += 1

for label in range(4):
    print(
        f"{label}: {LABEL_NAMES[label]:15s} "
        f"{counts[label]} pairs"
    )

print()
print(f"Saved to: {output}")
