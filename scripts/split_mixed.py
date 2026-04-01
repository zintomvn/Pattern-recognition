#!/usr/bin/env python3
"""
Split *_mixed.txt files into _pos.txt (label=0) and _neg.txt (label=1).
Then update src/configs/small.yml with all 9 dataset path fields.
"""

from pathlib import Path
import re, yaml

ROOT = Path("/home/nesfan/Desktop/HCMUS/Nam3/HK2/NhanDang/Pattern-recognition")
SRC  = ROOT / "src" / "configs" / "small.yml"
OUT  = ROOT / "datasets" / "processed" / "meta_lists_small_basic"

# ── Files to split: (input_name, output_prefix) ──────────────────────────────
SPLITS = [
    ("train_small_mixed.txt", "train_small_mixed"),
    ("train_small_2.txt",     "train_small_2"),
    ("train_small_3.txt",     "train_small_3"),
    ("val_small_mixed.txt",   "val_small_mixed"),
    ("test_small_mixed.txt",  "test_small_mixed"),
]

def split_mixed(input_path: Path, prefix: str):
    """Read 'path label' lines, write pos/neg split files."""
    pos_lines, neg_lines = [], []
    for line in input_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # last space-separated token is the label
        label = line.rsplit(maxsplit=1)[-1]
        if label == "0":
            pos_lines.append(line)
        else:
            neg_lines.append(line)

    pos_path = OUT / f"{prefix}_pos.txt"
    neg_path = OUT / f"{prefix}_neg.txt"

    def write(path, lines):
        path.write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"  ✓ {path.name}  ({len(lines)} lines)")

    write(pos_path, sorted(set(pos_lines)))
    write(neg_path, sorted(set(neg_lines)))
    return pos_path, neg_path

# ── 1. Split all mixed files ───────────────────────────────────────────────────
print("=== Splitting mixed files ===")
results = {}   # prefix -> (pos_path, neg_path)

for input_name, prefix in SPLITS:
    input_path = OUT / input_name
    if not input_path.exists():
        print(f"  ⚠ {input_name} not found – skipped")
        continue
    print(f"\n{input_name}:")
    pos_path, neg_path = split_mixed(input_path, prefix)
    results[prefix] = (pos_path, neg_path)

# ── 2. Map prefixes → yaml field names ────────────────────────────────────────
FIELD_MAP = {
    "train_small_mixed": ("train_pos_list1_path", "train_neg_list1_path"),
    "train_small_2":     ("train_pos_list2_path", "train_neg_list2_path"),
    "train_small_3":     ("train_pos_list3_path", "train_neg_list3_path"),
    "val_small_mixed":   ("val_pos_list_path",    "val_neg_list_path"),
    "test_small_mixed":  ("test_pos_list_path",   "test_neg_list_path"),
}

# ── 3. Read & update small.yml ────────────────────────────────────────────────
print("\n=== Updating small.yml ===")
cfg = yaml.safe_load(SRC.read_text())
dg  = cfg["dataset"]["DG_Dataset"]

for prefix, (pos_path, neg_path) in results.items():
    pos_key, neg_key = FIELD_MAP[prefix]
    # Store as absolute paths
    dg[pos_key] = str(pos_path.resolve())
    dg[neg_key] = str(neg_path.resolve())
    print(f"  {pos_key:<28} = {pos_path.name}")
    print(f"  {neg_key:<28} = {neg_path.name}")

SRC.write_text(yaml.dump(cfg, allow_unicode=True, default_flow_style=False, sort_keys=False))
print(f"\n✅ small.yml updated → {SRC}")
