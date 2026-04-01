#!/usr/bin/env python3
"""
Preprocess depth maps for all datasets listed in small.yml using PRNet.

Files listed in each *_pos.txt / *_neg.txt list are copied to a temp dir,
PRNet runs on the temp dir, output goes to the depth folder, then tmp is deleted.
This avoids processing unrelated files that may exist in the same directory.

Usage (from src/ directory):
    python preprocess_depth.py                     # process all
    python preprocess_depth.py --train            # train only
    python preprocess_depth.py --val              # val only
    python preprocess_depth.py --test             # test only
    python preprocess_depth.py --force             # re-run even if depth files exist
"""

import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

# Resolve paths relative to this script's location
SCRIPT_DIR   = Path(__file__).parent.resolve()
PRNET_DIR    = SCRIPT_DIR.parent / "PRNet"    # ../PRNet
CONFIG_DIR   = SCRIPT_DIR / "configs"
TMP_ROOT     = SCRIPT_DIR / ".." / "data" / "processed" / "tmp"   # data/processed/tmp


# ── Depth folder mapping ───────────────────────────────────────────────────────
DEPTH_SUBDIR_MAP = {
    "CASIA_database": "CASIA_database_depth",
    "MSU-MFSD":       "MSU-MFSD_depth",
    "NUAA":           "NUAA_depth",
    "ReplayAttack":   "ReplayAttack_depth",
}


# ── CLI group → yaml key list ──────────────────────────────────────────────────
ALL_LIST_KEYS = [
    "train_pos_list1_path", "train_neg_list1_path",
    "train_pos_list2_path", "train_neg_list2_path",
    "train_pos_list3_path", "train_neg_list3_path",
    "val_pos_list_path",    "val_neg_list_path",
    "test_pos_list_path",   "test_neg_list_path",
]
TRAIN_KEYS = [k for k in ALL_LIST_KEYS if k.startswith("train")]
VAL_KEYS   = [k for k in ALL_LIST_KEYS if k.startswith("val")]
TEST_KEYS  = [k for k in ALL_LIST_KEYS if k.startswith("test")]


# ─────────────────────────────────────────────────────────────────────────────
def _read_yml(config_path: Path) -> dict:
    from omegaconf import OmegaConf
    return OmegaConf.load(config_path)


def _resolve(raw_path: str) -> Path:
    """Resolve path: absolute as-is; relative to SCRIPT_DIR."""
    p = Path(raw_path)
    return p.resolve() if p.is_absolute() else (SCRIPT_DIR / p).resolve()


def _find_depth_path(rgb_path: str) -> str:
    for key, depth_name in DEPTH_SUBDIR_MAP.items():
        if key.lower() in rgb_path.lower():
            return rgb_path.replace(key, depth_name, 1)
    raise ValueError(
        f"Cannot map RGB path → depth folder: {rgb_path}\n"
        "Add the dataset name to DEPTH_SUBDIR_MAP in preprocess_depth.py"
    )


def _collect_groups(list_keys: list, cfg_dg: dict) -> dict:
    """
    Read every list file and group images by depth_dir.

    Returns:
        Dict[depth_dir: Path] → (rgb_paths: list[Path], depth_dir: Path)
        Only the depth_dir is needed for PRNet output.
    """
    groups = defaultdict(list)   # depth_dir → list of rgb Paths

    for key in list_keys:
        list_file = _resolve(cfg_dg[key])

        if not list_file.exists():
            print(f"[SKIP] list file not found: {list_file}")
            continue

        for line in list_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue

            img_path_str = line.split()[0]
            depth_path_str = _find_depth_path(img_path_str)

            rgb_path  = _resolve(img_path_str)
            depth_dir = _resolve(depth_path_str).parent

            groups[depth_dir].append(rgb_path)

    return dict(groups)


def _rename_depth_files(output_dir: Path) -> int:
    """Rename frame_XXXX_depth.jpg → frame_XXXX.jpg in output_dir."""
    renamed = 0
    for f in output_dir.iterdir():
        if "_depth." in f.name:
            f.rename(f.parent / f.name.replace("_depth.", "."))
            renamed += 1
    return renamed


def _run_prnet(rgb_paths: list, depth_dir: Path, tmp_dir: Path, force: bool = False) -> bool:
    """
    Copy listed RGB files to tmp_dir, run PRNet on tmp → depth_dir, then delete tmp.

    Returns True if PRNet ran, False if skipped.
    """
    # ── 1. Check if already done ──────────────────────────────────────────────
    if not force and depth_dir.exists():
        existing = [
            f for f in depth_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png") and "_depth" not in f.name
        ]
        if existing:
            print(f"[SKIP] already has {len(existing)} depth files: {depth_dir}")
            return False

    # ── 2. Create tmp dir ─────────────────────────────────────────────────────
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # ── 3. Copy only listed files into tmp ────────────────────────────────────
    for rgb_path in rgb_paths:
        if not rgb_path.exists():
            print(f"[WARN] file not found (skipping): {rgb_path}")
            continue
        shutil.copy2(rgb_path, tmp_dir / rgb_path.name)

    print(f"[RUN ] {tmp_dir}  ({len(rgb_paths)} files)")
    print(f"      → {depth_dir}")

    # ── 4. Run PRNet ──────────────────────────────────────────────────────────
    cmd = [
        sys.executable,
        "demo.py",
        "-i", str(tmp_dir),
        "-o", str(depth_dir),
        "--isDlib",  "True",
        "--isDepth", "True",
        "--isMat", "False"
    ]
    res = subprocess.run(cmd, cwd=str(PRNET_DIR), capture_output=False)

    # ── 5. Delete tmp ──────────────────────────────────────────────────────────
    shutil.rmtree(tmp_dir)

    if res.returncode != 0:
        print(f"[ERROR] PRNet failed: {depth_dir}")
        return False

    n = _rename_depth_files(depth_dir)
    print(f"[OK  ] {n} depth files → {depth_dir.name}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
def run(list_keys: list, config_path: Path, force: bool = False):
    print(f"Config: {config_path}\n")
    cfg = _read_yml(config_path)
    dg  = cfg.dataset.DG_Dataset

    groups = _collect_groups(list_keys, dg)

    if not groups:
        print("[ERROR] No dataset directories found. Check the list files.")
        return

    print(f"{'─'*60}")
    print(f"Found {len(groups)} depth directories to process:")
    for depth_dir, rgb_paths in groups.items():
        print(f"  {depth_dir.name:40s}  ({len(rgb_paths)} images)")
    print(f"{'─'*60}\n")

    run_count = 0
    for depth_dir, rgb_paths in groups.items():
        if _run_prnet(rgb_paths, depth_dir, TMP_ROOT, force=force):
            run_count += 1

    print(f"\n✅ Done. PRNet ran on {run_count}/{len(groups)} directories.")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Preprocess depth maps using PRNet")
    parser.add_argument("--config", type=str, default="small.yml",
                        help="Config filename in src/configs/ (default: small.yml)")
    parser.add_argument("--train", action="store_true", help="Process train sets only")
    parser.add_argument("--val",   action="store_true", help="Process val set only")
    parser.add_argument("--test",  action="store_true", help="Process test set only")
    parser.add_argument("--all",   action="store_true", help="Process everything (default)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run PRNet even if depth files already exist")
    args = parser.parse_args()

    config_path = CONFIG_DIR / args.config

    if args.all or not any([args.train, args.val, args.test]):
        keys = ALL_LIST_KEYS
    else:
        keys = []
        if args.train: keys += TRAIN_KEYS
        if args.val:   keys += VAL_KEYS
        if args.test:  keys += TEST_KEYS

    run(keys, config_path=config_path, force=args.force)


if __name__ == "__main__":
    main()
