"""
Dataset Download Script for ATM Facial Occlusion Detection

Downloads dataset from Roboflow Universe in YOLOv8 format,
then restructures it into the project's dataset/ directory.

Usage:
    python download_dataset.py                           # Interactive
    ROBOFLOW_API_KEY=xxx python download_dataset.py      # Non-interactive
"""

import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

TARGET_CLASSES = ["open_face", "mask", "sunglasses", "hat"]

# Broad class name mapping: various Roboflow label names → our standardized label
CLASS_MAP = {
    # open_face variants
    "open_face": "open_face", "open face": "open_face",
    "no-mask": "open_face", "no_mask": "open_face", "nomask": "open_face",
    "without_mask": "open_face", "without mask": "open_face",
    "face": "open_face", "unmasked": "open_face",
    # mask variants
    "mask": "mask", "with_mask": "mask", "with mask": "mask",
    "face_mask": "mask", "face mask": "mask", "masked": "mask",
    # sunglasses variants
    "sunglasses": "sunglasses", "sun_glasses": "sunglasses",
    "glasses": "sunglasses", "eyeglasses": "sunglasses",
    "goggles": "sunglasses", "sunglass": "sunglasses",
    # hat variants
    "hat": "hat", "cap": "hat", "helmet": "hat",
    "hard_hat": "hat", "hard hat": "hat", "headwear": "hat",
}


def get_api_key() -> str:
    key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if key:
        return key
    print("=" * 60)
    print("  Roboflow API Key Required")
    print("=" * 60)
    print()
    print("  1. Sign up free at https://app.roboflow.com")
    print("  2. Go to Settings → API → Private API Key")
    print("  3. Paste it below")
    print()
    key = input("  API Key: ").strip()
    if not key:
        print("❌ No API key provided. Exiting.")
        sys.exit(1)
    return key


def get_dataset_config() -> tuple[str, str, int]:
    """Get Roboflow workspace/project/version — from env vars or interactive prompt."""
    ws = os.getenv("RF_WORKSPACE", "").strip()
    pj = os.getenv("RF_PROJECT", "").strip()
    ver = os.getenv("RF_VERSION", "").strip()

    if ws and pj and ver:
        return ws, pj, int(ver)

    print()
    print("=" * 60)
    print("  Roboflow Dataset Configuration")
    print("=" * 60)
    print()
    print("  Go to universe.roboflow.com, find a face occlusion dataset,")
    print("  and click 'Download Dataset' → 'Show download code'.")
    print("  You'll see code like:")
    print()
    print('    rf.workspace("WORKSPACE").project("PROJECT").version(N)')
    print()
    print("  Enter those values below.")
    print()
    print("  Suggested search terms on Roboflow Universe:")
    print("    - 'face mask sunglasses hat detection'")
    print("    - 'face covering detection'")
    print("    - 'facial accessories detection'")
    print()

    ws = input("  Workspace ID: ").strip()
    pj = input("  Project ID:   ").strip()
    ver = input("  Version (number): ").strip()

    if not all([ws, pj, ver]):
        print("❌ Missing configuration. Exiting.")
        sys.exit(1)

    return ws, pj, int(ver)


def download_from_roboflow(api_key: str, workspace: str, project: str, version: int) -> Path:
    """Download dataset from Roboflow in YOLOv8 format."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("  Installing roboflow SDK...")
        os.system(f"{sys.executable} -m pip install roboflow -q")
        from roboflow import Roboflow

    raw_dir = PROJECT_ROOT / "roboflow_raw"
    print(f"\n📦 Downloading: {workspace}/{project} v{version}")
    print(f"   Format: YOLOv8")
    print(f"   Target: {raw_dir}\n")

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ds = proj.version(version).download("yolov8", location=str(raw_dir))
    return Path(ds.location)


def remap_labels(raw_dir: Path):
    """
    Remap class IDs in YOLO .txt label files to match our TARGET_CLASSES.
    Reads data.yaml from the raw download, builds ID mapping, rewrites labels.
    """
    import yaml

    yaml_path = raw_dir / "data.yaml"
    if not yaml_path.exists():
        print("⚠️  No data.yaml found — skipping label remapping")
        return

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    old_names = cfg.get("names", [])
    if isinstance(old_names, dict):
        old_names = [old_names[k] for k in sorted(old_names.keys())]

    # Build old_id → new_id mapping
    id_map = {}
    unmapped = []
    for old_id, old_name in enumerate(old_names):
        normalized = old_name.lower().strip().replace(" ", "_")
        mapped = CLASS_MAP.get(normalized)
        if mapped and mapped in TARGET_CLASSES:
            id_map[old_id] = TARGET_CLASSES.index(mapped)
        else:
            unmapped.append(f"{old_id}:{old_name}")

    print(f"\n🔄 Class Remapping:")
    print(f"   Source classes:  {old_names}")
    print(f"   Target classes:  {TARGET_CLASSES}")
    print(f"   Mapping:         {id_map}")
    if unmapped:
        print(f"   ⚠️  Unmapped (dropped): {unmapped}")

    if not id_map:
        print("⚠️  No class mapping matches found — using raw labels as-is")
        return

    # Rewrite label files
    remapped_count = 0
    dropped_count = 0
    for split in ["train", "valid", "test"]:
        labels_dir = raw_dir / split / "labels"
        if not labels_dir.exists():
            continue
        for txt_file in labels_dir.glob("*.txt"):
            lines = txt_file.read_text().strip().split("\n")
            new_lines = []
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                old_id = int(parts[0])
                if old_id in id_map:
                    parts[0] = str(id_map[old_id])
                    new_lines.append(" ".join(parts))
                    remapped_count += 1
                else:
                    dropped_count += 1
            txt_file.write_text("\n".join(new_lines) + "\n" if new_lines else "")

    print(f"✅ Labels remapped: {remapped_count} kept, {dropped_count} dropped")


def restructure_dataset(raw_dir: Path):
    """Move downloaded data into the project's dataset/ directory structure."""
    for split in ["train", "valid", "test"]:
        for subdir in ["images", "labels"]:
            src = raw_dir / split / subdir
            dst = DATASET_DIR / split / subdir
            if not src.exists():
                continue
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if f.is_file():
                    shutil.copy2(f, dst / f.name)

    print("✅ Dataset copied to dataset/ directory")


def update_data_yaml():
    """Update training/data.yaml with absolute paths."""
    yaml_content = f"""# Auto-generated dataset config
train: {DATASET_DIR / 'train' / 'images'}
val: {DATASET_DIR / 'valid' / 'images'}
test: {DATASET_DIR / 'test' / 'images'}

nc: {len(TARGET_CLASSES)}

names:
"""
    for i, name in enumerate(TARGET_CLASSES):
        yaml_content += f"  {i}: {name}\n"

    yaml_path = Path(__file__).parent / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"✅ Updated {yaml_path}")


def print_summary():
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("  📊 Dataset Summary")
    print("=" * 60)
    total = 0
    for split in ["train", "valid", "test"]:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"
        n_img = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        total += n_img
        print(f"    {split:>5}: {n_img:>5} images, {n_lbl:>5} labels")
    print(f"    {'total':>5}: {total:>5} images")
    print()


def main():
    print()
    print("  🏧 ATM Facial Occlusion Detection — Dataset Downloader")
    print()

    api_key = get_api_key()
    workspace, project, version = get_dataset_config()

    raw_dir = download_from_roboflow(api_key, workspace, project, version)
    remap_labels(raw_dir)
    restructure_dataset(raw_dir)
    update_data_yaml()

    # Cleanup
    raw_path = PROJECT_ROOT / "roboflow_raw"
    if raw_path.exists():
        shutil.rmtree(raw_path)
        print("🗑️  Cleaned up raw download")

    print_summary()
    print("  🎉 Dataset ready! Next: python training/train.py")
    print()


if __name__ == "__main__":
    main()
