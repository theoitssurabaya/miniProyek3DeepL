"""
YOLOv8 Custom Training Script

Trains a YOLOv8 model on facial occlusion dataset and copies
the best weights to the backend model directory.
"""

import shutil
from pathlib import Path
from ultralytics import YOLO

# ── Training Config ───────────────────────────────────────────────────────────
BASE_MODEL = "yolov8n.pt"
DATA_YAML = Path(__file__).parent / "data.yaml"
EPOCHS = 1
IMG_SIZE = 640
BATCH_SIZE = 16
PROJECT_NAME = "atm_occlusion"
EXPERIMENT_NAME = "yolov8n_custom"

# Where the backend expects the trained weights
EXPORT_PATH = Path(__file__).parent.parent / "backend" / "model" / "best.pt"


def main():
    model = YOLO(BASE_MODEL)

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        exist_ok=True,
        patience=10,
        save=True,
        plots=True,
    )

    # Copy best weights to backend/model/
    best_weights = Path(PROJECT_NAME) / EXPERIMENT_NAME / "weights" / "best.pt"
    if best_weights.exists():
        EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, EXPORT_PATH)
        print(f"✅ Best weights copied to {EXPORT_PATH}")
    else:
        print("⚠️  best.pt not found — check training output directory")

    return results


if __name__ == "__main__":
    main()
