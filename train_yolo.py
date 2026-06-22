from ultralytics import YOLO

# Logic: Load small fast YOLOv8 brain
print("Waking up baby YOLO brain...")
model = YOLO("yolov8n.pt")  # build from pre-trained weights

# Logic: Feed it our downloaded pictures so it learn "mask", "hat", "sunglasses"
print("Feeding pictures to brain...")
results = model.train(
    data="/home/theo/Documents/miniProyek3DeepL/dataset/data.yaml",
    epochs=20, # How many times brain see data. Adjust if brain stupid.
    imgsz=640, # Picture size
    device="0" # Use GPU 0 for speed
)

print("Brain is now smart! Saved in 'runs/detect/train/weights/best.pt'")
