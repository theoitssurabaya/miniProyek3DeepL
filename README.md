# ATM Custom Smart Eye (YOLOv8)

Fast, custom brain for ATM security. Uses a lightweight YOLOv8 model trained to detect faces, masks, hats, and sunglasses to prevent hidden faces during transactions.

## How Brain Works (Logic)

We trained a custom YOLOv8 model for this task.
* **YOLOv8**: Extremely fast object detection model, perfect for hardware with limited power like ATMs.
* **Custom Dataset**: Brain was fed thousands of pictures of masks, hats, and sunglasses from Kaggle to learn its specific job.

**The "Why"**:
We need speed. By gathering our own dataset and training a lightweight YOLO model, we get instant "DANGER" or "SAFE" signals without lag on edge devices.

## Installation & Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies (requires ultralytics for YOLO):
   ```bash
   pip install ultralytics opencv-python
   ```
3. Prepare Data:
   Download the Kaggle dataset (Face Coverings & Accessories) in YOLO format and place it in the `dataset/` directory.

## Usage

### 1. Teach Brain (Training)
Run the training script to cook the baby YOLO model:
```bash
python train_yolo.py
```
This saves the smart brain weights into `runs/detect/train/weights/best.pt`.

### 2. Test Magic Eye (Inference)
Open webcam and test the security system:
```bash
python atm_yolo_eye.py
```
Put on a hat or mask to trigger the red DANGER alert.
