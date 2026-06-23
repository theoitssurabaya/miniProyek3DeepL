# ATM Custom Smart Eye (YOLOv8)

Fast, custom brain for ATM security. Uses a lightweight YOLOv8 model trained to detect faces, masks, hats, and sunglasses to prevent hidden faces during transactions.

## How Brain Works (Logic)

We trained a custom YOLOv8 model for this task.
* **YOLOv8**: Extremely fast object detection model, perfect for hardware with limited power like ATMs.
* **Custom Dataset**: Brain was fed thousands of pictures of masks, hats, and sunglasses from Kaggle to learn its specific job.

**The "Why"**:
We need speed. By gathering our own dataset and training a lightweight YOLO model, we get instant "DANGER" or "SAFE" signals without lag on edge devices.

## Installation & Setup

You must build the Python environment before cooking the brain. Here are the steps for your operating system.

### For Ubuntu (Linux)

1. Open your terminal.
2. Create and wake up the virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the tools:
   ```bash
   pip install -r requirements.txt
   ```

### For Windows

1. Open Command Prompt or PowerShell.
2. Create and wake up the virtual environment:
   ```cmd
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install the tools:
   ```cmd
   pip install -r requirements.txt
   ```

## Prepare Data
1. Download the dataset (Face Coverings & Accessories) in YOLOv8 format from Kaggle.
2. Unpack the zip file.
3. Put the folders (`train`, `valid`) and `data.yaml` inside a new `dataset/` directory in this project.
4. Open `dataset/data.yaml` and make sure the `path` points to the absolute path of your dataset directory.

## Usage

### 1. Teach Brain (Training)
Run the training script to cook the baby YOLO model:
```bash
python train_yolo.py
```
* **Logic**: This feeds all your pictures to the YOLO model for 20 epochs. A strong GPU will make this fast.
* This saves the smart brain weights into `runs/detect/train/weights/best.pt`.

### 2. Test Magic Eye (Inference)
Open webcam and test the security system:
```bash
python atm_yolo_eye.py
```
* **Logic**: The magic eye (webcam) takes pictures and the trained YOLO brain draws boxes.
* Put on a hat, mask, or sunglasses to trigger the red "DANGER" alert. Show a bare face to see the green "SAFE" alert.
