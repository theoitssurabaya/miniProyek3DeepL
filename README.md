# ATM Facial Occlusion Detection Security System

Real-time web-based ATM security prototype that uses **YOLOv8** to detect facial occlusions and control transaction access.

## Architecture

```
Browser (Webcam) ──WebSocket──► FastAPI Server ──► YOLOv8 Inference
       ▲                              │
       └──── JSON (bbox + status) ◄───┘
```

## Target Classes

| Class | Access |
|---|---|
| `open_face` | ✅ Granted |
| `mask` | ❌ Denied |
| `sunglasses` | ❌ Denied |
| `hat` | ❌ Denied |

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the Server

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Open in Browser

Navigate to `http://localhost:8000` — allow camera access when prompted.

## Training a Custom Model

1. Place your YOLO-format dataset in the `dataset/` directory.
2. Update `training/data.yaml` paths if needed.
3. Run training:

```bash
cd training
python train.py
```

The best weights will be automatically copied to `backend/model/best.pt`.

## Project Structure

```
├── backend/
│   ├── main.py              # FastAPI WebSocket server
│   ├── requirements.txt
│   └── model/               # Trained weights (best.pt)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── training/
│   ├── train.py
│   └── data.yaml
└── dataset/                  # YOLO-format dataset
    ├── train/
    ├── valid/
    └── test/
```

## Tech Stack

- **Frontend:** Vanilla HTML/CSS/JS, WebSocket API, Canvas API
- **Backend:** FastAPI, Uvicorn, WebSockets
- **Model:** YOLOv8 (Ultralytics)