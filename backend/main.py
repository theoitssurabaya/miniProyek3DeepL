"""
ATM Facial Occlusion Detection — FastAPI WebSocket Server

Receives base64-encoded JPEG frames from the frontend, runs YOLOv8 inference,
and returns bounding boxes + access decision (GRANTED / DENIED / SCANNING).
"""

import base64
import json
import logging
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "model"
CUSTOM_WEIGHTS = MODEL_DIR / "best.pt"
FALLBACK_WEIGHTS = "yolov8n.pt"

CONFIDENCE_THRESHOLD = 0.5

# Classes that trigger DENIED status
OCCLUSION_CLASSES = {"mask", "sunglasses", "hat"}
ALLOWED_CLASS = "open_face"

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="ATM Occlusion Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model Loading ─────────────────────────────────────────────────────────────
def load_model() -> YOLO:
    weights = CUSTOM_WEIGHTS if CUSTOM_WEIGHTS.exists() else FALLBACK_WEIGHTS
    logger.info(f"Loading YOLOv8 model from: {weights}")
    model = YOLO(str(weights))
    logger.info(f"Model classes: {model.names}")
    return model

model = load_model()

# ── Inference Helpers ─────────────────────────────────────────────────────────
def decode_frame(data: str) -> np.ndarray:
    """Decode a base64-encoded JPEG string into a BGR numpy array."""
    img_bytes = base64.b64decode(data)
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def determine_status(detections: list[dict]) -> str:
    """
    GRANTED  — at least one open_face, zero occlusions
    DENIED   — any occlusion detected
    SCANNING — nothing relevant detected
    """
    has_open_face = False
    has_occlusion = False

    for det in detections:
        cls = det["class"]
        if cls == ALLOWED_CLASS:
            has_open_face = True
        if cls in OCCLUSION_CLASSES:
            has_occlusion = True

    if has_occlusion:
        return "DENIED"
    if has_open_face:
        return "GRANTED"
    return "SCANNING"


def run_inference(frame: np.ndarray) -> dict:
    results = model.predict(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class": cls_name,
                "confidence": round(conf, 3),
                "bbox": [round(v, 1) for v in [x1, y1, x2, y2]],
            })

    return {
        "status": determine_status(detections),
        "detections": detections,
    }

# ── WebSocket Endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected")

    try:
        while True:
            raw = await ws.receive_text()
            frame = decode_frame(raw)
            result = run_inference(frame)
            await ws.send_text(json.dumps(result))
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await ws.close()

# ── Serve Frontend ────────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
