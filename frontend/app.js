/**
 * ATM Security System — Client Application
 *
 * Captures webcam frames, streams to backend via WebSocket,
 * renders bounding box overlays and access status.
 */

// ── Config ──────────────────────────────────────────────────────────────────
const WS_URL = `ws://${location.host}/ws`;
const CAPTURE_FPS = 8;
const JPEG_QUALITY = 0.7;
const RECONNECT_DELAY_MS = 2000;

// ── DOM Refs ────────────────────────────────────────────────────────────────
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const overlay = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const otx = overlay.getContext("2d");

const statusCard = document.getElementById("statusCard");
const statusText = document.getElementById("statusText");
const statusSub = document.getElementById("statusSub");
const connectionBadge = document.getElementById("connectionBadge");
const fpsBadge = document.getElementById("fpsBadge");
const detectionsList = document.getElementById("detectionsList");
const logList = document.getElementById("logList");
const scanLine = document.getElementById("scanLine");
const btnClearLog = document.getElementById("btnClearLog");

const icons = {
    idle: document.getElementById("iconIdle"),
    scanning: document.getElementById("iconScanning"),
    granted: document.getElementById("iconGranted"),
    denied: document.getElementById("iconDenied"),
};

// ── State ───────────────────────────────────────────────────────────────────
let ws = null;
let captureInterval = null;
let frameCount = 0;
let lastFpsTime = performance.now();
let currentStatus = "idle";

// ── Color Map per Class ─────────────────────────────────────────────────────
const CLASS_COLORS = {
    open_face: { box: "#34d399", fill: "rgba(52,211,153,0.12)", dot: "#34d399" },
    mask: { box: "#f87171", fill: "rgba(248,113,113,0.12)", dot: "#f87171" },
    sunglasses: { box: "#fbbf24", fill: "rgba(251,191,36,0.12)", dot: "#fbbf24" },
    hat: { box: "#c084fc", fill: "rgba(192,132,252,0.12)", dot: "#c084fc" },
};
const DEFAULT_COLOR = { box: "#38bdf8", fill: "rgba(56,189,248,0.12)", dot: "#38bdf8" };

// ── Logging ─────────────────────────────────────────────────────────────────
function addLog(message, level = "info") {
    const entry = document.createElement("div");
    entry.className = `log-entry ${level}`;
    const time = new Date().toLocaleTimeString("en-GB");
    entry.innerHTML = `<span class="log-time">[${time}]</span> ${message}`;
    logList.appendChild(entry);
    logList.scrollTop = logList.scrollHeight;

    // Keep last 50 entries
    while (logList.children.length > 50) logList.removeChild(logList.firstChild);
}

btnClearLog.addEventListener("click", () => { logList.innerHTML = ""; });

// ── Status Management ───────────────────────────────────────────────────────
function setStatus(status, subtitle) {
    if (status === currentStatus && statusSub.textContent === subtitle) return;
    currentStatus = status;

    statusCard.dataset.status = status;
    statusText.textContent = status.toUpperCase();
    statusSub.textContent = subtitle;

    Object.values(icons).forEach(i => i.classList.remove("active"));
    if (icons[status]) icons[status].classList.add("active");

    scanLine.classList.toggle("active", status === "scanning");
}

function setConnection(connected) {
    connectionBadge.classList.toggle("connected", connected);
    connectionBadge.querySelector(".conn-text").textContent = connected ? "Connected" : "Disconnected";
}

// ── Webcam Setup ────────────────────────────────────────────────────────────
async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
            audio: false,
        });
        video.srcObject = stream;
        await video.play();

        // Match canvas to video resolution
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;

        addLog("Camera initialized", "success");
        setStatus("scanning", "Looking for face...");
        connectWebSocket();
    } catch (err) {
        addLog(`Camera error: ${err.message}`, "error");
        setStatus("idle", "Camera access denied");
    }
}

// ── WebSocket ───────────────────────────────────────────────────────────────
function connectWebSocket() {
    if (ws && ws.readyState <= WebSocket.OPEN) return;

    addLog(`Connecting to ${WS_URL}...`);
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        setConnection(true);
        addLog("WebSocket connected", "success");
        startCapture();
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleResult(data);
    };

    ws.onclose = () => {
        setConnection(false);
        stopCapture();
        addLog("Connection lost — reconnecting...", "warn");
        setTimeout(connectWebSocket, RECONNECT_DELAY_MS);
    };

    ws.onerror = () => {
        addLog("WebSocket error", "error");
    };
}

// ── Frame Capture & Streaming ───────────────────────────────────────────────
function startCapture() {
    if (captureInterval) return;
    captureInterval = setInterval(captureAndSend, 1000 / CAPTURE_FPS);
}

function stopCapture() {
    clearInterval(captureInterval);
    captureInterval = null;
}

function captureAndSend() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    // Draw current video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Extract as JPEG base64, strip the data-url header
    const dataUrl = canvas.toDataURL("image/jpeg", JPEG_QUALITY);
    const base64 = dataUrl.split(",")[1];

    ws.send(base64);
    updateFps();
}

function updateFps() {
    frameCount++;
    const now = performance.now();
    const delta = now - lastFpsTime;
    if (delta >= 1000) {
        const fps = Math.round((frameCount / delta) * 1000);
        fpsBadge.textContent = `${fps} FPS`;
        frameCount = 0;
        lastFpsTime = now;
    }
}

// ── Result Handling ─────────────────────────────────────────────────────────
function handleResult(data) {
    const { status, detections } = data;

    // Update status
    const subtitleMap = {
        GRANTED: "Clear face detected — access allowed",
        DENIED: "Facial occlusion detected — access blocked",
        SCANNING: "Looking for face...",
    };
    setStatus(status.toLowerCase(), subtitleMap[status] || "Processing...");

    // Log status changes
    if (status === "DENIED") {
        const occluders = detections.filter(d => d.class !== "open_face").map(d => d.class);
        addLog(`DENIED — ${occluders.join(", ")} detected`, "error");
    } else if (status === "GRANTED") {
        addLog("GRANTED — clear face verified", "success");
    }

    renderDetections(detections);
    drawOverlay(detections, status);
}

// ── Detection List Rendering ────────────────────────────────────────────────
function renderDetections(detections) {
    if (!detections.length) {
        detectionsList.innerHTML = '<div class="empty-state">No detections</div>';
        return;
    }

    detectionsList.innerHTML = detections.map(d => {
        const color = (CLASS_COLORS[d.class] || DEFAULT_COLOR).dot;
        const conf = (d.confidence * 100).toFixed(1);
        return `
            <div class="det-item">
                <span class="det-dot" style="background:${color}"></span>
                <span class="det-class">${d.class.replace("_", " ")}</span>
                <span class="det-conf">${conf}%</span>
            </div>`;
    }).join("");
}

// ── Bounding Box Overlay Rendering ──────────────────────────────────────────
function drawOverlay(detections, status) {
    otx.clearRect(0, 0, overlay.width, overlay.height);

    // Semi-transparent screen tint on DENIED
    if (status === "DENIED") {
        otx.fillStyle = "rgba(248, 113, 113, 0.06)";
        otx.fillRect(0, 0, overlay.width, overlay.height);
    }

    for (const det of detections) {
        const [x1, y1, x2, y2] = det.bbox;
        const w = x2 - x1;
        const h = y2 - y1;
        const colors = CLASS_COLORS[det.class] || DEFAULT_COLOR;

        // Filled background
        otx.fillStyle = colors.fill;
        otx.fillRect(x1, y1, w, h);

        // Border
        otx.strokeStyle = colors.box;
        otx.lineWidth = 2.5;
        otx.strokeRect(x1, y1, w, h);

        // Corner accents (thicker L-shaped corners)
        const cornerLen = Math.min(w, h) * 0.15;
        otx.lineWidth = 4;
        drawCorners(x1, y1, x2, y2, cornerLen, colors.box);

        // Label
        const label = `${det.class.replace("_", " ")} ${(det.confidence * 100).toFixed(0)}%`;
        drawLabel(x1, y1, label, colors.box);
    }
}

function drawCorners(x1, y1, x2, y2, len, color) {
    otx.strokeStyle = color;

    // Top-left
    otx.beginPath(); otx.moveTo(x1, y1 + len); otx.lineTo(x1, y1); otx.lineTo(x1 + len, y1); otx.stroke();
    // Top-right
    otx.beginPath(); otx.moveTo(x2 - len, y1); otx.lineTo(x2, y1); otx.lineTo(x2, y1 + len); otx.stroke();
    // Bottom-left
    otx.beginPath(); otx.moveTo(x1, y2 - len); otx.lineTo(x1, y2); otx.lineTo(x1 + len, y2); otx.stroke();
    // Bottom-right
    otx.beginPath(); otx.moveTo(x2 - len, y2); otx.lineTo(x2, y2); otx.lineTo(x2, y2 - len); otx.stroke();
}

function drawLabel(x, y, text, bgColor) {
    otx.font = "bold 13px Inter, sans-serif";
    const metrics = otx.measureText(text);
    const pad = 6;
    const lh = 20;
    const ly = y - lh - 4;

    // Background pill
    otx.fillStyle = bgColor;
    otx.beginPath();
    otx.roundRect(x, ly, metrics.width + pad * 2, lh, 4);
    otx.fill();

    // Text
    otx.fillStyle = "#000";
    otx.fillText(text, x + pad, ly + 14);
}

// ── Bootstrap ───────────────────────────────────────────────────────────────
addLog("System initializing...");
initCamera();
