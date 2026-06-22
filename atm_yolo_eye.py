import cv2
from ultralytics import YOLO

# Logic: Load our trained custom brain
print("Waking up trained ATM brain...")
try:
    model = YOLO("runs/detect/train/weights/best.pt")
except Exception as e:
    print("Error: Could not find smart brain. Did you run train_yolo.py first?")
    exit()

# Labels from data.yaml
class_names = {
    0: "cap",
    1: "mask",
    2: "person",
    3: "scarf-kerchief",
    4: "sunglasses"
}

# Logic: Things that cover face are danger
DANGER_CLASSES = ["cap", "mask", "sunglasses", "scarf-kerchief"]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open magic eye.")
    exit()

print("Magic eye opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Brain looks at picture
    results = model(frame, verbose=False)
    
    is_danger = False
    danger_items = []
    
    # Process what brain sees
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # We only care if brain is confident
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                name = class_names.get(cls, "unknown")
                
                # Logic: Is item a danger?
                if name in DANGER_CLASSES:
                    is_danger = True
                    danger_items.append(name)
                    color = (0, 0, 255) # Red for danger
                else:
                    color = (0, 255, 0) # Green for person
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} {box.conf[0]:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Logic: Set status text
    if is_danger:
        status_text = "DANGER: " + ", ".join(set(danger_items))
        status_color = (0, 0, 255)
    else:
        status_text = "SAFE: Face clear"
        status_color = (0, 255, 0)
        
    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

    cv2.imshow("ATM Smart Eye", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
