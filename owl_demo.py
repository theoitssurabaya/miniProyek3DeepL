import cv2
import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# Load the OWL-ViT model and processor
print("Waking up giant OWL-ViT brain... (This may take a while to download weights if first time)")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Brain running on: {device.upper()}")
model.to(device)
model.eval()

# Define the text prompts for detection
texts = [["a face", "a medical mask", "a hat", "sunglasses"]]

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

print("Camera opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame.")
        break
        
    # Convert BGR to RGB for PIL
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Preprocess the image and text
    inputs = processor(text=texts, images=pil_image, return_tensors="pt").to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Post-process results
    target_sizes = torch.tensor([pil_image.size[::-1]])
    results = processor.image_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    
    i = 0  # We process one image at a time
    text = texts[i]
    
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    
    is_danger = False
    danger_reason = []
    
    # Draw boxes
    for box, score, label in zip(boxes, scores, labels):
        # We only care about somewhat confident predictions
        if score > 0.15:
            box = [round(i, 2) for i in box.tolist()]
            x1, y1, x2, y2 = map(int, box)
            
            label_name = text[label.item()]
            
            # Check for danger items
            if label_name in ["a medical mask", "a hat", "sunglasses"]:
                is_danger = True
                danger_reason.append(label_name)
                color = (0, 0, 255) # Red for danger in BGR
            else:
                color = (0, 255, 0) # Green for face
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label_name} {round(score.item(), 2)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    # Draw global status
    status_text = "DANGER: " + ", ".join(set(danger_reason)) if is_danger else "SAFE: clear face"
    status_color = (0, 0, 255) if is_danger else (0, 255, 0)
    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

    # Show the frame
    cv2.imshow("ATM Python Eye", frame)
    
    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
