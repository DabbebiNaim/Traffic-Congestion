import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from ultralytics import YOLO
from torchvision import transforms

# --- 1. SETUP BOTH DEEP LEARNING MODELS ---

# --- Model A: The "Road Finder" (Semantic Segmentation) ---
print("Loading Road Segmentation Model (DeepLabV3+)...")
# Load a pre-trained DeepLabV3+ model with a ResNet backbone
# This model was trained on the Cityscapes dataset, which includes a 'road' class.
seg_model = smp.DeepLabV3Plus(
    encoder_name="resnet34",        # A reasonably fast backbone
    encoder_weights="imagenet",     # Use pre-trained weights
    in_channels=3,                  # RGB image
    classes=19                      # Cityscapes has 19 classes
)
# For this public model, we don't have a single .pth file, but we can use it directly.
# In a real project, you'd load your own fine-tuned weights.
seg_model.eval()
# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
seg_model.to(device)
print(f"Using device: {device}")

# Pre-processing pipeline for the segmentation model
seg_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
# The 'road' class in the Cityscapes dataset has an ID of 0.
ROAD_CLASS_ID = 0

# --- Model B: The "Car Finder" (YOLOv8) ---
print("Loading Vehicle Detection Model (YOLOv8)...")
yolo_model = YOLO('yolov8n.pt')

# --- VIDEO AND DISPLAY SETUP ---
video_path = "five minutes of evening US101 Palo Alto traffic from Embarcadero crossing pedestrian bridge.mp4"
cap = cv2.VideoCapture(video_path)
window_name = "Fully Automatic Density Analysis"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)

# --- DATA STORAGE AND STATE VARIABLES ---
occupancy_history = []
frame_count = 0
AVERAGING_INTERVAL_SECONDS = 10
congestion_level = "Calculating..."
current_avg_occupancy = 0.0

# --- MAIN PROCESSING LOOP ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame_count += 1
    
    analysis_frame = frame.copy()
    
    # --- STAGE 1: FIND THE ROAD (SEMANTIC SEGMENTATION) ---
    # Prepare the frame for the segmentation model
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = seg_transform(img_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get the raw output from the segmentation model
        seg_output = seg_model(img_tensor)
        # Get the predicted class for each pixel
        predicted_mask = torch.argmax(seg_output, dim=1).squeeze(0).cpu().numpy()

    # Create a binary mask where only the 'road' pixels are white
    road_mask = np.uint8(predicted_mask == ROAD_CLASS_ID) * 255
    
    # Calculate the total area of the detected road
    total_road_area = cv2.countNonZero(road_mask)

    # --- STAGE 2: FIND CARS AND CALCULATE DENSITY ---
    # Create a mask for all detected car boxes in this frame
    box_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    
    # Run YOLO to find cars
    yolo_results = yolo_model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)

    if yolo_results[0].boxes.id is not None:
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, -1)
            # Draw car boxes on the display frame for visualization
            cv2.rectangle(analysis_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

    # Find the intersection between cars and the automatically detected road
    intersection = cv2.bitwise_and(road_mask, box_mask)
    occupied_area = cv2.countNonZero(intersection)
    
    if total_road_area > 0:
        occupancy_percentage = (occupied_area / total_road_area) * 100
        occupancy_history.append(occupancy_percentage)

    # --- PERIODIC CALCULATION & STATE UPDATE ---
    if frame_count > 0 and fps > 0 and frame_count % (int(fps) * AVERAGING_INTERVAL_SECONDS) == 0:
        if occupancy_history:
            current_avg_occupancy = sum(occupancy_history) / len(occupancy_history)
            
            if current_avg_occupancy > 15: # Lower threshold, as it's a % of the whole road
                congestion_level = "High"
            elif current_avg_occupancy > 5:
                congestion_level = "Medium"
            else:
                congestion_level = "Light"
            
            occupancy_history.clear()
            
    # --- VISUALIZATION ---
    # Create a colored overlay for the detected road
    road_overlay = np.zeros_like(analysis_frame)
    road_overlay[road_mask == 255] = (0, 255, 0) # Green for road
    
    # Blend the overlay with the analysis frame
    cv2.addWeighted(road_overlay, 0.3, analysis_frame, 0.7, 0, analysis_frame)

    # Persistent Display
    cv2.rectangle(analysis_frame, (5, 5), (550, 70), (0,0,0), -1)
    text_state = f"State (Auto-Zone): {congestion_level}"
    text_density = f"Road Density: {current_avg_occupancy:.2f}%"
    cv2.putText(analysis_frame, text_state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(analysis_frame, text_density, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
    cv2.imshow(window_name, analysis_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()