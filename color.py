import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision import transforms

# --- 1. SETUP MODELS ---
print("Loading models...")
try:
    seg_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', weights='DeepLabV3_ResNet101_Weights.DEFAULT')
except Exception:
    seg_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True, force_reload=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
seg_model.to(device); seg_model.eval()
yolo_model = YOLO('yolov8n.pt');
print(f"Models loaded on device: {device}")

seg_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
           ])
BACKGROUND_CLASS_ID = 0 

# --- 2. VIDEO SETUP ---
video_path = "Dave Day 2025.mp4"
cap = cv2.VideoCapture(video_path)
window_name = "Full Road Segmentation"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30

# --- 3. DATA STORAGE AND STATE ---
occupancy_history = []
frame_count = 0
AVERAGING_INTERVAL_SECONDS = 10
congestion_level = "Calculating..."
current_avg_occupancy = 0.0

# --- 4. MAIN LOOP ---
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame_count += 1
    
    analysis_frame = frame.copy()
    
    # --- STAGE 1: ROAD MASK REFINEMENT ---
    input_size = (513, 513)
    frame_resized = cv2.resize(frame, input_size)
    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    img_tensor = seg_transform(img_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        seg_output = seg_model(img_tensor)['out']
        predicted_mask_small = torch.argmax(seg_output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    
    # This is our broad guess for the ground plane
    ground_plane_mask = np.uint8(predicted_mask_small == BACKGROUND_CLASS_ID)

    # --- NEW: "UNION OF ALL ROAD PARTS" LOGIC ---
    # Create an empty mask to build our final road surface
    final_road_mask_small = np.zeros_like(ground_plane_mask, dtype=np.uint8)
    
    # Find all separate blobs in the ground plane mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ground_plane_mask, 4, cv2.CV_32S)
    
    if num_labels > 1:
        # Loop through each detected component (blob)
        for label_id in range(1, num_labels):
            # Get the stats for the current component
            area = stats[label_id, cv2.CC_STAT_AREA]
            y = stats[label_id, cv2.CC_STAT_TOP]
            h = stats[label_id, cv2.CC_STAT_HEIGHT]
            
            # --- Geometric Filter ---
            # Keep any component that is reasonably large AND touches the bottom half of the image
            if area > 1000 and (y + h) > (input_size[1] * 0.5):
                # If it passes the filter, add this component to our final mask
                final_road_mask_small[labels == label_id] = 255

    # The final small mask now contains all valid road parts
    road_mask = cv2.resize(final_road_mask_small, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
    total_road_area = cv2.countNonZero(road_mask)

    # --- STAGE 2: FIND CARS & CALCULATE DENSITY ---
    box_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    yolo_results = yolo_model(frame, classes=[2, 3, 5, 7], verbose=False)

    if yolo_results and yolo_results[0].boxes.shape[0] > 0:
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, -1)
            cv2.rectangle(analysis_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

    # --- STAGE 3: CALCULATE OCCUPANCY ---
    intersection = cv2.bitwise_and(road_mask, box_mask)
    occupied_area = cv2.countNonZero(intersection)
    
    if total_road_area > 0:
        occupancy_percentage = (occupied_area / total_road_area) * 100
        occupancy_history.append(occupancy_percentage)
    else:
        occupancy_history.append(0)

    # --- PERIODIC STATE UPDATE ---
    if frame_count % (int(fps) * AVERAGING_INTERVAL_SECONDS) == 0:
        if occupancy_history:
            current_avg_occupancy = np.mean(occupancy_history)
            if current_avg_occupancy > 15: congestion_level = "High"
            elif current_avg_occupancy > 5: congestion_level = "Medium"
            else: congestion_level = "Light"
            occupancy_history.clear()
        else:
            congestion_level = "No Data"; current_avg_occupancy = 0.0
            
    # --- VISUALIZATION ---
    road_overlay = np.zeros_like(analysis_frame)
    road_overlay[road_mask == 255] = (0, 255, 0)
    analysis_frame = cv2.addWeighted(analysis_frame, 1.0, road_overlay, 0.3, 0)

    cv2.rectangle(analysis_frame, (5, 5), (550, 70), (0,0,0), -1)
    text_state = f"State (Auto-Zone): {congestion_level}"
    text_density = f"Road Density: {current_avg_occupancy:.2f}%"
    cv2.putText(analysis_frame, text_state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(analysis_frame, text_density, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
    cv2.imshow(window_name, analysis_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()