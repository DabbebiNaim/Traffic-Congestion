import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision import transforms

# --- 1. SETUP BOTH DEEP LEARNING MODELS ---
print("Loading Road Segmentation Model (DeepLabV3+)...")
try:
    seg_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
except Exception:
    seg_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True, force_reload=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
seg_model.to(device)
seg_model.eval()
print("Loading Vehicle Detection Model (YOLOv8)...")
yolo_model = YOLO('yolov8n.pt')
yolo_model.to(device)
print(f"Models loaded on device: {device}")

seg_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
BACKGROUND_CLASS_ID = 0

# --- 2. VIDEO AND DISPLAY SETUP ---
video_path = "Dave Day 2025.mp4"
cap = cv2.VideoCapture(video_path)
window_name = "Automatic Density Analysis (with Color Filter)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# --- !!! FIX 1: ROBUST FPS HANDLING !!! ---
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    print("Warning: Could not get FPS from video. Defaulting to 30.")
    fps = 30

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
        predicted_mask_small = torch.argmax(seg_output, dim=1).squeeze(0).cpu().numpy()
    ground_plane_mask_small = np.uint8(predicted_mask_small == BACKGROUND_CLASS_ID)

    hls_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HLS)
    l_channel = hls_frame[:,:,1]; s_channel = hls_frame[:,:,2]
    color_mask = np.zeros_like(l_channel); color_mask[(s_channel < 40) & (l_channel > 50) & (l_channel < 200)] = 1

    final_road_mask_small = cv2.bitwise_and(ground_plane_mask_small, color_mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_road_mask_small, 4, cv2.CV_32S)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        road_mask_small = np.uint8(labels == largest_label)
    else:
        road_mask_small = np.zeros_like(final_road_mask_small, dtype=np.uint8)

    road_mask = cv2.resize(road_mask_small, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
    # This is the area of the VISIBLE road surface (with holes where cars are)
    visible_road_area = np.sum(road_mask)

    # --- STAGE 2: FIND CARS & CALCULATE DENSITY (Corrected Logic) ---
    area_of_cars_on_road = 0
    yolo_results = yolo_model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)

    if yolo_results and yolo_results[0].boxes.id is not None:
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # Heuristic: Check if the bottom-center of the car's box is on the road mask
            bottom_center_x = int((x1 + x2) / 2)
            bottom_center_y = y2
            
            if 0 <= bottom_center_y < frame_height and 0 <= bottom_center_x < frame_width:
                # If the point is on the road mask, we consider the car to be on the road
                if road_mask[bottom_center_y, bottom_center_x] == 1:
                    area_of_cars_on_road += (x2 - x1) * (y2 - y1)
                    cv2.rectangle(analysis_frame, (x1, y1), (x2, y2), (0, 255, 100), 2) # Green for cars on road
                else:
                    cv2.rectangle(analysis_frame, (x1, y1), (x2, y2), (255, 0, 255), 2) # Purple for other vehicles
    
    # New Calculation: Total area is visible road + area occluded by cars
    estimated_total_road_area = visible_road_area + area_of_cars_on_road

    if estimated_total_road_area > 0:
        occupancy_percentage = (area_of_cars_on_road / estimated_total_road_area) * 100
        occupancy_history.append(occupancy_percentage)
    else:
        occupancy_history.append(0)

    # --- !!! FIX 2: CORRECT PERIODIC STATE UPDATE !!! ---
    if frame_count % (int(fps) * AVERAGING_INTERVAL_SECONDS) == 0:
        if occupancy_history:
            current_avg_occupancy = np.mean(occupancy_history)
            if current_avg_occupancy > 15:
                congestion_level = "High"
            elif current_avg_occupancy > 5:
                congestion_level = "Medium"
            else:
                congestion_level = "Light"
            occupancy_history.clear()
        else:
            congestion_level = "No Data"
            current_avg_occupancy = 0.0

    # --- VISUALIZATION (Now always uses the latest state) ---
    road_overlay = np.zeros_like(analysis_frame)
    road_overlay[road_mask == 1] = (0, 255, 0)
    analysis_frame = cv2.addWeighted(analysis_frame, 1.0, road_overlay, 0.3, 0)

    cv2.rectangle(analysis_frame, (5, 5), (550, 70), (0,0,0), -1)
    text_state = f"State (Auto-Zone): {congestion_level}"
    text_density = f"Avg Road Density: {current_avg_occupancy:.2f}%"
    cv2.putText(analysis_frame, text_state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(analysis_frame, text_density, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow(window_name, analysis_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()