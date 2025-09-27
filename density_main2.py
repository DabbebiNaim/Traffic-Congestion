import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# --- SETUP ---
model = YOLO('yolov8n.pt')
video_path = "five minutes of evening US101 Palo Alto traffic from Embarcadero crossing pedestrian bridge.mp4"
cap = cv2.VideoCapture(video_path)

# --- CALIBRATION FOR TWO DENSITY ZONES ---


# --- ZONE 1 ---
ZONE1_POINTS = np.array([
    [656, 335],  # Top-Left
    [1060, 335], # Top-Right
    [1300, 454], # Bottom-Right
    [707, 454]   # Bottom-Left
], dtype=np.int32)

# --- ZONE 2  ---
ZONE2_POINTS = np.array([[118, 345],[480, 345],[458, 415],[4, 408]], dtype=np.int32)
#(118, 345), (480, 345), (4, 408), (458, 415)]

# --- DENSITY SETUP FOR BOTH ZONES ---
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# Create masks to calculate the total pixel area of each zone
zone1_mask = np.zeros((frame_height, frame_width), dtype=np.uint8); cv2.fillPoly(zone1_mask, [ZONE1_POINTS], 255)
zone2_mask = np.zeros((frame_height, frame_width), dtype=np.uint8); cv2.fillPoly(zone2_mask, [ZONE2_POINTS], 255)
total_zone_area_z1 = cv2.countNonZero(zone1_mask)
total_zone_area_z2 = cv2.countNonZero(zone2_mask)

# --- DATA STORAGE ---
# Use dictionaries to manage data for each zone
occupancy_history = {1: [], 2: []}
congestion_levels = {1: "Calculating...", 2: "Calculating..."}
average_occupancies = {1: 0.0, 2: 0.0}
AVERAGING_INTERVAL_SECONDS = 10

# --- INTERACTIVE PLAYER SETUP ---
window_name = "Multi-Zone Density Analysis"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
paused = False; cv2.createTrackbar('Position', window_name, 0, total_frames, lambda pos: cap.set(cv2.CAP_PROP_POS_FRAMES, pos))

# --- PROCESSING LOOP ---
while cap.isOpened():
    current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos('Position', window_name, current_frame_pos)
    if not paused:
        success, frame = cap.read()
        if not success: break
    
    analysis_frame = frame.copy()
    # Draw both measurement zones
    cv2.polylines(analysis_frame, [ZONE1_POINTS], isClosed=True, color=(0, 255, 0), thickness=2) # Green for Zone 1
    cv2.polylines(analysis_frame, [ZONE2_POINTS], isClosed=True, color=(255, 0, 0), thickness=2) # Blue for Zone 2
    
    # Create a single mask for all car boxes in the current frame
    box_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    
    results = model.track(analysis_frame, persist=True, classes=[2, 3, 5, 7])
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Draw all car boxes onto the mask for efficient calculation
        for box in boxes:
            cv2.rectangle(box_mask, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 255, -1)
        
        # Loop again to draw individual labels
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            track_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
            # Determine which zone the car is in for coloring
            is_in_zone1 = cv2.pointPolygonTest(ZONE1_POINTS, track_point, False) >= 0
            is_in_zone2 = cv2.pointPolygonTest(ZONE2_POINTS, track_point, False) >= 0
            
            color = (0, 255, 0) if is_in_zone1 else (255, 0, 0) if is_in_zone2 else (200, 200, 200)
            
            # Draw the box and the simple ID label
            cv2.rectangle(analysis_frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id}"
            cv2.putText(analysis_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # --- DENSITY CALCULATION FOR BOTH ZONES ---
    # This is done once per frame, outside the car loop
    for zone_id, zone_mask, total_area in [(1, zone1_mask, total_zone_area_z1), (2, zone2_mask, total_zone_area_z2)]:
        intersection = cv2.bitwise_and(zone_mask, box_mask)
        occupied_area = cv2.countNonZero(intersection)
        if total_area > 0:
            occupancy_history[zone_id].append((occupied_area / total_area) * 100)

    # --- PERIODIC CALCULATION AND STATE UPDATE ---
    if current_frame_pos > 0 and fps > 0 and current_frame_pos % (int(fps) * AVERAGING_INTERVAL_SECONDS) == 0:
        for zone_id in [1, 2]:
            if occupancy_history[zone_id]:
                average_occupancies[zone_id] = sum(occupancy_history[zone_id]) / len(occupancy_history[zone_id])
                
                # Classification based on density
                if average_occupancies[zone_id] > 30:
                    congestion_levels[zone_id] = "High"
                elif average_occupancies[zone_id] > 10:
                    congestion_levels[zone_id] = "Medium"
                else:
                    congestion_levels[zone_id] = "Light"
                
                occupancy_history[zone_id].clear()
            
    # --- PERSISTENT DISPLAY ---
    cv2.rectangle(analysis_frame, (5, 5), (600, 70), (0,0,0), -1)
    text_z1 = f"Zone 1 (Green) State: {congestion_levels[1]} | Density: {average_occupancies[1]:.2f}%"
    text_z2 = f"Zone 2 (Blue)   State: {congestion_levels[2]} | Density: {average_occupancies[2]:.2f}%"
    cv2.putText(analysis_frame, text_z1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(analysis_frame, text_z2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
    cv2.imshow(window_name, analysis_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == 32: paused = not paused

cap.release()
cv2.destroyAllWindows()