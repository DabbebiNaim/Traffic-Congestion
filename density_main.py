import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# --- SETUP ---
model = YOLO('yolov8n.pt')
video_path = "congestion5.mp4"
cap = cv2.VideoCapture(video_path)

# --- YOUR CALIBRATION DATA ---
# Note: For density, the real-world dimensions are not used for the final classification,
# but they are still needed for the individual car speed display.

TOP_LEFT     = [321, 724]
TOP_RIGHT    = [646, 718]
BOTTOM_LEFT  = [312, 1000]
BOTTOM_RIGHT = [644, 1000]
DRAWING_POINTS = np.array([TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT], dtype=np.int32)
SOURCE_POINTS = np.float32([TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT])
RECT_WIDTH_METERS = 3.6
RECT_LENGTH_METERS = 3.6
DESTINATION_POINTS = np.float32([[0, 0], [RECT_WIDTH_METERS, 0], [0, RECT_LENGTH_METERS], [RECT_WIDTH_METERS, RECT_LENGTH_METERS]])
transformation_matrix = cv2.getPerspectiveTransform(SOURCE_POINTS, DESTINATION_POINTS)
MEASUREMENT_ZONE = DRAWING_POINTS

# --- SETUP FOR DENSITY CALCULATION ---
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
zone_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
cv2.fillPoly(zone_mask, [MEASUREMENT_ZONE], 255)
total_zone_area = cv2.countNonZero(zone_mask)

# --- DATA STORAGE ---
track_history = defaultdict(list)
vehicle_speeds = {}
occupancy_history = []
# State Variables
congestion_level = "Calculating..."
current_avg_occupancy = 0.0
AVERAGING_INTERVAL_SECONDS = 4

# --- INTERACTIVE PLAYER SETUP ---
window_name = "Density-Based Analysis with Speed Display"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
paused = False

def on_trackbar_change(pos):
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

cv2.createTrackbar('Position', window_name, 0, total_frames, on_trackbar_change)

# --- PROCESSING LOOP ---
while cap.isOpened():
    current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos('Position', window_name, current_frame_pos)

    if not paused:
        success, frame = cap.read()
        if not success:
            break
    
    analysis_frame = frame.copy()
    cv2.polylines(analysis_frame, [MEASUREMENT_ZONE], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # Create mask for density calculation in this frame
    box_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    results = model.track(analysis_frame, persist=True, classes=[2, 3, 5, 7])
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # First, draw all boxes on the mask for the overall density calculation
        for box in boxes:
            x1_b, y1_b, x2_b, y2_b = map(int, box)
            cv2.rectangle(box_mask, (x1_b, y1_b), (x2_b, y2_b), 255, -1)
        
        # Now, loop again for individual car analysis and visualization
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            track_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            history = track_history[track_id]
            history.append(track_point)
            if len(history) > 30: history.pop(0)

            # --- Calculate Speed and Direction (for display only) ---
            direction = "Unknown"
            if len(history) > 10:
                vertical_movement = history[-1][1] - history[0][1]
                if vertical_movement < -5: direction = "Going"
                elif vertical_movement > 5: direction = "Coming"

            if len(history) > 10:
                point_now, point_then = history[-1], history[-10]
                point_then_np = np.array([[point_then]], dtype="float32")
                point_now_np = np.array([[point_now]], dtype="float32")
                real_world_point_then = cv2.perspectiveTransform(point_then_np, transformation_matrix)
                real_world_point_now = cv2.perspectiveTransform(point_now_np, transformation_matrix)
                if real_world_point_now is not None and real_world_point_then is not None:
                    distance_meters = np.linalg.norm(real_world_point_now - real_world_point_then)
                    time_seconds = 10 / fps
                    if time_seconds > 0:
                        speed_kmh = (distance_meters / time_seconds) * 3.6
                        vehicle_speeds[track_id] = speed_kmh
            
            # --- Individual car visualization ---
            color = (255, 255, 255) if direction == "Going" else (0, 0, 255) if direction == "Coming" else (128, 128, 128)
            cv2.rectangle(analysis_frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id} {direction}"
            if track_id in vehicle_speeds:
                label += f" {vehicle_speeds[track_id]:.1f}km/h"
            cv2.putText(analysis_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Calculate and store current frame's occupancy percentage
    intersection = cv2.bitwise_and(zone_mask, box_mask)
    occupied_area = cv2.countNonZero(intersection)
    if total_zone_area > 0:
        occupancy_percentage = (occupied_area / total_zone_area) * 100
        occupancy_history.append(occupancy_percentage)

    # --- PERIODIC CALCULATION (DENSITY ONLY) ---
    if current_frame_pos > 0 and current_frame_pos % (int(fps) * AVERAGING_INTERVAL_SECONDS) == 0:
        if occupancy_history:
            current_avg_occupancy = sum(occupancy_history) / len(occupancy_history)
            
            # Classification based ONLY on density
            if current_avg_occupancy > 30:
                congestion_level = "High"
            elif current_avg_occupancy > 10:
                congestion_level = "Medium"
            else:
                congestion_level = "Light"
            
            occupancy_history.clear()
            
    # --- PERSISTENT DISPLAY ---
    cv2.rectangle(analysis_frame, (5, 5), (550, 70), (0,0,0), -1)
    text_state = f"the state of the traffic is : {congestion_level}"
    text_density = f"current average density : {current_avg_occupancy:.2f}%"
    cv2.putText(analysis_frame, text_state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(analysis_frame, text_density, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
    cv2.imshow(window_name, analysis_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:
        paused = not paused
        print("--- Paused ---" if paused else "--- Resumed ---")

cap.release()
cv2.destroyAllWindows()