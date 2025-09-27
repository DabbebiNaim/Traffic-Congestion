import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# --- SETUP ---
model = YOLO('yolov8n.pt')
video_path = "congestion5.mp4"  # Use your long video file
cap = cv2.VideoCapture(video_path)

# --- YOUR CALIBRATION DATA ---
TOP_LEFT     = [526, 749]
TOP_RIGHT    = [1112, 745]
BOTTOM_LEFT  = [526, 803]
BOTTOM_RIGHT = [1118, 798]
DRAWING_POINTS = np.array([TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT], dtype=np.int32)
SOURCE_POINTS = np.float32([TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT])
RECT_WIDTH_METERS = 3.6
RECT_LENGTH_METERS = 3.6
DESTINATION_POINTS = np.float32([[0, 0], [RECT_WIDTH_METERS, 0], [0, RECT_LENGTH_METERS], [RECT_WIDTH_METERS, RECT_LENGTH_METERS]])
transformation_matrix = cv2.getPerspectiveTransform(SOURCE_POINTS, DESTINATION_POINTS)
MEASUREMENT_ZONE = DRAWING_POINTS

# --- DATA STORAGE ---
track_history = defaultdict(list)
vehicle_speeds = {}
vehicle_directions = {}
speeds_going = []
speeds_coming = []
going_congestion_level = "Calculating..."
coming_congestion_level = "Calculating..."
AVERAGING_INTERVAL_SECONDS = 5

# --- NEW: INTERACTIVE PLAYER SETUP ---
window_name = "Interactive Traffic Analysis"
cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
paused = False

def on_trackbar_change(pos):
    """Callback function to seek to the specified frame."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

# Create the trackbar
cv2.createTrackbar('Position', window_name, 0, total_frames, on_trackbar_change)


# --- PROCESSING LOOP ---
while cap.isOpened():
    # Get the current frame position to update the trackbar
    current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos('Position', window_name, current_frame_pos)

    # Only read a new frame if the video is not paused
    if not paused:
        success, frame = cap.read()
        if not success:
            print("End of video or cannot read frame. Exiting.")
            break
    
    # --- YOUR ANALYSIS LOGIC (runs on the current frame, paused or not) ---
    analysis_frame = frame.copy() # Work on a copy to preserve the original frame for display when paused
    cv2.polylines(analysis_frame, [DRAWING_POINTS], isClosed=True, color=(0, 255, 0), thickness=2)

    results = model.track(analysis_frame, persist=True, classes=[2, 3, 5, 7])
    
    if results[0].boxes.id is not None:
        # (The rest of your analysis logic for boxes, speed, direction is the same)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            track_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            history = track_history[track_id]
            history.append(track_point)
            if len(history) > 30: history.pop(0)

            direction = "Unknown"
            if len(history) > 10:
                vertical_movement = history[-1][1] - history[0][1]
                if vertical_movement < -5: direction = "Going"
                elif vertical_movement > 5: direction = "Coming"
            vehicle_directions[track_id] = direction

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

            is_inside_zone = cv2.pointPolygonTest(MEASUREMENT_ZONE, track_point, False) >= 0
            if is_inside_zone and track_id in vehicle_speeds:
                if direction == "Going": speeds_going.append(vehicle_speeds[track_id])
                elif direction == "Coming": speeds_coming.append(vehicle_speeds[track_id])

            color = (255, 255, 255) if direction == "Going" else (0, 0, 255) if direction == "Coming" else (128, 128, 128)
            cv2.rectangle(analysis_frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id} {direction}"
            if track_id in vehicle_speeds: label += f" {vehicle_speeds[track_id]:.1f}km/h"
            cv2.putText(analysis_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # --- PERIODIC CALCULATION ---
    if current_frame_pos > 0 and current_frame_pos % (int(fps) * AVERAGING_INTERVAL_SECONDS) == 0:
        if speeds_going:
            avg_speed_going = sum(speeds_going) / len(speeds_going)
            going_congestion_level = "High" if avg_speed_going < 20 else ("Medium" if avg_speed_going < 50 else "Light")
            speeds_going.clear()
        if speeds_coming:
            avg_speed_coming = sum(speeds_coming) / len(speeds_coming)
            coming_congestion_level = "High" if avg_speed_coming < 20 else ("Medium" if avg_speed_coming < 50 else "Light")
            speeds_coming.clear()
            
    # --- PERSISTENT DISPLAY ---
    cv2.rectangle(analysis_frame, (5, 5), (450, 70), (0,0,0), -1)
    text_going = f"the state of the going is : {going_congestion_level} , {sum(speeds_going) / (len(speeds_going)+0.0000001)}"
    text_coming = f"the state of the coming is : {coming_congestion_level}, {sum(speeds_coming) / (len(speeds_coming)+0.0000001)}"
    cv2.putText(analysis_frame, text_going, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(analysis_frame, text_coming, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
    # Display the final, annotated frame
    cv2.imshow(window_name, analysis_frame)

    # --- NEW: KEYBOARD CONTROLS ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32: # SPACEBAR
        paused = not paused
        if paused:
            print("--- Paused ---")
        else:
            print("--- Resumed ---")

cap.release()
cv2.destroyAllWindows()