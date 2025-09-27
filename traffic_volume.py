import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# --- SETUP ---
model = YOLO('yolov8n.pt')

# Video Setup
video_path = "4K Video of Highway Traffic!.mp4"
cap = cv2.VideoCapture(video_path)
window_name = "Advanced Traffic State Analysis"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Get video properties
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)

# --- VIRTUAL GATE AND ANALYSIS SETUP ---
GATE_POSITION_PERCENT = 0.6 # Position of the gate, 80% from the top
GATE_Y_POSITION = int(frame_height * GATE_POSITION_PERCENT)

# Data storage
track_history = defaultdict(list)
already_crossed_ids = set()

# Variables for periodic analysis
frame_count = 0
INTERVAL_SECONDS = 5
going_car_count = 0
coming_car_count = 0
going_speeds = []
coming_speeds = []

# <<< NEW, SIMPLIFIED THRESHOLDS BASED ON SPEED (pixels/sec) >>>
# These are the ONLY thresholds you need to tune for state classification.
# Speed below this value is considered "High" (Congestion).
SPEED_THRESHOLD_HIGH = 75
# Speed between HIGH and LIGHT is "Medium".
SPEED_THRESHOLD_LIGHT = 150
# Speed above this value is "Light" (Free-Flow).

# State variables for persistent display
going_vph, coming_vph = 0, 0
going_avg_speed, coming_avg_speed = 0, 0
going_state, coming_state = "Calculating...", "Calculating..."


# --- HELPER FUNCTIONS ---
def line_intersects(p1, p2, y_gate):
    return (p1[1] < y_gate and p2[1] >= y_gate) or (p2[1] < y_gate and p1[1] >= y_gate)

def calculate_speed(p1_data, p2_data, fps):
    p1, frame1 = p1_data
    p2, frame2 = p2_data
    distance_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    time_seconds = abs(frame2 - frame1) / fps
    return distance_pixels / time_seconds if time_seconds > 0 else 0


# --- MAIN VIDEO LOOP ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame_count += 1
    analysis_frame = frame.copy()
    cv2.line(analysis_frame, (0, GATE_Y_POSITION), (frame_width, GATE_Y_POSITION), (0, 255, 255), 2)

    results = model.track(frame, persist=True, classes=[2, 3, 5, 7])

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            history = track_history[track_id]
            history.append(((int(x), int(y)), frame_count))
            if len(history) > 2:
                history.pop(0)

            if len(history) == 2 and track_id not in already_crossed_ids:
                p_then_data, p_now_data = history
                if line_intersects(p_then_data[0], p_now_data[0], GATE_Y_POSITION):
                    already_crossed_ids.add(track_id)
                    speed = calculate_speed(p_then_data, p_now_data, fps)
                    if p_now_data[0][1] > p_then_data[0][1]:
                        coming_car_count += 1
                        coming_speeds.append(speed)
                    else:
                        going_car_count += 1
                        going_speeds.append(speed)
            cv2.rectangle(analysis_frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 255, 0), 2)

    # --- PERIODIC ANALYSIS AND STATE UPDATE ---
    if fps > 0 and frame_count % (int(fps) * INTERVAL_SECONDS) == 0 and frame_count > 0:
        going_vph = (going_car_count / INTERVAL_SECONDS) * 3600
        coming_vph = (coming_car_count / INTERVAL_SECONDS) * 3600
        going_avg_speed = np.mean(going_speeds) if going_speeds else 0
        coming_avg_speed = np.mean(coming_speeds) if coming_speeds else 0

        # <<< REVISED LOGIC: Classify state based purely on average speed >>>
        # -- Going Direction --
        if going_avg_speed < SPEED_THRESHOLD_HIGH:
            going_state = "High"  # (Congestion)
        elif going_avg_speed < SPEED_THRESHOLD_LIGHT:
            going_state = "Medium"
        else:
            going_state = "Light" # (Free-Flow)
        
        # -- Coming Direction --
        if coming_avg_speed < SPEED_THRESHOLD_HIGH:
            coming_state = "High"  # (Congestion)
        elif coming_avg_speed < SPEED_THRESHOLD_LIGHT:
            coming_state = "Medium"
        else:
            coming_state = "Light" # (Free-Flow)

        # Reset for next interval
        going_car_count, coming_car_count = 0, 0
        going_speeds.clear(), coming_speeds.clear()
        already_crossed_ids.clear()

    # --- PERSISTENT DISPLAY ---
    cv2.rectangle(analysis_frame, (5, 5), (600, 70), (0,0,0), -1)
    text_going = f"Going: {going_vph:.0f} VPH | Avg Speed: {going_avg_speed:.0f} px/s | State: {going_state}"
    text_coming = f"Coming: {coming_vph:.0f} VPH | Avg Speed: {coming_avg_speed:.0f} px/s | State: {coming_state}"
    cv2.putText(analysis_frame, text_going, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(analysis_frame, text_coming, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow(window_name, analysis_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()