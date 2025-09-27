import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# --- SETUP ---
model = YOLO('yolov8n.pt')
# --- UPDATED VIDEO PATH ---
video_path = "five minutes of evening US101 Palo Alto traffic from Embarcadero crossing pedestrian bridge.mp4"
cap = cv2.VideoCapture(video_path)
window_name = "Auto-Calibrated Speed Detector"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# --- CONSTANTS AND CONFIGURATION ---
AVG_CAR_WIDTH_METERS = 1.8  # Our reference assumption for a car's width
CALIBRATION_FRAME_NUM = 150 # A frame number deep enough into the video for stable traffic

# --- !!! CALIBRATION FACTOR FOR FINE-TUNING !!! ---
# A multiplier to adjust the final speed calculation.
# If speeds are consistently too low, INCREASE this value (e.g., to 1.5, 1.8).
# If speeds are too high, DECREASE this value (e.g., to 0.8, 0.7).
# Start with 1.8 as a good guess to boost the previous 50km/h result.
CALIBRATION_FACTOR = 1.8

# --- DATA STORAGE ---
track_history = defaultdict(list)
vehicle_speeds = {}
pixels_per_meter = None
avg_speed_kph = 0.0
congestion_level = "Calibrating..."

# --- ROBUST AUTO-CALIBRATION FUNCTION ---
def auto_calibrate_scale(video_capture, frame_number, avg_car_width):
    """
    Analyzes a specific frame to find a reliable pixels-per-meter ratio
    using the median width of several candidate cars.
    """
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = video_capture.read()
    if not success:
        print("Error: Could not read calibration frame.")
        return None

    print(f"Running detection on calibration frame #{frame_number}...")
    results = model(frame, classes=[2, 3, 5, 7], verbose=False)
    
    candidate_widths = []
    for box in results[0].boxes.xywh:
        x, y, w, h = box.cpu().numpy()
        
        # Filters: Consider cars in the bottom half of the screen that have a reasonable width
        if y > frame.shape[0] / 2 and 50 < w < 250:
            candidate_widths.append(w)
    
    if len(candidate_widths) < 3:
        print(f"Calibration failed: Not enough suitable cars found. Found only {len(candidate_widths)}.")
        return None

    # Use the median width, which is robust to outliers
    median_car_width_pixels = np.median(candidate_widths)
    ppm = median_car_width_pixels / avg_car_width
    
    print(f"--- Auto-Calibration Successful ---")
    print(f"Found {len(candidate_widths)} candidate cars. Using median width.")
    print(f"Median car pixel width: {median_car_width_pixels:.2f}")
    print(f"Calculated scale: {ppm:.2f} pixels per meter")
    
    # Reset video to the beginning for processing
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return ppm

# --- Run Auto-Calibration ---
pixels_per_meter = auto_calibrate_scale(cap, CALIBRATION_FRAME_NUM, AVG_CAR_WIDTH_METERS)
if pixels_per_meter is None:
    print("Exiting due to calibration failure. Try adjusting CALIBRATION_FRAME_NUM or width filters.")
    exit()

# --- MAIN PROCESSING LOOP ---
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
speed_history = []
AVERAGING_INTERVAL_SECONDS = 5

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame_count += 1
    analysis_frame = frame.copy()

    results = model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track_point = (int(x), int(y))

            history = track_history[track_id]
            history.append(track_point)
            if len(history) > 30: history.pop(0)

            if len(history) > 10:
                point_now, point_then = history[-1], history[-10]
                pixel_distance = np.linalg.norm(np.array(point_now) - np.array(point_then))
                meter_distance = pixel_distance / pixels_per_meter
                
                time_seconds = 10 / fps
                if time_seconds > 0:
                    # Apply the calibration factor to the final speed
                    speed_kmh = (meter_distance / time_seconds) * 3.6 * CALIBRATION_FACTOR
                    vehicle_speeds[track_id] = speed_kmh
                    # Only add speeds of cars in the analysis region to the average
                    if y > frame.shape[0] / 2:
                        speed_history.append(speed_kmh)

            # --- Visualization ---
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            cv2.rectangle(analysis_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            label = "" # Initialize label as empty
            if track_id in vehicle_speeds:
                label += f"{vehicle_speeds[track_id]:.1f}km/h" # Only add speed
            if label: # Only display if there's actual text
                cv2.putText(analysis_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # --- PERIODIC CONGESTION CLASSIFICATION ---
    if frame_count > 0 and fps > 0 and frame_count % (int(fps) * AVERAGING_INTERVAL_SECONDS) == 0:
        if speed_history:
            # Filter out extreme outliers before averaging for a more stable result
            valid_speeds = [s for s in speed_history if 0 < s < 150]
            if valid_speeds:
                avg_speed_kph = np.mean(valid_speeds)
            
                if avg_speed_kph < 30:
                    congestion_level = "High"
                elif avg_speed_kph < 70:
                    congestion_level = "Medium"
                else:
                    congestion_level = "Light"
            
            speed_history.clear()

    # --- PERSISTENT DISPLAY ---
    cv2.rectangle(analysis_frame, (5, 5), (600, 45), (0,0,0), -1)
    display_text = f"Congestion State: {congestion_level} (Avg Speed: {avg_speed_kph:.2f} km/h)"
    cv2.putText(analysis_frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow(window_name, analysis_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()