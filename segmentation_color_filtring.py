import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# --- CONSTANTS ---
VIDEO_PATH = "Dave Day 2025.mp4"
# Number of frames to sample at the start to learn the background
CALIBRATION_FRAMES = 150 
AVERAGING_INTERVAL_SECONDS = 10

# --- 1. LEARNING PHASE FUNCTION ---
def learn_road_mask(video_path, num_frames_to_sample):
    """
    Analyzes the start of a video to automatically generate a road mask.
    """
    print(f"Starting Phase 1: Learning road area from {num_frames_to_sample} frames...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video for learning.")
        return None

    # Collect a buffer of frames
    frame_buffer = []
    for i in range(num_frames_to_sample):
        success, frame = cap.read()
        if not success:
            print(f"Warning: Video ended before sampling {num_frames_to_sample} frames.")
            break
        frame_buffer.append(frame)
    
    cap.release()
    
    if not frame_buffer:
        print("Error: No frames were collected.")
        return None

    # Calculate the median frame across the buffer to get a clean background
    print("Calculating median frame to remove moving objects...")
    background_model = np.median(frame_buffer, axis=0).astype(np.uint8)

    # Now, create a mask from this clean background image
    print("Creating road mask from clean background...")
    hls_background = cv2.cvtColor(background_model, cv2.COLOR_BGR2HLS)
    l_channel = hls_background[:,:,1]
    s_channel = hls_background[:,:,2]
    
    # Use color filtering on the clean image (much more reliable)
    road_mask = np.zeros_like(l_channel, dtype=np.uint8)
    # Tune these thresholds once for your camera type if needed
    road_mask[(s_channel < 50) & (l_channel > 40)] = 255
    
    # Optional: Clean up small noise from the mask
    road_mask = cv2.medianBlur(road_mask, 5)

    print("--- Phase 1: Road Mask Learning Complete ---")
    return road_mask

# --- MAIN SCRIPT ---

# --- Phase 1: Learn the Road Automatically ---
road_mask = learn_road_mask(VIDEO_PATH, CALIBRATION_FRAMES)
if road_mask is None:
    print("Could not generate road mask. Exiting.")
    exit()

total_road_area = cv2.countNonZero(road_mask)
if total_road_area == 0:
    print("Error: Learned road mask is empty. Check color filter thresholds. Exiting.")
    exit()

# --- Phase 2: Setup for Real-time Analysis ---
print("\nStarting Phase 2: Real-time Density Analysis...")
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(VIDEO_PATH)
window_name = "Fully Automatic Density Detector"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30

# Data storage and state variables
occupancy_history = deque(maxlen=int(fps * AVERAGING_INTERVAL_SECONDS))
congestion_level = "Calculating..."
current_avg_occupancy = 0.0
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame_count += 1
    
    analysis_frame = frame.copy()
    frame_height, frame_width, _ = frame.shape
    
    # Create a mask for all cars detected in the current frame
    box_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    yolo_results = model(frame, classes=[2, 3, 5, 7], verbose=False)

    if yolo_results and yolo_results[0].boxes.shape[0] > 0:
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, -1)
            cv2.rectangle(analysis_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
    
    # Calculate occupancy for the current frame
    intersection = cv2.bitwise_and(road_mask, box_mask)
    occupied_area = cv2.countNonZero(intersection)
    occupancy_percentage = (occupied_area / total_road_area) * 100
    occupancy_history.append(occupancy_percentage)

    # --- Periodic State Update ---
    # We use a moving average from the deque for a smoother, continuous update
    if occupancy_history:
        current_avg_occupancy = np.mean(occupancy_history)
        if current_avg_occupancy > 15:
            congestion_level = "High"
        elif current_avg_occupancy > 5:
            congestion_level = "Medium"
        else:
            congestion_level = "Light"

    # --- Visualization ---
    road_overlay = np.zeros_like(analysis_frame)
    road_overlay[road_mask == 255] = (0, 255, 0)
    analysis_frame = cv2.addWeighted(analysis_frame, 1.0, road_overlay, 0.3, 0)

    cv2.rectangle(analysis_frame, (5, 5), (550, 70), (0,0,0), -1)
    text_state = f"State (Auto-Zone): {congestion_level}"
    text_density = f"Road Density: {current_avg_occupancy:.2f}%"
    cv2.putText(analysis_frame, text_state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(analysis_frame, text_density, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
    cv2.imshow(window_name, analysis_frame)
    # For debugging, you can show the learned mask
    # cv2.imshow("Learned Road Mask", road_mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()