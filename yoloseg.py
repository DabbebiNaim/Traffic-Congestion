import cv2
import numpy as np
from ultralytics import YOLO

# --- 1. LOAD YOLO-SEG MODEL ---
print("Loading YOLOv8 segmentation model...")
model = YOLO('yolov8n-seg.pt')  # Using the nano model for speed
print("Model loaded successfully.")

# --- 2. VIDEO SETUP ---
# IMPORTANT: Change this to your video file path or 0 for webcam
video_path = "4K Video of Highway Traffic!.mp4"

try:
    video_source = int(video_path)
except ValueError:
    video_source = video_path

cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print(f"Error: Could not open video source: {video_path}")
    exit()

window_name = "YOLO Road Segmentation"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# --- 3. MAIN VIDEO LOOP ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("End of video.")
        break

    # --- PERFORM OBJECT SEGMENTATION ---
    # We only care about classes that are likely to be on the road
    # COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
    results = model(frame, classes=[2, 3, 5, 7], verbose=False)

    # --- STAGE 1: CREATE A MASK OF EVERYTHING THAT IS *NOT* THE ROAD ---
    h, w, _ = frame.shape
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    # Check if YOLO found any masks
    if results[0].masks is not None:
        # Iterate through each detected object's mask
        for mask_tensor in results[0].masks.data:
            # Convert tensor to numpy array
            object_mask = mask_tensor.cpu().numpy().astype(np.uint8)
            # Resize mask to match frame dimensions
            object_mask_resized = cv2.resize(object_mask, (w, h))
            # Add the object's mask to our combined mask
            combined_mask = cv2.bitwise_or(combined_mask, object_mask_resized)

    # --- STAGE 2: INVERT THE MASK TO GET THE "POTENTIAL ROAD" ---
    # Everything that was not an object is now white (255)
    potential_road_mask = cv2.bitwise_not(combined_mask)

    # --- STAGE 3: REFINE THE MASK ---
    # 1. Remove the top portion of the frame (sky, buildings, etc.)
    # You can adjust this cutoff percentage
    sky_cutoff = int(h * 0.40)
    potential_road_mask[:sky_cutoff, :] = 0
    
    # 2. Keep only the largest continuous area (contour)
    # This removes small, disconnected noisy areas
    contours, _ = cv2.findContours(potential_road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_road_mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Check if the largest contour is reasonably large
        if cv2.contourArea(largest_contour) > 2000: # Min area threshold
            cv2.drawContours(final_road_mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    # --- VISUALIZATION ---
    # Create a green overlay for the road
    road_overlay = np.zeros_like(frame)
    road_overlay[final_road_mask == 255] = (0, 255, 0) # Green

    # Blend the original frame with the road overlay
    display_frame = cv2.addWeighted(frame, 1, road_overlay, 0.4, 0)
    
    cv2.imshow(window_name, display_frame)
    # Optionally, show the final mask itself
    cv2.imshow("Final Road Mask", final_road_mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. CLEANUP ---
cap.release()
cv2.destroyAllWindows()