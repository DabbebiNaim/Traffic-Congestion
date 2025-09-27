import cv2
import numpy as np
import onnxruntime as ort

# --- 1. SETUP ---
onnx_model_path = "BiseNet.onnx"
video_path = "4K Video of Highway Traffic!.mp4"

# --- 1a. PARAMETERS & CONSTANTS ---
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
# We are guessing the road is ID 0, but we will find the true ID.
ROAD_CLASS_ID = 0
ROAD_OVERLAY_COLOR = [128, 64, 128] # Purple

# The full colormap lets us see what the model is detecting for all classes
cityscapes_colormap = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]], dtype=np.uint8)

# --- 2. MODEL LOADING ---
print("Loading BiSeNet ONNX model...")
try:
    sess = ort.InferenceSession(onnx_model_path)
    input_name = sess.get_inputs()[0].name
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit()

INPUT_WIDTH = 960
INPUT_HEIGHT = 720
frame_count = 0

# --- 3. MAIN VIDEO LOOP ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("End of video stream.")
        break
    
    original_h, original_w, _ = frame.shape
    
    # --- 4. PREPARE FRAME & RUN INFERENCE ---
    resized_frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    scaled_frame = rgb_frame / 255.0
    normalized_frame = (scaled_frame - MEAN) / STD
    blob = np.expand_dims(normalized_frame.transpose(2, 0, 1), axis=0).astype(np.float32)
    output = sess.run(None, {input_name: blob})[0]
    class_map = np.argmax(output[0], axis=0)

    # --- 5. DEBUG: PRINT UNIQUE CLASS IDs TO TERMINAL ---
    frame_count += 1
    if frame_count % 60 == 0:  # Print every ~2 seconds
        unique_classes = np.unique(class_map)
        print(f"DEBUG: Class IDs found in this frame -> {unique_classes}")

    # --- 6. CREATE VISUALIZATIONS ---
    # A. Full Segmentation (to see what the model detects)
    full_color_mask = cityscapes_colormap[class_map].astype(np.uint8)
    resized_full_mask = cv2.resize(full_color_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    full_overlay_view = cv2.addWeighted(frame, 0.7, resized_full_mask, 0.3, 0)

    # B. Road-Only Segmentation (based on our guess)
    road_mask = (class_map == ROAD_CLASS_ID)
    resized_road_mask = cv2.resize(road_mask.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    road_overlay_frame = frame.copy()
    road_overlay_frame[resized_road_mask == 1] = ROAD_OVERLAY_COLOR
    road_only_view = cv2.addWeighted(frame, 0.7, road_overlay_frame, 0.3, 0)

    # --- 7. COMBINE AND DISPLAY ---
    # Put text labels on each view
    cv2.putText(full_overlay_view, "Full Segmentation", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(road_only_view, f"Road-Only (ID: {ROAD_CLASS_ID})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Stack the two views side-by-side
    combined_display = np.hstack((full_overlay_view, road_only_view))
    cv2.imshow("Debug View: Full (Left) vs. Road-Only (Right)", combined_display)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# --- 8. CLEANUP ---
cap.release()
cv2.destroyAllWindows()