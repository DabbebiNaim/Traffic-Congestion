import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import warnings

warnings.filterwarnings("ignore", message=".*VisibleDeprecationWarning.*")

# --- 1. SETUP ---
print("Loading SAM model...")
model_type = "vit_b"
sam_checkpoint = "sam_vit_b.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
print("SAM model loaded.")

# --- 2. REVISED STRATEGIC PROMPT GENERATION FUNCTION ---
def get_strategic_lane_prompts(frame):
    """
    This function now uses color thresholding to isolate white lanes
    before finding lines and generating prompts for SAM.
    """
    # --- Step A: Isolate White Pixels using HLS Color Space ---
    # Convert the frame to HLS (Hue, Lightness, Saturation)
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    # Define the threshold for white/yellowish lanes.
    # The L (Lightness) channel is the most important one here.
    # Adjust the middle value (Lightness) to tune sensitivity.
    # Format is [Hue, Lightness, Saturation]
    lower_white = np.array([0, 200, 0])   # Lower bound for "white"
    upper_white = np.array([255, 255, 255]) # Upper bound for "white"
    
    # Create a mask that keeps only the pixels within the defined white range
    white_mask = cv2.inRange(hls, lower_white, upper_white)

    # --- Step B: Region of Interest (ROI) ---
    height, width = white_mask.shape
    mask = np.zeros_like(white_mask)
    # Adjust these vertices to fit your video's perspective
    roi_vertices = np.array([[(0, height), (width * 0.45, height * 0.6), (width * 0.55, height * 0.6), (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    
    # Apply the ROI to our color mask
    masked_image = cv2.bitwise_and(white_mask, mask)

    # --- Step C: Hough Line Transform on the Clean Mask ---
    lines = cv2.HoughLinesP(
        masked_image, 
        rho=2, 
        theta=np.pi/180, 
        threshold=40,  # Lowered threshold as the input is cleaner
        minLineLength=20, 
        maxLineGap=25
    )

    # --- Step D: Filter, Classify, and Average Lines (Same as before) ---
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if x2 - x1 == 0: continue
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < -0.5:
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                right_fit.append((slope, intercept))
    
    left_fit_average = np.average(left_fit, axis=0) if left_fit else None
    right_fit_average = np.average(right_fit, axis=0) if right_fit else None

    # --- Step E: Generate 4 Strategic Points (Same as before) ---
    def make_points(image, line_parameters):
        if line_parameters is None: return None
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1 * 0.65)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([[x1, y1], [x2, y2]])

    left_points = make_points(frame, left_fit_average)
    right_points = make_points(frame, right_fit_average)

    input_points = []
    if left_points is not None: input_points.extend(left_points.tolist())
    if right_points is not None: input_points.extend(right_points.tolist())
        
    if not input_points: return None, None, None

    point_coords = np.array(input_points)
    point_labels = np.ones(len(input_points))
    
    # Return the masked image for debugging visualization
    return point_coords, point_labels, masked_image

# --- 3. MAIN VIDEO LOOP ---
video_path = "Dave Day 2025.mp4"
cap = cv2.VideoCapture(video_path)

window_name = "Automated SAM Lane Detection"
debug_window_name = "White Lane Mask (Debug View)" # New window for debugging
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.namedWindow(debug_window_name, cv2.WINDOW_NORMAL)

success, frame = cap.read()
if success:
    print("Setting initial image for SAM predictor...")
    predictor.set_image(frame)
    print("Ready to process video.")
else:
    print(f"Error: Failed to read the first frame from {video_path}")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Get prompts and the debug mask
    prompt_points, prompt_labels, debug_mask = get_strategic_lane_prompts(frame)

    display_frame = frame.copy()
    current_mask = None

    if prompt_points is not None and len(prompt_points) > 0:
        predictor.set_image(frame)
        masks, scores, logits = predictor.predict(
            point_coords=prompt_points,
            point_labels=prompt_labels,
            multimask_output=False, 
        )
        current_mask = masks[0]

    if current_mask is not None:
        color = np.array([30, 144, 255])
        h, w = current_mask.shape[-2:]
        mask_image = current_mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        cv2.addWeighted(mask_image.astype(np.uint8), 0.6, display_frame, 0.4, 0, display_frame)

    if prompt_points is not None:
        for point in prompt_points:
            cv2.circle(display_frame, tuple(point), 8, (0, 255, 0), -1)

    cv2.imshow(window_name, display_frame)
    if debug_mask is not None:
        # Show the black and white mask to see what the algorithm is "seeing"
        cv2.imshow(debug_window_name, debug_mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break

cap.release()
cv2.destroyAllWindows()