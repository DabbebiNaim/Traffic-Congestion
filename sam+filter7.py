import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# --- 1. SETUP SAM MODEL ---
print("Loading SAM model... This may take a moment.")
model_type = "vit_b"
sam_checkpoint = "sam_vit_b.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.94,
    min_mask_region_area=200, # Increased this to pre-filter small noise
)

# --- 2. VIDEO SETUP ---
video_path = "4K Video of Highway Traffic!.mp4"
cap = cv2.VideoCapture(video_path)
window_name = "SAM Automatic Road Segmentation (Smart Filter)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# --- 3. MAIN VIDEO LOOP ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("End of video.")
        break

    analysis_frame = frame.copy()
    
    # --- STAGE 1: SEGMENT EVERYTHING ---
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    generated_masks = mask_generator.generate(frame_rgb)

    # --- STAGE 2: SMART FILTERING TO FIND THE ROAD ---
    road_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    hls_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    for mask_data in generated_masks:
        segmentation_mask = mask_data['segmentation']
        
        # --- Filter 1: Area (already partially handled by min_mask_region_area) ---
        # The area is provided by SAM, we can do an additional check if needed.
        area = mask_data['area']
        if area < 500: # You can increase this threshold to ignore smaller objects
            continue

        # Get the bounding box of the mask for geometric filtering
        x, y, w, h = mask_data['bbox']
        
        # --- Filter 2: Position ---
        # The object must be in the bottom 70% of the screen.
        if (y + h) < (frame_height * 0.3):
            continue

        # --- Filter 3: Shape ---
        # The object should be wider than it is tall.
        aspect_ratio = w / h
        if aspect_ratio < 1.0: # Discard tall, thin objects
            continue

        # --- Filter 4: Color (The Final Check) ---
        # Only check the color for masks that passed the geometric tests.
        binary_mask = segmentation_mask.astype(bool)
        
        # Get the median HLS values from under the mask
        median_lightness = np.median(hls_frame[:, :, 1][binary_mask])
        median_saturation = np.median(hls_frame[:, :, 2][binary_mask])
        
        # Tune these color thresholds
        is_road_color = median_saturation < 50 and 50 < median_lightness < 210

        if is_road_color:
            # If it passes all tests, add this mask to our final road_mask
            road_mask[binary_mask] = 255

    # --- VISUALIZATION ---
    road_overlay = np.zeros_like(analysis_frame)
    road_overlay[road_mask == 255] = (0, 255, 0) # Green for road
    display_frame = cv2.addWeighted(analysis_frame, 0.7, road_overlay, 0.3, 0)
    
    cv2.imshow(window_name, display_frame)
    cv2.imshow("Final Road Mask", road_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()