import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# --- 1. LOAD PRE-TRAINED DEEPLABV3+ MODEL ---
print("Loading DeepLabV3+ model from TensorFlow Hub...")
# More models can be found at: https://tfhub.dev/s?q=deeplab
model_url = "https://tfhub.dev/tensorflow/deeplabv3/mnv2_pascal/1"
model = hub.load(model_url)
print("Model loaded successfully.")

# --- 2. VIDEO SETUP ---
# To use your webcam, change the video_path to 0
video_path = "path_to_your_video.mp4" 
cap = cv2.VideoCapture(video_path)
window_name = "DeepLabV3+ Real-Time Segmentation"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# --- 3. HELPER FUNCTION FOR VISUALIZATION ---
def create_pascal_label_colormap():
    """Creates a color map for visualizing PASCAL VOC segmentation.
    Returns:
        A colormap mapping class IDs to RGB colors.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap

def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label."""
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')
    
    colormap = create_pascal_label_colormap()
    return colormap[label]

# --- 4. MAIN VIDEO LOOP ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("End of video or failed to read frame.")
        break

    # --- PREPARE FRAME FOR MODEL ---
    # The model expects RGB images with pixel values normalized to [0, 1]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.image.convert_image_dtype(rgb_frame, tf.float32)[tf.newaxis, ...]

    # --- PERFORM SEGMENTATION ---
    results = model(input_tensor)
    
    # The output is a tensor containing the segmentation mask
    segmentation_mask = tf.argmax(results['predictions'][0], axis=-1)
    
    # Convert the mask to a color image for visualization
    mask_vis = label_to_color_image(segmentation_mask.numpy()).astype(np.uint8)

    # --- VISUALIZATION ---
    # Resize mask to match original frame dimensions
    mask_vis_resized = cv2.resize(mask_vis, (frame.shape[1], frame.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)

    # Blend the original frame with the segmentation mask
    overlay = cv2.addWeighted(frame, 0.7, mask_vis_resized, 0.3, 0)
    
    cv2.imshow(window_name, overlay)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 5. CLEANUP ---
cap.release()
cv2.destroyAllWindows()