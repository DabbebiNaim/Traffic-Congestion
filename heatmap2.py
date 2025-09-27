import cv2
import numpy as np
import os
from ultralytics import YOLO

# --- INSTRUCTIONS ---
# 1. Install necessary libraries:
#    pip install opencv-python numpy ultralytics
#
# 2. Change the VIDEO_PATH variable below to your video file.
#
# 3. The YOLO model 'yolov8n.pt' will be downloaded automatically on the first run.

class VehicleHeatmap:
    """
    A class to generate a smooth heatmap of detected vehicles using YOLO.
    """
    def __init__(self, width, height, blur_kernel_size=(35, 35), decay_rate=0.99, heatmap_cap=10.0,
                 light_threshold_pct=0.01, medium_threshold_pct=0.03):
        """
        Initializes the vehicle heatmap generator.

        Args:
            width (int): The width of the video frames.
            height (int): The height of the video frames.
            blur_kernel_size (tuple): The (width, height) for the Gaussian blur kernel.
            decay_rate (float): The rate at which the heatmap fades over time.
            heatmap_cap (float): The maximum value a pixel in the accumulator can reach.
            light_threshold_pct (float): Pct of total possible heat for "Light" traffic.
            medium_threshold_pct (float): Pct of total possible heat for "Medium" traffic.
        """
        self.width = width
        self.height = height
        self.decay_rate = decay_rate
        self.blur_kernel_size = blur_kernel_size
        self.heatmap_cap = heatmap_cap
        
        # Accumulator for the heatmap
        self.heatmap_accumulator = np.zeros((height, width), dtype=np.float32)
        
        # Initialize YOLO model
        print("Initializing YOLO model...")
        self.model = YOLO('yolov8n.pt')
        # COCO class IDs for vehicles: 2=car, 3=motorcycle, 5=bus, 7=truck
        self.vehicle_classes = [2, 3, 5, 7]
        print("YOLO model loaded.")

        # Calculate traffic state thresholds based on heat
        max_possible_heat = self.width * self.height * self.heatmap_cap
        self.light_threshold = max_possible_heat * light_threshold_pct
        self.medium_threshold = max_possible_heat * medium_threshold_pct
        print(f"Traffic State Thresholds: Light < {self.light_threshold:.0f}, Medium < {self.medium_threshold:.0f}")

    def process_frame(self, frame):
        """
        Processes a single video frame to detect vehicles and update the heatmap.

        Args:
            frame (np.ndarray): The input video frame.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The original frame with the heatmap and detections overlaid.
                - str: The analyzed traffic state ("Light", "Medium", "High").
                - float: The raw total heat score for the current frame.
        """
        # 1. Detect Vehicles using YOLO
        # We only care about detections, so we use 'predict' for speed.
        # 'persist=True' is not necessary if we don't need tracking across frames for the heatmap.
        results = self.model.predict(frame, classes=self.vehicle_classes, verbose=False)
        
        # Create a black mask to draw detections on
        vehicle_mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # Draw filled rectangles for each detected vehicle on the mask
        if results and results[0].boxes.shape[0] > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vehicle_mask, (x1, y1), (x2, y2), 255, -1) # Draw filled white rectangle

        # 2. Apply spatial smoothing (blur) to the vehicle mask
        # This makes the heatmap "blobby" and smooth instead of sharp-edged.
        blurred_mask = cv2.GaussianBlur(vehicle_mask, self.blur_kernel_size, 0)
        
        # 3. Update the heatmap accumulator
        # Add the blurred mask to the accumulator. Normalize by 255 to keep values small.
        self.heatmap_accumulator += blurred_mask / 255.0
        
        # 4. Apply temporal smoothing (decay)
        # This makes the heatmap fade out over time where there's no activity.
        self.heatmap_accumulator *= self.decay_rate
        
        # 5. Apply intensity capping
        # Prevents any single spot from becoming excessively bright.
        np.clip(self.heatmap_accumulator, 0, self.heatmap_cap, out=self.heatmap_accumulator)

        # 6. Normalize and colorize the heatmap for visualization
        normalized_heatmap = (self.heatmap_accumulator / self.heatmap_cap) * 255
        normalized_heatmap = normalized_heatmap.astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)
        
        # 7. Blend the heatmap with the original frame
        alpha = 0.5
        overlaid_frame = cv2.addWeighted(frame, 1 - alpha, colored_heatmap, alpha, 0)
        
        # (Optional) Draw bounding boxes on the final frame for verification
        if results and results[0].boxes.shape[0] > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(overlaid_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 8. Analyze traffic state based on total heat
        total_heat = np.sum(self.heatmap_accumulator)
        
        if total_heat < self.light_threshold:
            traffic_state = "Light"
        elif total_heat < self.medium_threshold:
            traffic_state = "Medium"
        else:
            traffic_state = "High"
            
        return overlaid_frame, traffic_state, total_heat

def main():
    """
    Main function to run the vehicle heatmap generation on a video.
    """
    # --- CONFIGURATION ---
    # !!! IMPORTANT: CHANGE THIS TO THE PATH OF YOUR VIDEO FILE !!!
    VIDEO_PATH = 'five minutes of evening US101 Palo Alto traffic from Embarcadero crossing pedestrian bridge.mp4'
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at '{VIDEO_PATH}'")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize the heatmap generator with video dimensions
    heatmap_generator = VehicleHeatmap(
        width=frame_width, 
        height=frame_height,
        # You can tweak these values for different visual effects
        decay_rate=0.98,
        blur_kernel_size=(45, 45),
        light_threshold_pct=0.01, # Lowered thresholds as detections are more sparse
        medium_threshold_pct=0.03
    )

    print("Processing video... Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or video error.")
                break
            
            # Process the frame to get the heatmap overlay
            heatmap_frame, traffic_state, total_heat_value = heatmap_generator.process_frame(frame)
            
            # --- Display Information On Screen ---
            # Create a black rectangle for text background
            cv2.rectangle(heatmap_frame, (10, 10), (450, 45), (0, 0, 0), -1) 
            
            # Format the text string
            info_text = f"Traffic: {traffic_state} | Heat Score: {total_heat_value:.0f}"
            
            # Put the text on the frame
            cv2.putText(
                heatmap_frame, 
                info_text, 
                (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (255, 255, 255), 2
            )
            
            # Show the final frame
            cv2.imshow('Vehicle Heatmap (YOLO)', heatmap_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Ensure resources are released
        cap.release()
        cv2.destroyAllWindows()
        print("Processing finished and windows closed.")

if __name__ == "__main__":
    main()