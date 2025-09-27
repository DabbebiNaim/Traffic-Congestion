import cv2
import numpy as np
import os
from ultralytics import YOLO
from collections import defaultdict

# --- INSTRUCTIONS ---
# 1. Install necessary libraries:
#    pip install opencv-python numpy ultralytics
#
# 2. Change the VIDEO_PATH variable below to your video file.
#
# 3. The YOLO model 'yolov8n.pt' will be downloaded automatically on the first run.

class DirectionalVehicleHeatmap:
    """
    Generates and analyzes separate heatmaps for vehicles based on their direction of travel.
    """
    def __init__(self, width, height, blur_kernel_size=(45, 45), decay_rate=0.98, heatmap_cap=10.0,
                 light_threshold_pct=0.01, medium_threshold_pct=0.03):
        """
        Initializes the directional vehicle heatmap generator.

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
        
        # Create separate accumulators for each direction
        self.heatmap_accumulator_going = np.zeros((height, width), dtype=np.float32)
        self.heatmap_accumulator_coming = np.zeros((height, width), dtype=np.float32)
        
        # Initialize YOLO model for tracking
        print("Initializing YOLO model for tracking...")
        self.model = YOLO('yolov8n.pt')
        self.vehicle_classes = [2, 3, 5, 7] # car, motorcycle, bus, truck
        
        # Data structure to store the history of tracked objects
        self.track_history = defaultdict(list)
        
        # Calculate traffic state thresholds (will be applied to each lane's score)
        # Note: These thresholds now apply to a smaller area (one direction of traffic)
        # so they are made more sensitive by default.
        max_possible_heat = self.width * self.height * self.heatmap_cap
        self.light_threshold = max_possible_heat * light_threshold_pct
        self.medium_threshold = max_possible_heat * medium_threshold_pct
        print(f"Per-Lane Traffic Thresholds: Light < {self.light_threshold:.0f}, Medium < {self.medium_threshold:.0f}")

    def _get_traffic_state(self, total_heat):
        """Helper function to determine traffic state from a heat score."""
        if total_heat < self.light_threshold:
            return "Light"
        elif total_heat < self.medium_threshold:
            return "Medium"
        return "High"

    def process_frame(self, frame):
        """
        Processes a frame to track vehicles, determine direction, and update heatmaps.
        
        Returns:
            dict: A dictionary containing the final image and analysis data.
        """
        # --- 1. Vehicle Tracking ---
        results = self.model.track(frame, persist=True, classes=self.vehicle_classes, verbose=False, device='cpu')
        
        # Create black masks to draw detections for each direction
        vehicle_mask_going = np.zeros((self.height, self.width), dtype=np.uint8)
        vehicle_mask_coming = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Get boxes and track IDs
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id
        
        # A frame to draw bounding boxes on for visualization
        annotated_frame = frame.copy()

        if track_ids is not None:
            track_ids = track_ids.int().cpu().tolist()
            # --- 2. Determine Direction and Update Masks ---
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                
                # Calculate center point for tracking history
                track_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                history = self.track_history[track_id]
                history.append(track_point)
                if len(history) > 30:  # Keep last 30 positions
                    history.pop(0)

                # Determine direction based on vertical movement
                direction = "Unknown"
                box_color = (128, 128, 128) # Grey for unknown
                if len(history) > 5: # Need at least 5 points to determine direction
                    vertical_movement = history[-1][1] - history[0][1]
                    if vertical_movement < -5:  # Moved up (significant threshold to avoid noise)
                        direction = "Going"
                        cv2.rectangle(vehicle_mask_going, (x1, y1), (x2, y2), 255, -1)
                        box_color = (0, 255, 0) # Green for going
                    elif vertical_movement > 5: # Moved down
                        direction = "Coming"
                        cv2.rectangle(vehicle_mask_coming, (x1, y1), (x2, y2), 255, -1)
                        box_color = (0, 0, 255) # Red for coming

                # Annotate the frame with bounding boxes and IDs
                label = f"ID:{track_id}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # --- 3. Update Both Heatmaps ---
        for accumulator, mask in [(self.heatmap_accumulator_going, vehicle_mask_going), 
                                  (self.heatmap_accumulator_coming, vehicle_mask_coming)]:
            blurred_mask = cv2.GaussianBlur(mask, self.blur_kernel_size, 0)
            accumulator += blurred_mask / 255.0  # Add new heat
            accumulator *= self.decay_rate       # Apply temporal decay
            np.clip(accumulator, 0, self.heatmap_cap, out=accumulator) # Cap intensity
            
        # --- 4. Create Combined Visual Heatmap ---
        combined_accumulator = self.heatmap_accumulator_going + self.heatmap_accumulator_coming
        normalized_heatmap = (combined_accumulator / self.heatmap_cap) * 255
        normalized_heatmap = normalized_heatmap.astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)
        
        # Blend the heatmap with the annotated frame
        alpha = 0.5
        overlaid_frame = cv2.addWeighted(annotated_frame, 1 - alpha, colored_heatmap, alpha, 0)
        
        # --- 5. Analyze Each Direction's Traffic State ---
        total_heat_going = np.sum(self.heatmap_accumulator_going)
        total_heat_coming = np.sum(self.heatmap_accumulator_coming)
        
        state_going = self._get_traffic_state(total_heat_going)
        state_coming = self._get_traffic_state(total_heat_coming)
        
        return {
            "final_frame": overlaid_frame,
            "heat_going": total_heat_going,
            "state_going": state_going,
            "heat_coming": total_heat_coming,
            "state_coming": state_coming
        }

def main():
    """ Main function to run the directional vehicle heatmap generation on a video. """
    # --- CONFIGURATION ---
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
    
    heatmap_generator = DirectionalVehicleHeatmap(
        width=frame_width, height=frame_height
    )

    print("Processing video... Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or video error.")
                break
            
            # Process the frame to get heatmap and analysis
            analysis_data = heatmap_generator.process_frame(frame)
            final_frame = analysis_data["final_frame"]
            
            # --- Display Information On Screen ---
            cv2.rectangle(final_frame, (10, 10), (450, 75), (0, 0, 0), -1) 
            
            text_going = f"Going:  {analysis_data['state_going']:<7} | Heat: {analysis_data['heat_going']:.0f}"
            text_coming = f"Coming: {analysis_data['state_coming']:<7} | Heat: {analysis_data['heat_coming']:.0f}"
            
            cv2.putText(final_frame, text_going, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(final_frame, text_coming, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow('Directional Vehicle Heatmap (YOLO)', final_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Processing finished and windows closed.")

if __name__ == "__main__":
    main()