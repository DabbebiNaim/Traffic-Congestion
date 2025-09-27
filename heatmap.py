import cv2
import numpy as np
import os

class SmoothTrafficHeatmap:
    """
    A class to generate a smooth heatmap and analyze the overall traffic state.
    """
    def __init__(self, width, height, blur_kernel_size=(35, 35), decay_rate=0.99, heatmap_cap=10.0,
                 light_threshold_pct=0.02, medium_threshold_pct=0.05):
        """
        Initializes the smooth heatmap generator and traffic analyzer.

        Args:
            width (int): The width of the video frames.
            height (int): The height of the video frames.
            blur_kernel_size (tuple): The (width, height) for the Gaussian blur kernel.
            decay_rate (float): The rate at which the heatmap fades.
            heatmap_cap (float): The maximum value a pixel in the accumulator can reach.
            light_threshold_pct (float): Percentage of total possible heat to be considered "Light" traffic.
            medium_threshold_pct (float): Percentage of total possible heat to be considered "Medium" traffic.
        """
        self.width = width
        self.height = height
        self.decay_rate = decay_rate
        self.blur_kernel_size = blur_kernel_size
        self.heatmap_cap = heatmap_cap
        
        self.heatmap_accumulator = np.zeros((height, width), dtype=np.float32)
        
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )

        max_possible_heat = self.width * self.height * self.heatmap_cap
        self.light_threshold = max_possible_heat * light_threshold_pct
        self.medium_threshold = max_possible_heat * medium_threshold_pct
        print(f"Traffic State Thresholds: Light < {self.light_threshold:.0f}, Medium < {self.medium_threshold:.0f}")


    def process_frame(self, frame):
        """
        Processes a single video frame to update the heatmap and analyze traffic state.

        Args:
            frame (np.ndarray): The input video frame.

        Returns:
            ### MODIFIED ###
            tuple: A tuple containing:
                - np.ndarray: The original frame with the smooth heatmap overlaid.
                - str: The analyzed traffic state ("Light", "Medium", "High").
                - float: The raw total heat score for the current frame.
        """
        # 1. Detect moving objects
        fg_mask = self.bg_subtractor.apply(frame)
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # 2. Apply spatial smoothing
        blurred_mask = cv2.GaussianBlur(fg_mask, self.blur_kernel_size, 0)
        
        # 3. Update the heatmap accumulator
        self.heatmap_accumulator += blurred_mask / 255.0
        
        # 4. Apply temporal smoothing (decay)
        self.heatmap_accumulator *= self.decay_rate
        
        # 5. Apply intensity capping
        np.clip(self.heatmap_accumulator, 0, self.heatmap_cap, out=self.heatmap_accumulator)

        # 6. Normalize and colorize the heatmap for visualization
        normalized_heatmap = (self.heatmap_accumulator / self.heatmap_cap) * 255
        normalized_heatmap = normalized_heatmap.astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)
        
        # 7. Blend the heatmap with the original frame
        alpha = 0.5
        overlaid_frame = cv2.addWeighted(frame, 1 - alpha, colored_heatmap, alpha, 0)
        
        # Analyze traffic state
        total_heat = np.sum(self.heatmap_accumulator)
        
        if total_heat < self.light_threshold:
            traffic_state = "Light"
        elif total_heat < self.medium_threshold:
            traffic_state = "Medium"
        else:
            traffic_state = "High"
            
        # ### MODIFIED: Return the frame, state, AND the raw heat value ###
        return overlaid_frame, traffic_state, total_heat

def main():
    """
    Main function to run the traffic heatmap generation and analysis on a video.
    """
    video_path = 'five minutes of evening US101 Palo Alto traffic from Embarcadero crossing pedestrian bridge.mp4'
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    heatmap_generator = SmoothTrafficHeatmap(
        width=frame_width, 
        height=frame_height,
        light_threshold_pct=0.02,
        medium_threshold_pct=0.05
    )

    print("Processing video... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or video error.")
            break
        
        # ### MODIFIED: Receive the frame, state, and the heat value ###
        heatmap_frame, traffic_state, total_heat_value = heatmap_generator.process_frame(frame)
        
        # ### MODIFIED: Draw the expanded text with the heat score ###
        # Increase the size of the background rectangle to fit more text
        cv2.rectangle(heatmap_frame, (10, 10), (450, 45), (0, 0, 0), -1) 
        
        # Format the text string to include the traffic state and the heat score
        info_text = f"Traffic: {traffic_state} | Heat Score: {total_heat_value:.0f}"
        
        cv2.putText(
            heatmap_frame, 
            info_text, 
            (20, 35), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, (255, 255, 255), 2
        )
        
        cv2.imshow('Smooth Traffic Heatmap & Analysis', heatmap_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Processing finished.")

if __name__ == "__main__":
    main()