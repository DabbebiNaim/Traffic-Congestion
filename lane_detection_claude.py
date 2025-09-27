import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os

class SCNN(nn.Module):
    """
    Spatial CNN for Lane Detection
    """
    def __init__(self, input_size=(288, 800), num_classes=4):
        super(SCNN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        
        # VGG-16 backbone (feature extraction)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            
            # Block 5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Spatial CNN layers
        self.scnn_d = nn.Conv2d(512, 128, (1, 9), padding=(0, 4))  # Down
        self.scnn_u = nn.Conv2d(512, 128, (1, 9), padding=(0, 4))  # Up
        self.scnn_r = nn.Conv2d(512, 128, (9, 1), padding=(4, 0))  # Right
        self.scnn_l = nn.Conv2d(512, 128, (9, 1), padding=(4, 0))  # Left
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(1024, self.num_classes, 1)
        )
        
    def spatial_conv(self, x):
        # Apply spatial convolutions in four directions
        h, w = x.size()[2:]
        
        # Down direction
        x_d = self.scnn_d(x)
        for i in range(1, h):
            x_d[:, :, i:i+1, :] = F.relu(x_d[:, :, i:i+1, :] + x_d[:, :, i-1:i, :])
        
        # Up direction
        x_u = self.scnn_u(x)
        for i in range(h-2, -1, -1):
            x_u[:, :, i:i+1, :] = F.relu(x_u[:, :, i:i+1, :] + x_u[:, :, i+1:i+2, :])
        
        # Right direction
        x_r = self.scnn_r(x)
        for i in range(1, w):
            x_r[:, :, :, i:i+1] = F.relu(x_r[:, :, :, i:i+1] + x_r[:, :, :, i-1:i])
        
        # Left direction
        x_l = self.scnn_l(x)
        for i in range(w-2, -1, -1):
            x_l[:, :, :, i:i+1] = F.relu(x_l[:, :, :, i:i+1] + x_l[:, :, :, i+1:i+2])
        
        return torch.cat([x_d, x_u, x_r, x_l], dim=1)
    
    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        
        # Spatial CNN
        x = self.spatial_conv(x)
        
        # Classification
        x = self.classifier(x)
        
        # Upsample to original size
        x = F.interpolate(x, size=self.input_size, mode='bilinear', align_corners=False)
        
        return x
class LaneDetector:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu', use_traditional=True):
        self.device = device
        self.use_traditional = use_traditional
        
        if not use_traditional:
            self.model = SCNN(input_size=(288, 800), num_classes=4)
            
            if model_path and os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Model loaded from {model_path}")
            else:
                # If you switch to DL mode, it's better to raise an error if the model isn't found
                print("WARNING: No pre-trained model found for deep learning mode. Model will be random.")
                
            self.model.to(device)
            self.model.eval()
            
            # Image preprocessing (only needed for deep learning mode)
            self.transform = transforms.Compose([
                transforms.Resize((288, 800)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            print("Using traditional computer vision approach for lane detection.")
        
    def preprocess_image(self, image):
        """Preprocess image for model input (only used in deep learning mode)"""
        if not hasattr(self, 'transform'):
            return None
            
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def postprocess_output(self, output, original_shape):
        """Convert model output to lane coordinates"""
        output = torch.softmax(output, dim=1)
        output = output.squeeze().cpu().numpy()
        
        lanes = []
        h, w = original_shape[:2]
        
        # Process each lane (excluding background)
        for lane_idx in range(1, output.shape[0]):
            lane_mask = output[lane_idx]
            
            # Resize to original image size
            lane_mask = cv2.resize(lane_mask, (w, h))
            
            # Find lane points
            lane_points = self.extract_lane_points(lane_mask, threshold=0.3)
            if len(lane_points) > 0:
                lanes.append(lane_points)
                
        return lanes

    def extract_lane_points(self, lane_mask, threshold=0.3):
        """Extract lane points from probability mask"""
        # This method was previously nested incorrectly. It's now a proper class method.
        points = []
        h, w = lane_mask.shape
        
        # Sample points along vertical lines
        for y in range(h//4, h, 10):  # Start from 1/4 height, sample every 10 pixels
            row = lane_mask[y, :]
            peaks = np.where(row > threshold)[0]
            
            if len(peaks) > 0:
                # Take the peak with highest probability
                best_x = peaks[np.argmax(row[peaks])]
                points.append((best_x, y))
                
        return points
    
    def detect_lanes_traditional(self, image):
        """Traditional computer vision approach for lane detection"""
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()
            
        if img is None:
            return [], None
            
        original_img = img.copy()
        
        # Region of interest (focus on lower half of image)
        height, width = img.shape[:2]
        
        # Define trapezoid ROI for lane detection
        roi_vertices = np.array([
            [(int(width * 0.1), height),
             (int(width * 0.45), int(height * 0.6)),
             (int(width * 0.55), int(height * 0.6)),
             (int(width * 0.9), height)]
        ], dtype=np.int32)
        
        # Image preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection with adjusted parameters
        edges = cv2.Canny(blur, 50, 150)
        
        # Create mask for region of interest
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Hough line detection with adjusted parameters
        lines = cv2.HoughLinesP(masked_edges, 
                               rho=1,
                               theta=np.pi/180, 
                               threshold=30,
                               minLineLength=50, 
                               maxLineGap=100)
        
        # Separate left and right lane lines
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate slope and filter out horizontal lines
                if x2 - x1 != 0:
                    slope = (y2 - y1) / (x2 - x1)
                    
                    # Filter lines based on slope and position
                    if abs(slope) > 0.3:  # Minimum slope threshold
                        line_center_x = (x1 + x2) / 2
                        
                        if slope < 0 and line_center_x < width * 0.5:  # Left lane
                            left_lines.append(line[0])
                        elif slope > 0 and line_center_x > width * 0.5:  # Right lane
                            right_lines.append(line[0])
        
        # Process lane lines and create lane points
        lanes = []
        
        for line_group, side in [(left_lines, 'left'), (right_lines, 'right')]:
            if len(line_group) > 0:
                try:
                    # Collect all points from lines
                    x_coords = []
                    y_coords = []
                    
                    for line in line_group:
                        x1, y1, x2, y2 = line
                        x_coords.extend([x1, x2])
                        y_coords.extend([y1, y2])
                    
                    # Fit polynomial to points
                    if len(x_coords) >= 2:
                        coeffs = np.polyfit(y_coords, x_coords, 1)
                        
                        # Generate lane points
                        y_top = int(height * 0.6)
                        y_bottom = height - 10
                        
                        lane_points = []
                        for y in range(y_top, y_bottom, 15):
                            x = int(np.polyval(coeffs, y))
                            
                            # Ensure points are within image bounds
                            if 0 <= x < width:
                                lane_points.append((x, y))
                        
                        if len(lane_points) >= 2:
                            lanes.append(lane_points)
                            
                except (np.RankWarning, np.linalg.LinAlgError):
                    # Skip if polynomial fitting fails
                    continue
        
        return lanes, original_img

    # ########################################################
    # ##### THIS IS THE CORRECTED METHOD #####
    # ########################################################
    def detect_lanes(self, image):
        """
        Main lane detection function.
        Routes to the correct method (traditional or deep learning) based on initialization.
        """
        if self.use_traditional:
            # Use the traditional computer vision method
            return self.detect_lanes_traditional(image)
        else:
            # Use the deep learning (SCNN) method
            # Store original image shape
            if isinstance(image, str):
                original_img = cv2.imread(image)
                if original_img is None:
                    raise FileNotFoundError(f"Image not found at {image}")
            else:
                original_img = image.copy()
            
            original_shape = original_img.shape
            
            # Preprocess
            input_tensor = self.preprocess_image(image)
            
            # Inference
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Postprocess
            lanes = self.postprocess_output(output, original_shape)
            
            return lanes, original_img

    def draw_lanes(self, image, lanes, colors=None):
        """Draw detected lanes on image"""
        if colors is None:
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        
        result = image.copy()
        
        for i, lane_points in enumerate(lanes):
            if len(lane_points) < 2:
                continue
                
            color = colors[i % len(colors)]
            
            # Draw lane points
            for point in lane_points:
                cv2.circle(result, point, 5, color, -1)
            
            # Fit polynomial and draw smooth curve
            if len(lane_points) >= 3:
                points_array = np.array(lane_points)
                x_coords = points_array[:, 0]
                y_coords = points_array[:, 1]
                
                # Fit 2nd degree polynomial
                try:
                    coeffs = np.polyfit(y_coords, x_coords, 2)
                    
                    # Generate smooth curve
                    y_range = np.linspace(min(y_coords), max(y_coords), 100)
                    x_range = np.polyval(coeffs, y_range)
                    
                    # Draw curve
                    curve_points = np.column_stack([x_range, y_range]).astype(np.int32)
                    cv2.polylines(result, [curve_points], False, color, 3)
                    
                except np.RankWarning:
                    # If polynomial fitting fails, draw straight lines
                    for j in range(len(lane_points) - 1):
                        cv2.line(result, lane_points[j], lane_points[j+1], color, 3)
        
        return result

def main():
    """Example usage"""
    # Initialize detector with traditional CV approach (more reliable without pre-trained weights)
    detector = LaneDetector(use_traditional=True)
    
    # Example with webcam
    def detect_from_webcam():
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect lanes
            lanes, original_img = detector.detect_lanes(frame)
            
            # Draw results
            result = detector.draw_lanes(original_img, lanes)
            
            # Add lane count info
            cv2.putText(result, f'Lanes detected: {len(lanes)}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('Lane Detection', result)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    # Example with image file
    def detect_from_image(image_path):
        lanes, original_img = detector.detect_lanes(image_path)
        result = detector.draw_lanes(original_img, lanes)
        
        # Display results
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected Lanes ({len(lanes)} lanes found)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return lanes
    
    # Example with video file
    def detect_from_video(video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"Processing frame {frame_count}/{total_frames}", end='\r')
            
            # Detect lanes
            lanes, original_img = detector.detect_lanes(frame)
            
            # Draw results
            result = detector.draw_lanes(original_img, lanes)
            
            # Add info overlay
            cv2.putText(result, f'Lanes: {len(lanes)} | Frame: {frame_count}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save or display
            if output_path:
                out.write(result)
            else:
                cv2.imshow('Lane Detection - Video', result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        print(f"\nProcessed {frame_count} frames")
        cap.release()
        if output_path:
            out.release()
            print(f"Output saved to: {output_path}")
        cv2.destroyAllWindows()
    
    # Usage examples
    print("Lane Detection Script (Traditional CV Mode)")
    print("1. detect_from_webcam() - Real-time detection from webcam")
    print("2. detect_from_image('path/to/image.jpg') - Detection from image")
    print("3. detect_from_video('path/to/video.mp4') - Detection from video")
    print("\nTo use deep learning mode, initialize with: LaneDetector(use_trad" \
    "itional=False)")
    
    # Test with your video file
    video_file = '4K Video of Highway Traffic!.mp4'
    if os.path.exists(video_file):
        print(f"\nProcessing video: {video_file}")
        detect_from_video(video_file, 'output_lanes_detected.mp4')
    else:
        print(f"\nVideo file '{video_file}' not found in current directory")
        print("Available files:")
        for file in os.listdir('.'):
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                print(f"  - {file}")
    
    # Uncomment the lines below to test other modes
    # detect_from_webcam()
    # lanes = detect_from_image('path/to/your/image.jpg')

if __name__ == "__main__":
    main()