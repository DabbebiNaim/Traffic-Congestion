import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from collections import deque
from efficientnet_pytorch import EfficientNet  # <-- Import the correct backbone

# --- 1. NEW, CORRECT MODEL DEFINITION ---
# This class defines a model with an EfficientNet backbone, which matches
# the architecture of the pre-trained file.
class PolyLaneNet(nn.Module):
    def __init__(self, num_lanes=4, num_coeffs=3):
        super(PolyLaneNet, self).__init__()
        # Use a pre-trained EfficientNet-b0 as the backbone
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        
        # We need to find the number of output features from the backbone
        # For efficientnet-b0, the classifier's in_features is 1280
        backbone_out_features = self.backbone._fc.in_features
        
        # Define the regressor head
        self.regressor = nn.Linear(backbone_out_features, num_lanes * num_coeffs)

    def forward(self, x):
        # Pass input through the EfficientNet backbone
        x = self.backbone.extract_features(x)
        x = self.backbone._avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x

# --- 2. SETUP AND MODEL LOADING ---
model_path = 'polylanenet_model.pt'
model = PolyLaneNet()

# The loading logic from before is now correct because the architectures match
checkpoint = torch.load(model_path, map_location='cpu')

# The keys in the checkpoint have a 'model.' prefix, so we need to adjust them
# This creates a new dictionary with the 'model.' prefix removed from each key
state_dict = {k.replace('model.', ''): v for k, v in checkpoint['model'].items()}
model.load_state_dict(state_dict)

model.eval()
transform = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

video_path = "Cars Moving On Road Stock Footage - Free Download.mp4"
cap = cv2.VideoCapture(video_path)
window_name = "Robust Lane Detection (EfficientNet)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# --- TEMPORAL SMOOTHING SETUP ---
HISTORY_LENGTH = 5
coeffs_history = deque(maxlen=HISTORY_LENGTH)

# --- HELPER FUNCTION TO DRAW LANES ---
def draw_lanes(image, poly_coeffs, num_lanes=4, num_coeffs=3):
    coeffs = poly_coeffs.view(num_lanes, num_coeffs)
    h, w, _ = image.shape
    y_points = np.linspace(h * 0.4, h - 1, int(h * 0.6))
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
    for i, lane_coeffs in enumerate(coeffs):
        a, b, c = lane_coeffs[0].item(), lane_coeffs[1].item(), lane_coeffs[2].item()
        x_points = (a * (y_points/h)**2 + b * (y_points/h) + c) * w
        lane_points = [[int(x), int(y)] for y, x in zip(y_points, x_points) if 0 < x < w]
        if lane_points:
            cv2.polylines(image, [np.array(lane_points, np.int32)], isClosed=False, color=colors[i], thickness=4)
    return image

# --- MAIN VIDEO LOOP ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(frame_rgb).unsqueeze(0)

    with torch.no_grad():
        predicted_coeffs = model(input_tensor)

    coeffs_history.append(predicted_coeffs[0])
    
    smoothed_coeffs = predicted_coeffs[0]
    if len(coeffs_history) == HISTORY_LENGTH:
        stacked_coeffs = torch.stack(list(coeffs_history))
        smoothed_coeffs = torch.mean(stacked_coeffs, dim=0)

    annotated_frame = draw_lanes(frame.copy(), smoothed_coeffs)
    
    cv2.imshow(window_name, annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()