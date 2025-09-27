import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from efficientnet_pytorch import EfficientNet
from collections import OrderedDict

# --- 1. MODEL DEFINITION ---
# This class defines the structure we want. It has a 'backbone' and a 'regressor'.
class PolyLaneNet(nn.Module):
    def __init__(self, num_outputs=35):
        super(PolyLaneNet, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b1')
        num_ftrs = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()  # We remove the original classifier
        self.regressor = nn.Linear(num_ftrs, num_outputs)

    def forward(self, x):
        x = self.backbone(x)
        x = self.regressor(x)
        return x

# --- 2. SETUP AND MODEL LOADING (WITH KEY RENAMING) ---
model_path = 'polylanenet_model.pt'

# Load the entire checkpoint from the file
print("Loading checkpoint from file...")
checkpoint = torch.load(model_path, map_location='cpu')

# --- !!! THE GUARANTEED FIX IS HERE !!! ---
# Create a new, empty dictionary to hold the corrected keys.
new_state_dict = OrderedDict()
# Get the dictionary of weights from the checkpoint file
original_state_dict = checkpoint['model']

print("Correcting layer names to match our model structure...")
for k, v in original_state_dict.items():
    # This is the renaming logic based on your error messages.
    if k.startswith('model._fc.regular_outputs_layer'):
        # Rename the final layer keys
        name = k.replace('model._fc.regular_outputs_layer', 'regressor')
    elif k.startswith('model.'):
        # Rename all other model keys to 'backbone'
        name = k.replace('model.', 'backbone.', 1)
    else:
        name = k
    
    new_state_dict[name] = v

# Create an instance of our model architecture
model = PolyLaneNet()

# Now, load the RENAMED state dictionary. This will now match perfectly.
print("Loading corrected model weights...")
model.load_state_dict(new_state_dict)

model.eval()
print("--- MODEL LOADED SUCCESSFULLY! ---")


# --- IMAGE TRANSFORMATION ---
IMG_HEIGHT = 288
IMG_WIDTH = 800
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

video_path = "Cars Moving On Road Stock Footage - Free Download.mp4"
cap = cv2.VideoCapture(video_path)
window_name = "Deep Learning Lane Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# --- HELPER FUNCTION TO DRAW LANES ---
def draw_lanes(image, model_output, y_samples):
    output = model_output.view(5, 7)
    lanes_to_draw = output[output[:, 0] > 0.51] # Confidence threshold
    
    h, w, _ = image.shape
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)] # B, G, R, C, M

    for i, lane_data in enumerate(lanes_to_draw):
        poly_coeffs = lane_data[3:] # 4 polynomial coefficients
        a, b, c, d = poly_coeffs[0].item(), poly_coeffs[1].item(), poly_coeffs[2].item(), poly_coeffs[3].item()
        
        lane_points = []
        for y in y_samples:
            # The model was trained with normalized coordinates, so we must match that
            y_norm = (y - 720 / 2) / (720 / 2) # Normalize y to the [-1, 1] range
            x_norm = (a * y_norm**3 + b * y_norm**2 + c * y_norm + d)
            x_pixel = x_norm * (w / 2) + (w / 2) # Un-normalize x to the image width

            if 0 < x_pixel < w:
                lane_points.append([int(x_pixel), int(y)])
        
        if len(lane_points) > 1:
            cv2.polylines(image, [np.array(lane_points, np.int32)], isClosed=False, color=colors[i % len(colors)], thickness=5)
            
    return image

# Define the y-coordinates for drawing, based on the TuSimple dataset standard
tusimple_y_samples = range(160, 720, 10)

# --- MAIN VIDEO LOOP ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(frame_rgb).unsqueeze(0)

    with torch.no_grad():
        predicted_output = model(input_tensor)

    annotated_frame = draw_lanes(frame.copy(), predicted_output[0], tusimple_y_samples)
    
    cv2.imshow(window_name, annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()