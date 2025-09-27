import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from collections import OrderedDict
import sys

# --- MODEL DEFINITION ---
# This class defines the architecture we are trying to load the weights into.
class PolyLaneNet(nn.Module):
    def __init__(self, num_outputs=35):
        super(PolyLaneNet, self).__init__()
        # The feature extractor is named 'backbone'
        self.backbone = EfficientNet.from_pretrained('efficientnet-b1')
        num_ftrs = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()
        # The final layer is named 'regressor'
        self.regressor = nn.Linear(num_ftrs, num_outputs)

    def forward(self, x):
        x = self.backbone(x)
        x = self.regressor(x)
        return x

# --- THE DIAGNOSTIC TOOL ---
model_path = 'polylanenet_model.pt'
print(f"--- Starting Diagnostic for '{model_path}' ---")

try:
    # Load the file into memory
    checkpoint = torch.load(model_path, map_location='cpu')
    print("Checkpoint file loaded successfully.")
except Exception as e:
    print(f"\nCRITICAL ERROR: Could not load the file. The file may be corrupt or not a PyTorch file.")
    print(f"Error details: {e}")
    sys.exit() # Stop the script if the file can't even be opened

# --- Test Case 1: Direct Load of the state_dict inside the 'model' key ---
print("\n--- Test 1: Assuming checkpoint['model'] is the correct state_dict ---")
try:
    model = PolyLaneNet()
    model.load_state_dict(checkpoint['model'])
    print("\n\n" + "="*50)
    print(">>> SUCCESS! Method 1 Worked. <<<")
    print("The model weights are stored directly in the 'model' key.")
    print("="*50)
    sys.exit() # Exit successfully
except Exception as e:
    print(">>> FAILED. Reason:")
    print(e)
    print("-"*50)


# --- Test Case 2: Removing a 'module.' prefix from the keys ---
print("\n--- Test 2: Assuming keys have a 'module.' prefix that needs to be removed ---")
try:
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model'].items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model = PolyLaneNet()
    model.load_state_dict(new_state_dict)
    print("\n\n" + "="*50)
    print(">>> SUCCESS! Method 2 Worked. <<<")
    print("The model was saved with a 'module.' prefix, which has been removed.")
    print("="*50)
    sys.exit() # Exit successfully
except Exception as e:
    print(">>> FAILED. Reason:")
    print(e)
    print("-"*50)


# --- Test Case 3: Renaming a 'model.' prefix to 'backbone.' ---
print("\n--- Test 3: Assuming keys are prefixed with 'model.' and need renaming to 'backbone.' ---")
try:
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model'].items():
        if k.startswith('model.'):
            name = k.replace('model.', 'backbone.', 1)
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    
    model = PolyLaneNet()
    model.load_state_dict(new_state_dict)
    print("\n\n" + "="*50)
    print(">>> SUCCESS! Method 3 Worked. <<<")
    print("The model's keys were renamed from 'model.*' to 'backbone.*'.")
    print("="*50)
    sys.exit() # Exit successfully
except Exception as e:
    print(">>> FAILED. Reason:")
    print(e)
    print("-"*50)


print("\n--- DIAGNOSTIC COMPLETE ---")
print("All standard loading methods have failed. The model structure in the file is unusual.")
print("Please provide the full output of this script for further analysis.")