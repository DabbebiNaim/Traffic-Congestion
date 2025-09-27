import torch

# The path to the model file you downloaded
model_path = 'polylanenet_model.pt'

print(f"--- Inspecting Model File: {model_path} ---")

try:
    # Load the entire checkpoint file
    checkpoint = torch.load(model_path, map_location='cpu')

    # Check if the loaded object is a dictionary (which it should be for checkpoints)
    if isinstance(checkpoint, dict):
        print("\nCheckpoint is a dictionary. Here are its top-level keys:")
        print("---------------------------------------------------------")
        for key in checkpoint.keys():
            print(key)
        print("---------------------------------------------------------")

        # Now, let's look inside the 'model' state dictionary, if it exists
        if 'model' in checkpoint:
            print("\nKeys inside the 'model' state_dict (first 10 keys):")
            print("----------------------------------------------------")
            model_state_dict = checkpoint['model']
            for i, key in enumerate(model_state_dict.keys()):
                if i < 10:
                    print(key)
            print("----------------------------------------------------")
        else:
            print("\nCould not find a 'model' key in the checkpoint.")

    else:
        print("\nCheckpoint is NOT a dictionary. It might be the model state_dict directly.")
        print("Here are the first 10 keys:")
        print("---------------------------------------------------------")
        for i, key in enumerate(checkpoint.keys()):
             if i < 10:
                print(key)
        print("---------------------------------------------------------")


except Exception as e:
    print(f"\n--- An Error Occurred ---")
    print(f"Could not inspect the file. The error was: {e}")
    print("This might indicate the file is corrupted or not a valid PyTorch file.")