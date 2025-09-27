# get_coords.py

import cv2
import numpy as np

# This list will store the clicked points
points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point added: ({x}, {y}). Total points: {len(points)}")
        # Draw a circle on the image to show where you clicked
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Click to get coordinates", image)

# Load the screenshot
image_path = 'new_screenshot.jpg'
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load image at {image_path}")
else:
    cv2.imshow("Click to get coordinates", image)
    cv2.setMouseCallback("Click to get coordinates", mouse_callback)
    print("Click on the 4 corners of your reference rectangle, then press 'q' to quit.")
    cv2.waitKey(0) # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()

    print("\nCollected points:")
    print(points)
