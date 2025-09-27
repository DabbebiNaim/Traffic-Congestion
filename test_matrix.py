# test_matrix.py
import numpy as np
import cv2



#  four corner points
TOP_LEFT     = [198, 483]
TOP_RIGHT    = [1055, 485]
BOTTOM_LEFT  = [5, 608]
BOTTOM_RIGHT = [1247, 609]

# real-world dimensions
RECT_WIDTH_METERS = 9.6
RECT_LENGTH_METERS = 28.8

# --- THE TEST ---

SOURCE_POINTS = np.float32([TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT])

# Define destination points 
DESTINATION_POINTS = np.float32([
    [0, 0], [RECT_WIDTH_METERS, 0],
    [0, RECT_LENGTH_METERS], [RECT_WIDTH_METERS, RECT_LENGTH_METERS]
])

# the transformation matrix
transformation_matrix = cv2.getPerspectiveTransform(SOURCE_POINTS, DESTINATION_POINTS)


# --- THE CRITICAL TEST ---
# We will transform ONLY the Top-Left point and see where it lands.
# It MUST land at [0, 0] for the matrix to be correct.

# The point we are testing
point_to_test = np.array([[TOP_LEFT]], dtype="float32")

# Apply the perspective transform
transformed_point = cv2.perspectiveTransform(point_to_test, transformation_matrix)

print("--- MATRIX TEST RESULTS ---")
print(f"We are testing the point we believe is TOP-LEFT: {TOP_LEFT}")
print(f"The matrix transformed it to the real-world coordinate: {transformed_point[0][0]}")
print("\n--- HOW TO INTERPRET ---")
print("If the transformed coordinate is [0. 0.], the matrix is CORRECT.")
print("If it's anything else, the point order in SOURCE_POINTS is WRONG.")