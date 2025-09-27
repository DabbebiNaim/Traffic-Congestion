import cv2

# --- INSTRUCTIONS ---
# 1. Change the video_path below to your NEW video file.
# 2. Run this script: python get_screenshot.py
# 3. A video window will open.
#
# --- CONTROLS ---
#   - SPACEBAR : Pause / Play the video to find a clear frame.
#   - s        : Save the current frame. It will be named "new_screenshot.jpg".
#   - q        : Quit the tool.

# --- STEP 1: SET THE PATH TO YOUR NEW VIDEO ---
video_path = "five minutes of evening US101 Palo Alto traffic from Embarcadero crossing pedestrian bridge.mp4"  # <--- CHANGE THIS LINE

# --- The rest of the script handles the logic ---

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open the video file at '{video_path}'")
else:
    window_name = "Screenshot Tool"
    print("Video opened. Find a clear frame and press 's' to save.")
    print("Controls: [SPACE] = Pause/Play | [s] = Save | [q] = Quit")

    paused = False

    while True:
        # Only read a new frame if the video is not paused
        if not paused:
            success, frame = cap.read()
            if not success:
                print("End of video reached.")
                break
        
        # Display the current frame
        cv2.imshow(window_name, frame)

        # Wait for user input
        key = cv2.waitKey(20) & 0xFF

        # Handle keyboard controls
        if key == ord('q'):
            print("Quitting tool.")
            break
        elif key == ord('s'):
            # Save the ORIGINAL, full-resolution frame
            output_filename = "new_screenshot.jpg"
            cv2.imwrite(output_filename, frame)
            print(f"--- Screenshot saved as '{output_filename}'! ---")
            print(f"Resolution of saved image: {frame.shape[1]}x{frame.shape[0]}")
        elif key == 32:  # ASCII code for SPACEBAR
            paused = not paused
            if paused:
                print("Video Paused.")
            else:
                print("Video Playing.")

    # Clean up and close windows
    cap.release()
    cv2.destroyAllWindows()