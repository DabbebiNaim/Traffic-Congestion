import cv2
import numpy as np
from collections import deque

# --- HELPER FUNCTIONS ---
def get_line_params(line):
    """Calculates slope and intercept for a single line."""
    x1, y1, x2, y2 = line[0]
    if x1 == x2: return None, None # Vertical line
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept

def get_averaged_line(lines):
    """
    Averages a group of lines to get a single representative line.
    Returns (slope, intercept).
    """
    if not lines: return None
    
    slopes = [get_line_params(line)[0] for line in lines if get_line_params(line)[0] is not None]
    intercepts = [get_line_params(line)[1] for line in lines if get_line_params(line)[1] is not None]

    if not slopes: return None

    # --- NEW: OUTLIER REJECTION ---
    # Reject this group of lines if the slopes are too inconsistent (likely noise from a car)
    if np.std(slopes) > 0.2: # Tune this threshold; a low value means lines must be very parallel
        return None
        
    return np.mean(slopes), np.mean(intercepts)

def find_intersection(m1, b1, m2, b2):
    """Finds the intersection point of two lines."""
    if m1 is None or m2 is None or abs(m1 - m2) < 1e-5: return None
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return int(x), int(y)

# --- MAIN SCRIPT ---
video_path = "4K Video of Highway Traffic!.mp4"
cap = cv2.VideoCapture(video_path)
window_name = "Robust Vanishing Point (Smart Filtering)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Temporal averaging setup
HISTORY_LENGTH = 10
left_line_history = deque(maxlen=HISTORY_LENGTH)
right_line_history = deque(maxlen=HISTORY_LENGTH)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    analysis_frame = frame.copy()
    h, w, _ = analysis_frame.shape

    # Pre-processing
    gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0) # Slightly more blur can help
    edges = cv2.Canny(blur, 50, 150)

    # ROI
    roi_vertices = np.array([
        [(0, h)], [(w * 0.45, h * 0.55)], [(w * 0.55, h * 0.55)], [(w, h)]
    ], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [roi_vertices], 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Line Detection
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 40, minLineLength=30, maxLineGap=150)

    left_lanes = []
    right_lanes = []
    center_x = w // 2

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope, _ = get_line_params(line)
            if slope is None: continue
            
            # --- NEW: SMARTER GROUPING ---
            # Group lines based on both slope and position
            if -1.5 < slope < -0.4 and x1 < center_x and x2 < center_x:
                left_lanes.append(line)
            elif 0.4 < slope < 1.5 and x1 > center_x and x2 > center_x:
                right_lanes.append(line)
    
    # Get the averaged line for the CURRENT frame after robust filtering
    current_left_line = get_averaged_line(left_lanes)
    current_right_line = get_averaged_line(right_lanes)
    
    # Only add to history if a clean line was found in this frame
    if current_left_line: left_line_history.append(current_left_line)
    if current_right_line: right_line_history.append(current_right_line)

    # Calculate the STABLE moving average from the history of clean lines
    stable_left_line, stable_right_line = None, None
    if left_line_history:
        stable_left_line = (np.mean([l[0] for l in left_line_history]), np.mean([l[1] for l in left_line_history]))
    if right_line_history:
        stable_right_line = (np.mean([l[0] for l in right_line_history]), np.mean([l[1] for l in right_line_history]))

    # Calculate vanishing point from the STABLE lines
    vanishing_point = find_intersection(stable_left_line[0] if stable_left_line else None, 
                                        stable_left_line[1] if stable_left_line else None, 
                                        stable_right_line[0] if stable_right_line else None,
                                        stable_right_line[1] if stable_right_line else None)

    # --- Visualization ---
    if stable_left_line:
        m, b = stable_left_line
        y1, y2 = h, int(h * 0.6)
        if m != 0:
            x1 = int((y1 - b) / m); x2 = int((y2 - b) / m)
            cv2.line(analysis_frame, (x1, y1), (x2, y2), (255, 0, 0), 10)
    if stable_right_line:
        m, b = stable_right_line
        y1, y2 = h, int(h * 0.6)
        if m != 0:
            x1 = int((y1 - b) / m); x2 = int((y2 - b) / m)
            cv2.line(analysis_frame, (x1, y1), (x2, y2), (0, 0, 255), 10)
    if vanishing_point:
        cv2.circle(analysis_frame, vanishing_point, 15, (0, 255, 0), -1)

    cv2.imshow(window_name, analysis_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()