import cv2
import numpy as np

# Function to apply region of interest (ROI) mask
def apply_roi(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Function to detect and draw lanes
def detect_lanes(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Define region of interest (ROI) vertices
    height, width = edges.shape[:2]
    roi_vertices = np.array([[(0, height), (width/2 - 50, height/2 + 50), (width/2 + 50, height/2 + 50), (width, height)]], dtype=np.int32)
    roi_edges = apply_roi(edges, roi_vertices)

    # Apply Hough transform to detect lines
    lines = cv2.HoughLinesP(roi_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)

    # Separate left and right lane lines
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)
        if slope < -0.5:
            left_lines.append((x1, y1, x2, y2))
        elif slope > 0.5:
            right_lines.append((x1, y1, x2, y2))

    # Fit the left and right lane lines using polynomial regression
    left_line = fit_lane_line(left_lines, height)
    right_line = fit_lane_line(right_lines, height)

    # Create an empty image to draw the lanes
    lane_image = np.zeros_like(image)

    # Draw the left lane line
    if left_line is not None:
        draw_curve(lane_image, left_line, color=(0, 255, 0), thickness=10)

    # Draw the right lane line
    if right_line is not None:
        draw_curve(lane_image, right_line, color=(0, 255, 0), thickness=10)

    # Create a mask for the area between the lanes
    lane_mask = np.zeros_like(image)

    if left_line is not None and right_line is not None:
        vertices = np.array([[(left_line[0][0], left_line[0][1]), (right_line[0][0], right_line[0][1]),
                              (right_line[-1][0], right_line[-1][1]), (left_line[-1][0], left_line[-1][1])]], dtype=np.int32)
        cv2.fillPoly(lane_mask, vertices, (0, 255, 0))

    # Combine the lane image and the lane mask
    lane_image = cv2.addWeighted(lane_image, 0.8, lane_mask, 0.2, 0)

    # Overlay the lane image on the original frame
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    return result


# Function to fit a lane line using polynomial regression
def fit_lane_line(lines, height, prev_fit=None, alpha=0.1):
    if len(lines) == 0:
        return None

    xs = []
    ys = []

    for line in lines:
        x1, y1, x2, y2 = line
        xs.append(x1)
        xs.append(x2)
        ys.append(y1)
        ys.append(y2)

    fit = np.polyfit(ys, xs, deg=2)

    # Smooth the lane fit using previous frame's fit
    if prev_fit is not None:
        fit = alpha * fit + (1 - alpha) * prev_fit # Use degree 2 for polynomial curve fitting
    min_y = int(height * 0.7)
    max_y = height

    curve_ys = np.linspace(min_y, max_y, num=100)
    curve_xs = np.polyval(fit, curve_ys)

    curve_points = np.column_stack((curve_xs, curve_ys)).astype(np.int32)

    return curve_points


# Function to draw a curve on an image
def draw_curve(image, curve_points, color, thickness):
    cv2.polylines(image, [curve_points], isClosed=False, color=color, thickness=thickness)


# Main loop
cap = cv2.VideoCapture("lane.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect lanes and fill the area between them with green
    result = detect_lanes(frame)

    # Display the result
    cv2.imshow("Autonomous RC Car", result)

    # Exit the loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
