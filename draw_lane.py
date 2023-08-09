import cv2
import numpy as np

def detect_lane_lines(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply color masks for white and yellow lanes
    white_lower = np.array([200, 200, 200], dtype=np.uint8)
    white_upper = np.array([255, 255, 255], dtype=np.uint8)
    yellow_lower = np.array([150, 150, 0], dtype=np.uint8)
    yellow_upper = np.array([255, 255, 150], dtype=np.uint8)
    white_mask = cv2.inRange(image, white_lower, white_upper)
    yellow_mask = cv2.inRange(image, yellow_lower, yellow_upper)

    # Combine the white and yellow masks
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Apply Gaussian blur to the masked image
    blur = cv2.GaussianBlur(combined_mask, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    return edges

def draw_lane_lines(image, lines, color=(0, 255, 0), thickness=3):
    # Create a blank image to draw the lines on
    lane_image = np.zeros_like(image)

    # If there are no lines detected, return the original image
    if lines is None:
        return image

    # Iterate over the detected lines and draw them on the lane image
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(lane_image, (x1, y1), (x2, y2), color, thickness)

    # Combine the lane image with the original image
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    return result

# Load the image
image = cv2.imread('road.jpeg')

# Detect lane lines
edges = detect_lane_lines(image)

# Perform Hough line detection
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=50)

# Draw the lane lines if lines are detected
if lines is not None:
    result = draw_lane_lines(image, lines)
else:
    result = image

# Display the result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
