import cv2
import numpy as np
import pyrealsense2 as rs

def detect_cones(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours of objects
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for cone detection
    left_cones = []
    right_cones = []

    # Loop over contours
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)

        # Filter contours based on area
        if area > 1000:
            # Calculate approximate polygonal curve
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

            # Filter contours based on number of vertices
            if len(approx) >= 3:
                # Calculate bounding rectangle
                x, y, w, h = cv2.boundingRect(approx)

                # Classify cones based on position
                if x < frame.shape[1] / 2:
                    left_cones.append((x, y, w, h))
                else:
                    right_cones.append((x, y, w, h))

    return left_cones, right_cones

def find_path(left_cones, right_cones):
    # Sort cones based on the y-coordinate (top to bottom)
    left_cones = sorted(left_cones, key=lambda cone: cone[1])
    right_cones = sorted(right_cones, key=lambda cone: cone[1])

    # Find the path to go through the cones
    if len(left_cones) > 0 and len(right_cones) > 0:
        # Calculate the midpoint between the closest left and right cones
        mid_x = (left_cones[0][0] + right_cones[0][0] + right_cones[0][2]) // 2

        # Return the path coordinates
        return mid_x, frame.shape[0]
    else:
        # No cones detected, return None
        return None

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Detect cones
    left_cones, right_cones = detect_cones(frame)

    # Find path
    path = find_path(left_cones, right_cones)

    if path is not None:
        # Draw path on the frame
        mid_x, height = path
        cv2.line(frame, (mid_x, 0), (mid_x, height), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Cone Detection", frame)

    # Check for key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
