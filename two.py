import cv2
import numpy as np
# Initialize the RealSense pipeline
# Initialize the RealSense pipeline with camera index 1
pipeline = cv2.VideoCapture(1)

# Configure depth and color streams (if applicable)
pipeline.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
pipeline.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Region of Interest (ROI) coordinates
roi_vertices = np.array([[(0, 480), (0, 320), (640, 320), (640, 480)]], dtype=np.int32)

# Canny edge detection parameters
canny_low_threshold = 50
canny_high_threshold = 150

# Hough transform parameters
rho = 1
theta = np.pi / 180
hough_threshold = 50
min_line_length = 100
max_line_gap = 50
while True:
    # Read a frame from the RealSense camera
    ret, frame = pipeline.read()
    
    # Apply grayscale and Gaussian blur to the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, canny_low_threshold, canny_high_threshold)
    
    # Apply region of interest mask
    masked_edges = cv2.bitwise_and(edges, cv2.fillPoly(np.zeros_like(edges), roi_vertices, 255))
    
    # Perform Hough transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, rho, theta, hough_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    # Draw lane lines on the frame
    lane_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
    
    # Combine lane lines with the original frame
    output_frame = cv2.addWeighted(frame, 0.8, lane_image, 1, 0)
    
    # Display the output frame
    cv2.imshow("Lane Detection", output_frame)
    
    # Check for key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the RealSense pipeline and close all windows
pipeline.release()
cv2.destroyAllWindows()
