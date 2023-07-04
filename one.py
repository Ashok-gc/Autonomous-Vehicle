import cv2
import numpy as np
import pyrealsense2 as rs

def detect_lanes(image):
    # Define region of interest (ROI)
    height, width = image.shape[:2]
    roi_vertices = [(0, height), (width // 2, height // 2), (width, height)]
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Mask the ROI
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.array([roi_vertices], dtype=np.int32), 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Apply Hough line transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
    
    # Filter and cluster lines
    left_lane_lines = []  # Initialize left lane lines
    right_lane_lines = []  # Initialize right lane lines
    slope_threshold = 0.4  # Adjust this value to filter out unwanted lines
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero
            
            # Cluster lines based on slope
            if abs(slope) > slope_threshold:
                if slope < 0:
                    left_lane_lines.append(line)
                else:
                    right_lane_lines.append(line)
    
    # Draw the detected lanes on a blank image
    lane_image = np.zeros_like(image)
    if left_lane_lines:
        left_lane = average_lines(left_lane_lines, image)
        draw_lane(lane_image, left_lane)
    
    if right_lane_lines:
        right_lane = average_lines(right_lane_lines, image)
        draw_lane(lane_image, right_lane)
    
    # Combine the lane image with the original image
    result = cv2.addWeighted(image, 1, lane_image, 0.8, 0)
    
    return result

def average_lines(lines, image):
    # Calculate the average slope and intercept of a set of lines
    x1_avg = np.mean([line[0][0] for line in lines])
    y1_avg = np.mean([line[0][1] for line in lines])
    x2_avg = np.mean([line[0][2] for line in lines])
    y2_avg = np.mean([line[0][3] for line in lines])
    
    slope = (y2_avg - y1_avg) / (x2_avg - x1_avg + 1e-6)  # Avoid division by zero
    intercept = y1_avg - slope * x1_avg
    
    y1 = int(image.shape[0])
    y2 = int(image.shape[0] * 0.6)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    return [[x1, y1, x2, y2]]

def draw_lane(image, lane):
    # Draw a lane line on the image
    x1, y1, x2, y2 = lane[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=20)

# Main loop
cap = cv2.VideoCapture(0)  # Replace 0 with the camera index if multiple cameras are connected

while True:
    ret, frame = cap.read()
    
    # Break the loop if no frame is captured
    if not ret:
        break
    
    # Detect lanes in the frame
    result = detect_lanes(frame)
    
    # Display the result
    cv2.imshow("Lane Detection", result)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
