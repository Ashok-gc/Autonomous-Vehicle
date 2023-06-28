import cv2
import numpy as np
import pyrealsense2 as rs

# Function to detect and draw lanes
def detect_lanes(frame, left_color=(255, 0, 0), right_color=(128, 0, 128)):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Define a region of interest (ROI) mask
    height, width = edges.shape
    mask = np.zeros_like(edges)
    roi_vertices = [(0, height), (width // 2, height // 2), (width, height)]
    cv2.fillPoly(mask, np.array([roi_vertices], dtype=np.int32), 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Apply Hough transform to detect lines in the ROI
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)
    
    # Separate left and right lane lines
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Fit a linear equation (y = mx + b) to each line
            # Calculate slope and intercept
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                
                # Classify lines based on slope
                if slope < -0.5:
                    left_lines.append((slope, intercept))
                elif slope > 0.5:
                    right_lines.append((slope, intercept))
    
    # Calculate average slope and intercept for left and right lanes
    left_lane = np.mean(left_lines, axis=0) if len(left_lines) > 0 else None
    right_lane = np.mean(right_lines, axis=0) if len(right_lines) > 0 else None
    
    # Calculate lane endpoints based on the image dimensions
    y1 = height
    y2 = int(height * 0.6)
    left_x1 = int((y1 - left_lane[1]) / left_lane[0]) if left_lane is not None and left_lane[0] != 0 else 0
    left_x2 = int((y2 - left_lane[1]) / left_lane[0]) if left_lane is not None and left_lane[0] != 0 else 0
    right_x1 = int((y1 - right_lane[1]) / right_lane[0]) if right_lane is not None and right_lane[0] != 0 else 0
    right_x2 = int((y2 - right_lane[1]) / right_lane[0]) if right_lane is not None and right_lane[0] != 0 else 0
    
    # Create an empty image to draw the lanes and fill the area between them
    lane_image = np.zeros_like(frame)
    
    # Draw left and right lanes
    # Draw left and right lanes
    if left_lane is not None and left_lane[0] != 0:
        cv2.line(lane_image, (left_x1, y1), (left_x2, y2), left_color, thickness=20)
    if right_lane is not None and right_lane[0] != 0:
        cv2.line(lane_image, (right_x1, y1), (right_x2, y2), right_color, thickness=20)

    # Fill the area between the lanes with green
    if left_lane is not None and left_lane[0] != 0 and right_lane is not None and right_lane[0] != 0:
        vertices = np.array([[(left_x1, y1), (left_x2, y2), (right_x2, y2), (right_x1, y1)]], dtype=np.int32)
        cv2.fillPoly(lane_image, vertices, (0, 255, 0))
    
    # Overlay the lane image on the original frame
    result = cv2.addWeighted(frame, 1, lane_image, 0.5, 0)
    
    return result

# Initialize RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        # Wait for a new frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        # Convert RealSense frame to OpenCV format
        frame = np.asanyarray(color_frame.get_data())
        
        # Detect and draw lanes
        # Detect and draw lanes
        result = detect_lanes(frame, left_color=(255, 0, 0), right_color=(128, 0, 128))

        
        # Display the result
        cv2.imshow("Autonomous RC Car", result)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline and close all windows
    pipeline.stop()
    cv2.destroyAllWindows()
