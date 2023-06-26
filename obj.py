import cv2
import numpy as np
import pickle

# Load class names
classNames = []
classFile = 'models/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load model
configPath = 'models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'models/frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Set threshold to detect objects
confThreshold = 0.5

# Function for object detection
def detect_objects(image):
    # Make a copy of the image
    img = np.copy(image)

    # Object detection
    classIds, confs, bbox = net.detect(img, confThreshold)

    # Initialize a list to store detected objects and their bounding boxes
    detected_objects = []
    detected_boxes = []

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Draw bounding box and label on the image
            cv2.rectangle(img, box, color=(255, 0, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Add detected object and bounding box to the lists
            detected_objects.append(classNames[classId - 1])
            detected_boxes.append(box)

    return img, detected_objects, detected_boxes

# Function for lane detection
# Function for lane detection
def detect_lanes(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for white and yellow lanes
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 30, 255], dtype=np.uint8)
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)

    # Create masks to isolate white and yellow lanes
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine the masks
    lane_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(lane_mask, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Define region of interest (ROI) for lane detection
    height, width = edges.shape
    roi_vertices = np.array([[(0, height), (width // 2, height // 2), (width, height)]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Perform Hough line detection
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50)

    # Draw the detected lanes
    lane_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_image, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)

    # Combine lane image with the original image
    result = cv2.addWeighted(image, 1, lane_image, 0.8, 0)

    return result


# Open video capture
# video_path = 'video.mp4'
cap = cv2.VideoCapture(0)

# Check if video capture is successfully opened
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Read the first frame of the video
ret, frame = cap.read()

while ret:
    # Perform object detection on the frame
    result_frame, objects, boxes = detect_objects(frame)

    # Perform lane detection on the frame
    result_frame = detect_lanes(result_frame)

    # Display the result frame with bounding boxes and lanes
    cv2.imshow('Object and Lane Detection', result_frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Read the next frame
    ret, frame = cap.read()

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
