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

# Open video capture
video_path = 'challenge_video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video capture is successfully opened
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Read the first frame of the video
ret, frame = cap.read()

while ret:
    # Perform object detection on the frame
    result_frame, objects, boxes = detect_objects(frame)

    # Display the result frame with bounding boxes and labels
    cv2.imshow('Object Detection', result_frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Read the next frame
    ret, frame = cap.read()

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
