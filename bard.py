import cv2
import numpy as np

# Load the YOLOv5 model
model = cv2.dnn.readNet("models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt", "models/frozen_inference_graph.pb")

# Load the COCO object classes
classes = []
with open("models/coco.names", "r") as f:
    classes = f.read().splitlines()

# Create a video capture object
cap = cv2.VideoCapture("video.mp4")

while True:
    # Capture the frame
    ret, frame = cap.read()

    # Convert the frame to a blob
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)

    # Pass the blob through the network
    model.setInput(blob)
    out = model.forward()

    # Get the bounding boxes and confidence scores
    boxes = out[0]
    scores = out[1]

    # Filter out the bounding boxes with low confidence scores
    keep = np.where(scores > 0.5)[0]
    boxes = boxes[keep]
    scores = scores[keep]

    # Draw the bounding boxes on the frame
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(round(scores[0], 2)) + " " + classes[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Quit if the user presses ESC
    if cv2.waitKey(1) == 27:
        break

# Release the video capture object
cap.release()

# Close all windows
cv2.destroyAllWindows()
