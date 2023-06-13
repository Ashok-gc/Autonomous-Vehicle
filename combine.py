import sys
import cv2
import numpy as np
import pickle
import serial
import pygame
from tracker import tracker
from moviepy.editor import VideoFileClip

pygame.init()
screen = pygame.display.set_mode((640, 480))

pygame.key.set_repeat()

ser = serial.Serial('COM20', 9600, timeout=1)
ser.flush()

SPEED = 0
DIRECTION = 30

# import warnings

object_inside_box = False
object_too_close = False
warning_displayed = False

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
thres = 0.45

# Focal length of camera (in mm)
focalLength = 10

# Define the background color
bg_color = (117, 117, 117)

# line color
line_color = (0, 0, 255)

# Set threshold distance for objects
object_distance_threshold = 0.6  # meters

# Lane detection
dist_pickle = pickle.load(open("models/calibration_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
processed_frames = []

def writeArduino(d, s):
    ACTION = (str(d) + "#" + str(s) + "\n").encode('utf-8')
    ser.write(ACTION)
    line = ser.readline().decode('utf-8').rstrip()

def moveLeft():
    DIRECTION = 0

def moveRight():
    DIRECTION = 60

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray
f_Gear = ["First Gear", "Second Gear", "Third Gear", "Fourth Gear", "Fifth Gear",
          "Sixth Gear", "Seventh Gear", "Eighth Gear", "Ninth Gear", "Tenth Gear"]


def frontGear(SPEED):
    if SPEED >= 1 and SPEED < 2:
        print(f_Gear[0])
    elif SPEED >= 2 and SPEED < 3:
        print(f_Gear[1])
    elif SPEED >= 3 and SPEED < 4:
        print(f_Gear[2])
    elif SPEED >= 4 and SPEED < 5:
        print(f_Gear[3])
    elif SPEED >= 5 and SPEED < 6:
        print(f_Gear[4])
    elif SPEED >= 6 and SPEED < 7:
        print(f_Gear[5])
    elif SPEED >= 7 and SPEED < 8:
        print(f_Gear[6])
    elif SPEED >= 8 and SPEED < 9:
        print(f_Gear[7])
    elif SPEED >= 9 and SPEED < 10:
        print(f_Gear[8])
    elif SPEED >= 10:
        print(f_Gear[9])

# Set the flag for the exit key
exit_key_pressed = False

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if the video capture object was successfully initialized
if not cap.isOpened():
    print('Error opening video file')
    sys.exit()

_event = "STOP"
while not exit_key_pressed:
    # Check for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()  # Quit Pygame
            exit_key_pressed = True
            break
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                exit_key_pressed = True
                break
            if event.key == ord('w'):
                _event = "FORWARD"
                SPEED = 1
            elif event.key == ord('s'):
                _event = "BACKWARD"
                SPEED = -1
            if event.key == ord('a'):
                DIRECTION = 60
            elif event.key == ord('d'):
                DIRECTION = 0
        if event.type == pygame.KEYUP:
            if event.key == ord('w') or event.key == ord('s'):
                _event = "STOP"
            if event.key == ord('a') or event.key == ord('d'):
                DIRECTION = 30
    if(_event == "FORWARD"):
        if(SPEED < 10):
            SPEED = SPEED + .02
            frontGear(SPEED)
    elif(_event == "BACKWARD"):
        if(SPEED > -10):
            SPEED = SPEED - .02
    elif(_event == "STOP"):
        if(SPEED > 0):
            SPEED = SPEED - .1
        elif(SPEED < 0):
            SPEED = SPEED + .1
    writeArduino(DIRECTION, SPEED)
    
    # Get a frame from the camera
    ret, frame = cap.read()

    # Check if a frame was successfully read
    if not ret:
        print("Error reading frame")
        break

    # Resize the frame
    frame = cv2.resize(frame, (640, 480))

    # Detect objects in the frame
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)

    # Draw bounding boxes and labels for detected objects
    if len(classIds) > 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX,
                        1, (0, 255, 0), 2)
            object_inside_box = True

            # Calculate the distance to the object
            object_width = box[2] - box[0]
            distance = (focalLength * 16) / object_width

            # Check if the object is too close
            if distance < object_distance_threshold:
                object_too_close = True

    # Display the frame with objects
    cv2.imshow("Object Detection", frame)

    # Perform lane detection on the frame
    undistorted_img = cv2.undistort(frame, mtx, dist, None, mtx)
    processed_frame = process_frame(undistorted_img)
    processed_frames.append(processed_frame)

    # Display the processed frame with lane detection
    cv2.imshow("Lane Detection", processed_frame)

    # Check for threat conditions
    if object_inside_box:
        if object_too_close:
            if not warning_displayed:
                print("Warning: Object too close!")
                warning_displayed = True
        else:
            warning_displayed = False
    else:
        warning_displayed = False

    # Update the display
    pygame.display.flip()

    # Check for the exit key again in case it was pressed during the loop
    if pygame.key.get_pressed()[pygame.K_ESCAPE]:
        exit_key_pressed = True

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()
