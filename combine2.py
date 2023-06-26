import sys
import cv2
import numpy as np
import pickle
import serial
import pygame
from trackerr import tracker
from moviepy.editor import VideoFileClip

pygame.init()
screen_width, screen_height = 640, 480
screen = pygame.display.set_mode((screen_width, screen_height))
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

# Functions for lane detection
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    gradmag = np.sqrt(sobelx**2+sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output
    
def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        absgraddir = np.absolute(np.arctan(sobely/sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

def color_threshold(image, sthresh=(0,255), vthresh=(0,255), lthresh=(0,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1
    return output
    
def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output

def draw_thumbnails(img_cp, img, window_list, thumb_w=100, thumb_h=80, off_x=30, off_y=30):

    for i, bbox in enumerate(window_list):
        thumbnail = img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
        vehicle_thumb = cv2.resize(thumbnail, dsize=(thumb_w, thumb_h))
        start_x = 300 + (i+1) * off_x + i * thumb_w
        img_cp[off_y + 30:off_y + thumb_h + 30, start_x:start_x + thumb_w, :] = vehicle_thumb

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

# Create a Pygame surface for object detection display
detection_surface = pygame.Surface((screen_width // 2, screen_height))

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
    # cv2.imshow("Object Detection", frame)

    # Perform lane detection on the frame
    undistorted_img = cv2.undistort(frame, mtx, dist, None, mtx)
    processed_frame = process_frame(undistorted_img)
    processed_frames.append(processed_frame)

    # Display the processed frame
    # cv2.imshow("Lane Detection", processed_frame)

    # Combine the controller and object detection screens
    resized_frame = cv2.resize(frame, (screen_width // 2, screen_height))
    resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(resized_frame_rgb)
    screen.fill(bg_color)
    screen.blit(frame_surface, (0, 0))
    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)
    processed_frame_surface = pygame.surfarray.make_surface(processed_frame_rgb)
    detection_surface.fill(bg_color)
    detection_surface.blit(processed_frame_surface, (0, 0))
    # Display the screen
    pygame.display.flip()

    # Check if the exit key is pressed
    if cv2.waitKey(1) == 27:
        exit_key_pressed = True

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
