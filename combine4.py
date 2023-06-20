import sys
import cv2
import numpy as np
import pickle
import serial
import pygame
from tracker import tracker
from moviepy.editor import VideoFileClip

pygame.init()
screen_width, screen_height = 640, 480
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.key.set_repeat()

ser = serial.Serial('COM19', 9600, timeout=1)
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


def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

def process_image(img):

    #set global variables
    global object_too_close
    global warning_displayed

    object_too_close = False

    # Make a copy of the input image
    img = np.copy(img)
    img_copy = np.copy(img)
    
    # Lane detection
    # Undistort the input image using calibration parameters
    img = cv2.undistort(img, mtx, dist, None, mtx)


    # Apply image preprocessing techniques to detect lanes
    preprocessImage = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img, orient='x', thresh=(12,255))
    grady = abs_sobel_thresh(img, orient='x', thresh=(25,255))
    c_binary = color_threshold(img, sthresh=(100,255),vthresh=(50,255))
    preprocessImage[((gradx==1)&(grady==1)|(c_binary==1))] = 255


    # Define source and destination points for perspective transformation
    img_size = (img.shape[1], img.shape[0])
    bot_width = .75 #changed from .76
    mid_width = .1 #changed this value - seemed to work a lot better than 0.08 
    height_pct = .62
    bottom_trim = .935
    src = np.float32([[img.shape[1]*(.5-mid_width/2),img.shape[0]*height_pct],[img.shape[1]*(.5+mid_width/2),img.shape[0]*height_pct],
                      [img.shape[1]*(.5+bot_width/2),img.shape[0]*bottom_trim],[img.shape[1]*(.5-bot_width/2),img.shape[0]*bottom_trim]])
    offset = img_size[0]*.25
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]],[offset, img_size[1]]])

    # Perform perspective transform on the preprocessed image
    M = cv2.getPerspectiveTransform(src, dst) 
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(preprocessImage, M, img_size,flags=cv2.INTER_LINEAR)


    # Lane line detection
    # Find the lane lines using a sliding window method
    window_width = 25
    window_height = 80
    curve_centers = tracker(Mywindow_width=window_width,Mywindow_height=window_height,Mymargin=25,My_ym=10/720,My_xm=4/384,Mysmooth_factor=15)
    window_centroids = curve_centers.find_window_centroids(warped)

    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    rightx = []
    leftx = []

    for level in range(0,len(window_centroids)):
        l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
        r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)

        leftx.append(window_centroids[level][0]) 
        rightx.append(window_centroids[level][1])

        l_points[(l_points==255)|((l_mask==1))] = 255
        r_points[(r_points==255)|((r_mask==1))] = 255

    template = np.array(r_points+l_points,np.uint8)
    zero_channel = np.zeros_like(template)
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8)
    warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8)
    result = cv2.addWeighted(warpage,1,template,0.5,0.0)

    yvals = range(0,warped.shape[0])
    res_yvals = np.arange(warped.shape[0]-(window_height/2),0,-window_height)

    left_fit = np.polyfit(res_yvals,leftx,2)
    left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
    left_fitx = np.array(left_fitx,np.int32)

    right_fit = np.polyfit(res_yvals,rightx,2)
    right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
    right_fitx = np.array(right_fitx,np.int32)

    left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2,left_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)

    right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)

    inner_lane = np.array(list(zip(np.concatenate((left_fitx+window_width/2,right_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)

    road = np.zeros_like(img)
    road_bkg = np.zeros_like(img)
    cv2.fillPoly(road,[left_lane],color=[255,0,0])
    # cv2.fillPoly(road,[inner_lane],color=[0,255,0])
    if object_too_close:
        cv2.fillPoly(road, [inner_lane], color=[0,0,255])  # Change inner lane color to red
    else:
        cv2.fillPoly(road, [inner_lane], color=[0, 255, 0])
    cv2.fillPoly(road,[right_lane],color=[0,0,255])
    cv2.fillPoly(road_bkg,[left_lane],color=[255,255,255])
    cv2.fillPoly(road_bkg,[right_lane],color=[255,255,255])

    road_warped = cv2.warpPerspective(road,Minv,img_size,flags=cv2.INTER_LINEAR)
    road_warped_bkg = cv2.warpPerspective(road_bkg,Minv,img_size,flags=cv2.INTER_LINEAR)

    base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
    result = cv2.addWeighted(base,1.0,road_warped,0.7,0.0)

    #measure pixels in y and x directions
    ym_per_pix = curve_centers.ym_per_pix
    xm_per_pix = curve_centers.xm_per_pix

    curve_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix,np.array(leftx,np.float32)*xm_per_pix,2)
    curverad = ((1+(2*curve_fit_cr[0]*yvals[-1]*ym_per_pix+curve_fit_cr[1])**2)**1.5)/np.absolute(2*curve_fit_cr[0]) #remember that it's the equation from the lesson (derivatives) - radius of curvature

    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'
    if center_diff > 0.2:
        turn_direction = 'Turn Right'
    elif center_diff < -0.2:
        turn_direction = 'Turn Left'
    else:
        turn_direction = 'Straight'
        

    # Object detection
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

     # Initialize a list to store detected objects and their distances
    detected_objects = []
    detected_object_imgs = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Calculate distance of object from camera (in meters)
            distance = round((2 * focalLength) / box[2], 2)
            cv2.rectangle(img, box, color=(255,255,255), thickness=3)
            object_name = classNames[classId - 1].upper()
            detected_objects.append(f"{object_name}: {distance}m")
            # Crop the detected object image
            x, y, w, h = box
            cropped_img = img_copy[y:y+h, x:x+w]
            detected_object_imgs.append(cropped_img)

            # Check if the detected object is inside the lane
            # Here, I'm assuming that the coordinates of the lane (left_fitx and right_fitx) can be used to determine if an object is in the lane.
            if (x > left_fitx[-1]) and (x + w < right_fitx[-1]): 
                object_too_close = True

            
            # Determine label background size based on box size
            label_size, base_line = cv2.getTextSize(object_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_w = max(label_size[0], w)
            label_h = label_size[1] + 10
            
            # Draw object name above detection box with background color
            label_ymin = max(y - label_h, 0)
            label_color = (0, 0, 0)  
            cv2.rectangle(img, (x, label_ymin), (x + label_w, y), label_color, cv2.FILLED)
            cv2.putText(img, object_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            cv2.putText(img, str(distance) + " m", (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


            # Check if the object is too close
            if distance < object_distance_threshold:
                object_too_close = True  

                #changes the color of the lane to red if an object is too close
                cv2.fillPoly(road, [inner_lane], color=[0, 0, 255])  # Change inner lane color to red
            else:
                cv2.fillPoly(road, [inner_lane], color=[0, 255, 0])

            # Draw lane overlays
            road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)
            road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size, flags=cv2.INTER_LINEAR)

            base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
            result = cv2.addWeighted(base, 1.0, road_warped, 0.7, 0.0)

    if object_too_close:
        # Display warning text when an object is too close
        font = cv2.FONT_HERSHEY_SIMPLEX
        warning_text = "WARNING: Object too close!"
        text_position = (50, 200)
        font_scale = 0.5
        font_thickness = 2
        warning_color = (0, 0, 255)

        # Change the inner_lane color to red if an object is too close
        cv2.fillPoly(road, [inner_lane], color=[0, 0, 255])

        cv2.putText(result, warning_text, text_position, font, font_scale, warning_color, font_thickness)
    else:
        warning_text = "No objects detected-Keep going!"
        cv2.fillPoly(road, [inner_lane], color=[0, 255, 0])


    # Get the width of the image
    img_width = result.shape[1]
    
    # Draw a filled rectangle as the background
    cv2.rectangle(result, (0, 0), (img_width, 150), bg_color, -1)

    # Add the text on top of the background
    # cv2.putText(result, 'Lane Status', (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    # #radius of curvature
    # cv2.putText(result, 'Radius of curvature = '+str(round(curverad,3))+'(m)', (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # #vehicle position
    # cv2.putText(result, 'Vehicle Position: '+str(abs(round(center_diff,3)))+'m '+side_pos+' of center', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # # Assistance
    # cv2.putText(result, 'Direction Assistance:'+ ' '+turn_direction, (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Draw a vertical line to separate text
    line_height = result.shape[0] // 5  # set the line height to half of the image height
    cv2.line(result, (360, 0), (360, line_height), (0, 0, 0), thickness=1)


    # detected objects
    cv2.putText(result, 'Detected Objects', (830, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Display detected objects names and their distances
    for i, detected_object in enumerate(detected_objects):
        cv2.putText(result, detected_object, (380, 20 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


    # Display detected object images at the top
    spacing = 20
    current_x = 500
    max_width = img_width - spacing

    for obj_img in detected_object_imgs:
        h, w, _ = obj_img.shape
        scale = 100 / h
        resized_width = int(w * scale)
        resized_obj_img = cv2.resize(obj_img, (resized_width, 100))

        if current_x + resized_width + spacing > max_width:
            # If there's not enough space to display the resized object image, break the loop
            break

        result[25:125, current_x:current_x + resized_width] = resized_obj_img
        current_x += resized_width + spacing

    final_result = cv2.addWeighted(result, 1, img, 0.5, 0)
    return final_result


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
    
    # # Rotate the frame
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

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
    resized_frame = cv2.resize(frame, (screen_height, screen_width // 2))
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
