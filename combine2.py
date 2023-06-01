import sys
import cv2
import numpy as np
import pickle
from tracker import tracker
from moviepy.editor import VideoFileClip

# Object detection
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

# Line color
line_color = (0, 0, 255)

# Set threshold distance for objects
object_distance_threshold = 0.6  # meters

# Initialize a flag to check if any object is too close
object_too_close = False

# Lane detection
dist_pickle = pickle.load(open("models/calibration_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
processed_frames = []


# Functions for lane detection
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)

    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)

    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output


def corners_unwarp(img, mtx, dist):
    img_size = (img.shape[1], img.shape[0])

    # Choose four source points (the four corners of a trapezoid in the image)
    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
         [((img_size[0] / 6) - 10), img_size[1]],
         [(img_size[0] * 5 / 6) + 60, img_size[1]],
         [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])

    # Define four destination points (which will be warped to a rectangle)
    dst = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0]])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)

    return warped


def process_frame(img):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    combined_binary = np.zeros_like(undist[:, :, 0])
    gradx = abs_sobel_thresh(undist, orient='x', sobel_kernel=3, thresh=(20, 100))
    grady = abs_sobel_thresh(undist, orient='y', sobel_kernel=3, thresh=(20, 100))
    mag_binary = mag_thresh(undist, sobel_kernel=3, mag_thresh=(30, 100))
    dir_binary = dir_threshold(undist, sobel_kernel=15, thresh=(0.7, 1.3))

    combined_binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    warped = corners_unwarp(combined_binary, mtx, dist)
    processed_frames.append(warped)

    # Object detection
    # Detect objects in the frame
    classIds, confs, bbox = net.detect(undist, confThreshold=thres)

    # Check if any objects are detected
    if len(classIds) > 0:
        object_too_close = False

        # Loop over detected objects
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className != 'person':
                continue

            # Calculate distance to the object
            object_width = box[2] - box[0]
            distance = (2 * focalLength) / object_width

            # Check if the object is too close
            if distance < object_distance_threshold:
                object_too_close = True
                break

    # Lane detection
    height, width = warped.shape

    # Define source and destination points for perspective transform
    src = np.float32([[560, 460], [720, 460], [250, 680], [1050, 680]])
    dst = np.float32([[0, 0], [1280, 0], [0, 720], [1280, 720]])

    # Perform perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(warped, M, (width, height), flags=cv2.INTER_LINEAR)

    # Find lane pixels and fit polynomial
    left_fit, right_fit, out_img = tracker(warped)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Generate x and y values for plotting
    ploty = np.linspace(0, height - 1, height)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = cv2.getPerspectiveTransform(dst, src)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # Add text indicating if an object is too close
    if object_too_close:
        cv2.putText(result, 'Object Too Close!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return result


# Process video frames
output_file = 'output.mp4'
clip1 = VideoFileClip('input.mp4')
processed_clip = clip1.fl_image(process_frame)
processed_clip.write_videofile(output_file, audio=False)
