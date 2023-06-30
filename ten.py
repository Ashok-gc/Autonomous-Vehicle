import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines, color=(0, 255, 0), thickness=3):
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def lane_detection(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Define region of interest
    height, width = edges.shape
    roi_vertices = [(0, height), (width / 2, height / 2), (width, height)]
    roi_edges = region_of_interest(edges, np.array([roi_vertices], np.int32))

    # Perform Hough line detection
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=100)

    # Draw detected lines on original image
    line_image = np.zeros_like(image)
    draw_lines(line_image, lines)

    # Combine line image with original image
    lane_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    return lane_image

# Open video capture
cap = cv2.VideoCapture('lane.mp4')

while True:
    # Read frame from video
    ret, frame = cap.read()

    # Break loop if video is finished
    if not ret:
        break

    # Perform lane detection on current frame
    output_frame = lane_detection(frame)

    # Display output frame
    cv2.imshow('Lane Detection', output_frame)

    # Check for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
