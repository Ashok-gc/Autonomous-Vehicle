import cv2
import numpy as np

def draw_center_line(frame, center_x, height):
    line_color = (255, 255, 255)  # BGR color for the line (blue in this case)
    line_thickness = 50
    cv2.line(frame, (center_x, 0), (center_x, height), line_color, line_thickness)

# Create a VideoCapture object to capture video from the camera
cap = cv2.VideoCapture(2)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Get the frame dimensions
    height, width, _ = frame.shape

    # Calculate the center x-coordinate
    center_x = width // 2

    # Draw a vertical line in the center
    draw_center_line(frame, center_x, height)

    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper boundaries for the left (green) and right (orange) cones in HSV color space
    green_lower = (40, 50, 50)
    green_upper = (80, 255, 255)

    orange_lower = (0, 50, 50)
    orange_upper = (20, 255, 255)

    # Threshold the HSV frame to get regions of left and right cones
    green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)
    orange_mask = cv2.inRange(hsv_frame, orange_lower, orange_upper)

    # Perform morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)

    # Find contours of left and right cones
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to remove small objects
    min_contour_area = 100  # Adjust this value according to your requirements

    # Draw bounding boxes around the larger size cone (green)
    green_cones = [contour for contour in green_contours if cv2.contourArea(contour) > min_contour_area]
    green_cones = sorted(green_cones, key=lambda x: cv2.contourArea(x), reverse=True)[:1]  # Select the largest cone

    for contour in green_cones:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw bounding boxes around the larger size cone (orange)
    orange_cones = [contour for contour in orange_contours if cv2.contourArea(contour) > min_contour_area]
    orange_cones = sorted(orange_cones, key=lambda x: cv2.contourArea(x), reverse=True)[:1]  # Select the largest cone

    for contour in orange_cones:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)

    # Calculate the center point between the detected cones
    center_x_cones = None

    if green_cones and orange_cones:
        green_x, _, green_w, _ = cv2.boundingRect(green_cones[0])
        orange_x, _, orange_w, _ = cv2.boundingRect(orange_cones[0])

        green_center_x = green_x + green_w // 2
        orange_center_x = orange_x + orange_w // 2

        center_x_cones = (green_center_x + orange_center_x) // 2

    # Draw a vertical central line
    line_color = (255, 0, 0)  # BGR color for the line (blue in this case)
    line_thickness = 10

    if center_x_cones is not None:
        draw_center_line(frame, center_x_cones, frame.shape[0])

        # Print instructions based on the position of the line of the center of the cones relative to the draw_center_line
        if center_x_cones < center_x - line_thickness:
            print("Go right")
        elif center_x_cones > center_x + line_thickness:
            print("Go left")
        elif center_x - line_thickness <= center_x_cones <= center_x + line_thickness:
            print("Good to go")
        else:
            print("No autopilot")
    else:
        print("No autopilot")

    # Display the frame
    cv2.imshow("Cone Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close the windows
cap.release()
cv2.destroyAllWindows()
