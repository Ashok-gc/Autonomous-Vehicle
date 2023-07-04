import cv2

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Set the video capture properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize the colors for cone detection
green_lower = (29, 86, 6)
green_upper = (64, 255, 255)
orange_lower = (0, 50, 80)
orange_upper = (20, 255, 255)

while True:
    # Read each frame from the video capture
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the frame to detect green cones
    green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)

    # Threshold the frame to detect orange cones
    orange_mask = cv2.inRange(hsv_frame, orange_lower, orange_upper)

    # Find contours in the green mask
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find contours in the orange mask
    orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process the detected green cones
    for green_contour in green_contours:
        # Calculate the bounding box of each green cone
        x, y, w, h = cv2.boundingRect(green_contour)

        # Draw a green box around the green cone
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Perform action based on the position of the green cone
        if x + w/2 < 320:
            print("Move left")
        else:
            print("Move right")

    # Process the detected orange cones
    for orange_contour in orange_contours:
        # Calculate the bounding box of each orange cone
        x, y, w, h = cv2.boundingRect(orange_contour)

        # Draw an orange box around the orange cone
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)

        # Perform action based on the position of the orange cone
        if x + w/2 < 320:
            print("Move left")
        else:
            print("Move right")

    # Display the frame with cone detections
    cv2.imshow("Cone Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy any remaining windows
cap.release()
cv2.destroyAllWindows()
