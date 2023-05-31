import cv2
import numpy as np
import pygame
from pygame.locals import *

# Load class names and model
classNames = []
with open('models/coco.names', 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

net = cv2.dnn.readNetFromTensorflow('models/frozen_inference_graph.pb', 'models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Set threshold to detect objects
confThreshold = 0.45

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption('Object Detection')
clock = pygame.time.Clock()

def process_events():
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

def display_text(text, position):
    font = pygame.font.SysFont(None, 20)
    text_surface = font.render(text, True, (255, 255, 255))
    screen.blit(text_surface, position)

def main():
    cap = cv2.VideoCapture(0)

    while True:
        process_events()

        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        blob = cv2.dnn.blobFromImage(frame_rgb, size=(300, 300), swapRB=True)
        net.setInput(blob)
        detections = net.forward()

        h, w, _ = frame.shape

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confThreshold:
                classId = int(detections[0, 0, i, 1])
                className = classNames[classId - 1]
                x = int(detections[0, 0, i, 3] * w)
                y = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)

                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), thickness=2)
                cv2.putText(frame, className, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)

                object_name = f'{className}: {confidence:.2f}'
                display_text(object_name, (10, 10 + i * 20))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(frame, (0, 0))

        pygame.display.flip()
        clock.tick(30)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
