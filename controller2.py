import pygame
import serial

# Arduino serial port settings
SERIAL_PORT = 'COM20'  # Replace with the appropriate serial port
BAUD_RATE = 9600

# Pygame key mappings
KEY_FORWARD = pygame.K_UP
KEY_BACKWARD = pygame.K_DOWN
KEY_LEFT = pygame.K_LEFT
KEY_RIGHT = pygame.K_RIGHT
KEY_STOP = pygame.K_SPACE
KEY_QUIT = pygame.K_ESCAPE

# Arduino command values
COMMAND_FORWARD = '+10\n'
COMMAND_BACKWARD = '-10\n'
COMMAND_LEFT = '30\n'
COMMAND_RIGHT = '90\n'
COMMAND_STOP = '0\n'

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((400, 300))

# Initialize serial communication with Arduino
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == KEY_FORWARD:
                ser.write(COMMAND_FORWARD.encode())
            elif event.key == KEY_BACKWARD:
                ser.write(COMMAND_BACKWARD.encode())
            elif event.key == KEY_LEFT:
                ser.write(COMMAND_LEFT.encode())
            elif event.key == KEY_RIGHT:
                ser.write(COMMAND_RIGHT.encode())
            elif event.key == KEY_STOP:
                ser.write(COMMAND_STOP.encode())
            elif event.key == KEY_QUIT:
                running = False

ser.close()
pygame.quit()
