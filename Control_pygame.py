import cv2  # Import the OpenCV library
import time  # Import the time library
import serial  # Import the Serial library
import pygame  # Import the Pygame library

pygame.init()  # Initialize Pygame

pygame.display.set_mode(size=(640, 480))  # Set the display window size

pygame.key.set_repeat()  # Enable key repeat for Pygame

ser = serial.Serial('COM9', 9600, timeout=1)  # Open a serial connection with the Arduino
ser.flush()  # Flush the serial buffer

SPEED = 0  # Initialize the speed variable
DIRECTION = 30  # Initialize the direction variable

def writeArduiono(d, s):
    """
    Function to write the direction and speed values to the Arduino

    Args:
        d (int): the direction value
        s (float): the speed value
    """
    ACTION = (str(d) + "#" + str(s) + "\n").encode('utf-8')  # Create a string to send to the Arduino
    ser.write(ACTION)  # Write the string to the Arduino
    line = ser.readline().decode('utf-8').rstrip()  # Read the response from the Arduino

def moveLeft():
    """
    Function to set the direction to left
    """
    DIRECTION = 0

def moveRight():
    """
    Function to set the direction to right
    """
    DIRECTION = 60