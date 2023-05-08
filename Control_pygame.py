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

#  main
_event = "STOP"  # Initialize the event variable to stop
f_Gear = ["First Gear", "Second Gear", "Third Gear", "Fourth Gear", "Fifth Gear",
          "Sixth Gear", "Seventh Gear", "Eighth Gear", "Ninth Gear", "Tenth Gear"]  # List of gear names

def frontGear(SPEED):
    """
    Function to determine the front gear based on the speed value

    Args:
        SPEED (float): the speed value
    """
    if SPEED >= 1 and SPEED < 2:
        print(f_Gear[0])  # Print the gear name for the current speed range
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
