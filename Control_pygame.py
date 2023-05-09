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

# Set the flag for the exit key
exit_key_pressed = False

while not exit_key_pressed:  # Loop until the exit key is pressed
    for event in pygame.event.get():  # Iterate over the events in the Pygame event queue
        if event.type == pygame.QUIT:  # Check if the user has requested to quit the game window
            pygame.quit()  # Quit Pygame
            exit_key_pressed = True  # Set the exit flag to True
            break  # Break out of the for loop
        if event.type == pygame.KEYDOWN:  # Check if a key has been pressed
            if event.key == pygame.K_ESCAPE:  # Check if the escape key has been pressed
                exit_key_pressed = True  # Set the exit flag to True
                break  # Break out of the for loop
            if event.key == ord('w'):  # Check if the 'w' key has been pressed
                _event = "FORWARD"  # Set the event to move forward
                SPEED = 1  # Set the speed to 1
            elif event.key == ord('s'):  # Check if the 's' key has been pressed
                _event = "BACKWARD"  # Set the event to move backward
                SPEED = -1  # Set the speed to -1
            # elif event.key == ord('s'):
            #     _event = "BACKWARD"
            #     if SPEED > -10:
            #         SPEED -= 0.1
            if event.key == ord('a'):  # Check if the 'a' key has been pressed
                DIRECTION = 60  # Set the direction to left
            elif event.key == ord('d'):  # Check if the 'd' key has been pressed
                DIRECTION = 0  # Set the direction to right
        if event.type == pygame.KEYUP:  # Check if a key has been released
            if event.key == ord('w') or event.key == ord('s'):  # Check if the 'w' or 's' key has been released
                _event = "STOP"  # Set the event to stop
            if event.key == ord('a') or event.key == ord('d'):  # Check if the 'a' or 'd' key has been released
                DIRECTION = 30  # Set the direction to center
    if(_event == "FORWARD"):  # Check if the event is to move forward
        if(SPEED < 10):  # Check if the speed is less than 10
            SPEED = SPEED + .02  # Increase the speed by 0.02
            frontGear(SPEED)  # Determine and print the gear based on the speed
    elif(_event == "BACKWARD"):  # Check if the event is to move backward
        if(SPEED > -10):  # Check if the speed is greater than -10
            SPEED = SPEED - .02  # Decrease the speed by 0.02
    elif(_event == "STOP"):  # Check if the event is to stop
        if(SPEED > 0):  # Check if the speed is greater than 0
            SPEED = SPEED - .1  # Decrease the speed by 0.1
        elif(SPEED < 0):  # Check if the speed is less than 0
            SPEED = SPEED + .1  # Increase the speed by 0.1
    writeArduiono(DIRECTION, SPEED)  # Write the direction and speed values to the Arduino
pygame.quit() 
