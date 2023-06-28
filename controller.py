import cv2  
import time  
import serial  
import pygame 
pygame.init()  

pygame.display.set_mode(size=(640, 480)) 
pygame.key.set_repeat()  

ser = serial.Serial('COM19', 9600, timeout=1) 
ser.flush()  

SPEED = 0  
DIRECTION = 30  

def writeArduiono(d, s):
    ACTION = (str(d) + "#" + str(s) + "\n").encode('utf-8')  
    ser.write(ACTION)  
    line = ser.readline().decode('utf-8').rstrip() 

def moveLeft():
    DIRECTION = 0

def moveRight():
    DIRECTION = 60

#  main
_event = "STOP" 
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

while not exit_key_pressed:  
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
                DIRECTION = 0 
            elif event.key == ord('d'): 
                DIRECTION = 60  
        if event.type == pygame.KEYUP:  
            if event.key == ord('s') or event.key == ord('w'): 
                _event = "STOP"  
            if event.key == ord('d') or event.key == ord('a'):  
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
    writeArduiono(DIRECTION, SPEED)  
pygame.quit()