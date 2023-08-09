import serial
import pygame

# Replace 'COMx' with the actual port your Arduino is connected to.
# The baud rate (9600) should match the baud rate set in the Arduino code.
ser = serial.Serial('COM8', 9600)

def send_command(command):
    ser.write(command.encode())

def main():
    pygame.init()
    screen = pygame.display.set_mode((200, 200))
    pygame.display.set_caption("RC Car Control")

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    ser.close()
                    pygame.quit()
                    return

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:  # Forward
                send_command("w\n")
            elif keys[pygame.K_s]:  # Backward
                send_command("s\n")
            elif keys[pygame.K_a]:  # Left
                send_command("a\n")
            elif keys[pygame.K_d]:  # Right
                send_command("d\n")
            elif keys[pygame.K_q]:  # Stop
                send_command("q\n")
    except KeyboardInterrupt:
        ser.close()
        pygame.quit()

if __name__ == "__main__":
    main()
