#include <Servo.h>
Servo pusherESC;
Servo directionESC;
int pin = 4;
int minSpeed = 1400;
int maxSpeed = 1900;
int servoPin = 5;

int minAngle = 1000; //60 degree
int midAngle = 1500; // 90 degree
int maxAngle = 1800; // 120 degree
int deadZone = 1500;
int _delay = 10;
int OFFSET = 50;


void writeSpeed();

int currentSpeed = deadZone;
int angleGiven = midAngle;
boolean status = true;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pusherESC.attach(pin,1000,2000);
  directionESC.attach(servoPin);
  directionESC.writeMicroseconds(midAngle);
  pusherESC.writeMicroseconds(currentSpeed);

}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    const char* string = data.c_str();
    char delim[] = "#";
    char* token = strtok(string, delim);
    angleGiven = map(atoi(token),0,60, minAngle, maxAngle);
    while (token != NULL) {
      token = strtok(NULL, delim);
      if (token != NULL) {
        if (atoi(token) > 0) {
          currentSpeed = map(atoi(token), 0,10,1560, 1900);
        } else if (atoi(token) < 0) {
          currentSpeed = map(atoi(token), 0,-10,1400, 1320);
        } else {
          currentSpeed = deadZone;
        }
      }
    }
  }
  pusherESC.writeMicroseconds(currentSpeed);
  directionESC.writeMicroseconds(angleGiven);
  Serial.println();

    // writeSpeed();
    delay(12.5);
}

void writeSpeed() {
  if (currentSpeed == deadZone) {
    currentSpeed = deadZone;
    pusherESC.writeMicroseconds(currentSpeed);
  } else if (currentSpeed > deadZone) {
    if (status == true) {
      if (currentSpeed >= maxSpeed) {
        currentSpeed = maxSpeed;
      }
      pusherESC.writeMicroseconds(currentSpeed);
      delay(_delay);
    } else {
      int tmp = currentSpeed;
      currentSpeed = deadZone;
      pusherESC.writeMicroseconds(currentSpeed);
      delay(20);
      while (currentSpeed <= maxSpeed) {
        if (currentSpeed >= maxSpeed) {
          currentSpeed = maxSpeed;
          break;
        }
        if (currentSpeed >= tmp) {
          currentSpeed = tmp;
          break;
        } 
          pusherESC.writeMicroseconds(currentSpeed);
          currentSpeed += 50;
          delay(_delay);
        
      }
    }
    status = true;
  } else if (currentSpeed < deadZone) {
    if (status == false) {
      if (currentSpeed <= minSpeed) {
        currentSpeed = minSpeed;
      }
      pusherESC.writeMicroseconds(currentSpeed);
      delay(_delay);
    } else {
      int tmp = currentSpeed;
      currentSpeed = deadZone;
      pusherESC.writeMicroseconds(currentSpeed);
      delay(25);
      pusherESC.writeMicroseconds(currentSpeed - OFFSET);
      delay(80);
      pusherESC.writeMicroseconds(currentSpeed);
      delay(80);
      while (currentSpeed >= minSpeed) {
        if (currentSpeed <= minSpeed) {
          currentSpeed = minSpeed;
          break;
        }
        if (currentSpeed <= tmp) {
          currentSpeed = tmp;
          break;
        }
        pusherESC.writeMicroseconds(currentSpeed);
        delay(_delay);
        currentSpeed -= 50;
      }
    }
    status = false;
  }
}