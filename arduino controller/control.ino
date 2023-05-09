#include <Servo.h>   // include Servo library
    Servo pusherESC; // create Servo object for pusherESC
Servo directionESC;  // create Servo object for directionESC
int pin = 4;         // pin number for pusherESC
int minSpeed = 1400; // minimum speed for pusherESC
int maxSpeed = 1900; // maximum speed for pusherESC
int servoPin = 5;    // pin number for directionESC

int minAngle = 1000; // minimum angle for directionESC
int midAngle = 1500; // middle angle for directionESC
int maxAngle = 1800; // maximum angle for directionESC
int deadZone = 1500; // dead zone for pusherESC
int _delay = 10;     // delay for writeSpeed function
int OFFSET = 50;     // offset for currentSpeed in writeSpeed function

void writeSpeed(); // function prototype for writeSpeed function

int currentSpeed = deadZone; // initialize currentSpeed to deadZone
int angleGiven = midAngle;   // initialize angleGiven to midAngle
boolean status = true;       // initialize status to true

void setup()
{
  // put your setup code here, to run once:
  Serial.begin(9600);                        // initialize serial communication
  pusherESC.attach(pin, 1000, 2000);         // attach pusherESC to pin with pulse width range
  directionESC.attach(servoPin);             // attach directionESC to pin
  directionESC.writeMicroseconds(midAngle);  // set directionESC to middle angle
  pusherESC.writeMicroseconds(currentSpeed); // set pusherESC to dead zone speed
}

void loop()
{
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0)
  {                                                           // check if serial data is available
    String data = Serial.readStringUntil('\n');               // read serial data until newline character
    const char *string = data.c_str();                        // convert serial data to char array
    char delim[] = "#";                                       // set delimiter for strtok function
    char *token = strtok(string, delim);                      // tokenize serial data
    angleGiven = map(atoi(token), 0, 60, minAngle, maxAngle); // map token to angleGiven
    while (token != NULL)
    {                              // loop through tokens
      token = strtok(NULL, delim); // get next token
      if (token != NULL)
      { // check if token exists
        if (atoi(token) > 0)
        {                                                     // check if token is positive
          currentSpeed = map(atoi(token), 0, 10, 1560, 1900); // map token to currentSpeed
        }
        else if (atoi(token) < 0)
        {                                                      // check if token is negative
          currentSpeed = map(atoi(token), 0, -10, 1400, 1320); // map token to currentSpeed
        }
        else
        {                          // if token is zero
          currentSpeed = deadZone; // set currentSpeed to dead zone
        }
      }
    }
  }
  pusherESC.writeMicroseconds(currentSpeed);  // write currentSpeed to pusherESC
  directionESC.writeMicroseconds(angleGiven); // write angleGiven to directionESC
  Serial.println();                           // print newline character

  // writeSpeed();
  delay(12.5); // delay for stability
}

void writeSpeed()
{
  if (currentSpeed == deadZone)
  { // if current speed is at the dead zone
    currentSpeed = deadZone;
    pusherESC.writeMicroseconds(currentSpeed); // write the current speed to the ESC
  }
  else if (currentSpeed > deadZone)
  { // if current speed is above the dead zone
    if (status == true)
    { // if status is true
      if (currentSpeed >= maxSpeed)
      {                          // if current speed is greater than or equal to the maximum speed
        currentSpeed = maxSpeed; // set current speed to the maximum speed
      }
      pusherESC.writeMicroseconds(currentSpeed); // write the current speed to the ESC
      delay(_delay);                             // delay for the specified time
    }
    else
    {                                            // if status is false
      int tmp = currentSpeed;                    // store the current speed in a temporary variable
      currentSpeed = deadZone;                   // set current speed to the dead zone
      pusherESC.writeMicroseconds(currentSpeed); // write the current speed to the ESC
      delay(20);                                 // delay for 20 milliseconds
      while (currentSpeed <= maxSpeed)
      { // while current speed is less than or equal to the maximum speed
        if (currentSpeed >= maxSpeed)
        {                          // if current speed is greater than or equal to the maximum speed
          currentSpeed = maxSpeed; // set current speed to the maximum speed
          break;                   // break out of the loop
        }
        if (currentSpeed >= tmp)
        {                     // if current speed is greater than or equal to the temporary variable
          currentSpeed = tmp; // set current speed to the temporary variable
          break;              // break out of the loop
        }
        pusherESC.writeMicroseconds(currentSpeed); // write the current speed to the ESC
        currentSpeed += 50;                        // increment the current speed by 50
        delay(_delay);                             // delay for the specified time
      }
    }
    status = true; // set status to true
  }
  else if (currentSpeed < deadZone)
  { // if current speed is below the dead zone
    if (status == false)
    { // if status is false
      if (currentSpeed <= minSpeed)
      {                          // if current speed is less than or equal to the minimum speed
        currentSpeed = minSpeed; // set current speed to the minimum speed
      }
      pusherESC.writeMicroseconds(currentSpeed); // write the current speed to the ESC
      delay(_delay);                             // delay for the specified time
    }
    else
    {                                                     // if status is true
      int tmp = currentSpeed;                             // store the current speed in a temporary variable
      currentSpeed = deadZone;                            // set current speed to the dead zone
      pusherESC.writeMicroseconds(currentSpeed);          // write the current speed to the ESC
      delay(25);                                          // delay for 25 milliseconds
      pusherESC.writeMicroseconds(currentSpeed - OFFSET); // write the current speed minus the offset to the ESC
      delay(80);                                          // delay for 80 milliseconds
      pusherESC.writeMicroseconds(currentSpeed);          // write the current speed to the ESC
      delay(80);                                          // delay for 80 milliseconds
      while (currentSpeed >= minSpeed)
      { // while current speed is greater than or equal to the minimum speed
        if (currentSpeed <= minSpeed)
        {                          // if current speed is less than or equal to the minimum speed
          currentSpeed = minSpeed; // set current speed to the minimum speed
          break;                   // break out of the loop
        }
        if (currentSpeed <= tmp)
        {                     // check if current speed is less than or equal to the temporary speed
          currentSpeed = tmp; // if so, set the current speed to the temporary speed
          break;              // break out of the while loop
        }
        pusherESC.writeMicroseconds(currentSpeed); // set the speed of the motor controller to the current speed
        delay(_delay);                             // wait for the specified delay time
        currentSpeed -= 50;                        // decrement the current speed by 50
      }
    }
    status = false;
  }
}