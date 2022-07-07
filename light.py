import RPi.GPIO as GPIO
import time

output = predictions1[0]

LED_PIN1 = 17
LED_PIN2 = 27

GPIO.setmode(GPIO.BCM)

GPIO.setup(LED_PIN1, GPIO.OUT)
GPIO.setup(LED_PIN2, GPIO.OUT)

i=0

if(output == 0):
    while i < 5:
        GPIO.output(LED_PIN1, GPIO.HIGH)
        time.sleep(1)
        
        GPIO.output(LED_PIN1, GPIO.LOW)
        time.sleep(1)
        
        i+=1

elif(output == 1):
    while i < 5:
        GPIO.output(LED_PIN2, GPIO.HIGH)
        time.sleep(1)
        
        GPIO.output(LED_PIN2, GPIO.LOW)
        time.sleep(1)
        
        i+=1

else:
    print("Invalid Output!!!")

GPIO.cleanup()
i = 0
