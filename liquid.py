import RPi.GPIO as GPIO
import time

liquid = 20
GPIO.setmode(GPIO.BCM)
GPIO.setup(liquid, GPIO.IN)
GPIO.setup

try:
    while True:
        result = GPIO.input(liquid)
        print(result)
        time.sleep(2)
finally:
    GPIO.cleanup()


