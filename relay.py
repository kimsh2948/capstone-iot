import RPi.GPIO as GPIO
import time

relay_1 = 21
GPIO.setmode(GPIO.BCM)
GPIO.setup(relay_1, GPIO.OUT)

GPIO.output(relay_1, GPIO.HIGH)
time.sleep(10)
GPIO.output(relay_1, GPIO.LOW)

GPIO.cleanup()
