from flask import Flask
import RPi.GPIO as GPIO
import time
import asyncio

app = Flask(__name__)

@app.route('/')
def hello_world():
    relay_1 = 21
    GPIO.setmode(GPIO.BCM)

    GPIO.setup(relay_1, GPIO.OUT) 

    GPIO.output(relay_1, GPIO.HIGH) 
    time.sleep(10) 
    GPIO.output(relay_1, GPIO.LOW)

    GPIO.cleanup()

    return 'Hello World'


