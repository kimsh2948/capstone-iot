import RPi.GPIO as GPIO
import  time
import spidev
#import dht11
import Adafruit_DHT
sensor = Adafruit_DHT.DHT11
pin = 16

liquid = 20
GPIO.setmode(GPIO.BCM)
GPIO.setup(liquid, GPIO.IN)

spi=spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz=50000

from socket import socket, AF_INET, SOCK_STREAM
import json

server_socket = socket(AF_INET, SOCK_STREAM)

ip = '192.168.0.14'

port = 3000

result = 0

#instance = dht11.DHT11(pin = 16)

print("process start..")

server_socket.bind((ip, port))

print("bind..")

server_socket.listen(100)

print("listen..")

client_socket, addr = server_socket.accept()

def read_spi_adc(adcChannel):
    adcValue=0
    buff=spi.xfer2([1,(8+adcChannel)<<4,0])
    adcValue=((buff[1]&3)<<8)+buff[2]
    return  adcValue

try:
    while True:
        # r -> soil
        adcValue=read_spi_adc(0)
        print("Moisture:%d "%(adcValue))
        #r = requests.post(url, data={'adcValue':adcValue}).text
        #print("r= %s"%r)

        # r2 -> temperature
        #result = instance.read()
        h, t = Adafruit_DHT.read_retry(sensor, pin)
        if h is not None and t is not None :
            print("Temperature = {0:0.1f}*C Humidity = {1:0.1f}%".format(t, h))
        else :
            print('Read error')

        #print("Temperature: %-3.1f C"% result.temperature)
        #print("Humidity: %-3.1f %%"% result.humidity)

        #r2 = requests.post(url, data={'temperature':result.temperature, 'humidity': result.humidity}).text
        #print('r2= %s'%r2)

        # r3 -> liquid
        r3 = GPIO.input(liquid)
        print('r3= %s'%r3)
        #r3 = requests.post(url, data={'liquid': r3}).text
        #print('r3= %s'%r3)

        #data = {'adcValue': adcValue, 'temperature': result.temperature, 'humidity': result.humidity, 'liquid': r3}

        data = {'adcValue': adcValue, 'temperature': t, 'humidity': h, 'liquid': r3}
        data_string = json.dumps(data)

        client_socket.send(bytes(data_string, 'utf-8'))
        #client_socket.recv(1024).decode('utf-8')

        time.sleep(1)

finally:
    GPIO.cleanup()
    spi.close()
    server_socket.close()
