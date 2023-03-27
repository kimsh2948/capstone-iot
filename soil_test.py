import RPi.GPIO as GPIO
import  time
import spidev
import requests
import dht11

liquid = 20
GPIO.setmode(GPIO.BCM)
GPIO.setup(liquid, GPIO.IN)

spi=spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz=50000

url = "http://210.125.29.233:80"

instance = dht11.DHT11(pin = 16)
result = instance.read()

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
        r = requests.post(url, data={'adcValue':adcValue}).text
        print("r= %s"%r)
        

        # r2 -> temperature
        print("Temperature: %-3.1f C"% result.temperature)
        print("Humidity: %-3.1f %%"% result.humidity)
        
        r2 = requests.post(url, data={'temperature':result.temperature, 'humidity': result.humidity}).text
        print('r2= %s'%r2)

        # r3 -> liquid
        r3 = GPIO.input(liquid)
        print('r3= %s'%r3)
        r3 = requests.post(url, data={'liquid': r3}).text
        print('r3= %s'%r3)

        time.sleep(1)

finally:
    GPIO.cleanup()
    spi.close()


