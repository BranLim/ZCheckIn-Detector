import amg8833_i2c
import time
#
#####################################
# Initialization of Sensor
#####################################
#
t0 = time.time()
sensor = []

while (time.time()-t0)<1: # wait 1sec for sensor to start
    try:
        # AD0 = GND, addr = 0x68 | AD0 = 5V, addr = 0x69
        sensor = amg8833_i2c.AMG8833(addr=0x69) # start AMG8833
    except:
        sensor = amg8833_i2c.AMG8833(addr=0x68)
    finally:
        pass
time.sleep(0.1) # wait for sensor to settle

# If no device is found, exit the script
if sensor==[]:
    print("No AMG8833 Found - Check Your Wiring")



def read_temperature():
    global sensor

    pix_to_read = 64 # read all 64 pixels
    status,pixels = sensor.read_temp(pix_to_read) # read pixels with status
    #Magic number for correction. We probably need some trigometry to fix it
    return max(pixels) + 2.0
        
    

