import serial
import time
ser = serial.Serial('com9', 115200) # Replace '/dev/ttyACM0' with the port your Arduino is connected to

#Define signal
#Hz = 100
#Duty cycle = DC

#Arduino recieve message from with three integers in the range 0-100
while True:
    num = int(input("input: "))
    num_str = "{:03d}".format(num)
    num_bytes = num_str.encode()
    ser.write(num_bytes)
