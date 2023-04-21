import serial
import time
ser = serial.Serial('com9', 115200) # Replace '/dev/ttyACM0' with the port your Arduino is connected to

while True:
    num = int(input("input: "))
    num_str = "{:03d}".format(num)
    num_bytes = num_str.encode()
    ser.write(num_bytes)
