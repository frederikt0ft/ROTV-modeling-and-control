import serial
import time
ser = serial.Serial('com9', 115200) # Replace '/dev/ttyACM0' with the port your Arduino is connected to

#Define signal
#Hz = 100
#Duty cycle = DC

#Arduino recieve message from with three integers in the range 0-100

var_l = []

def encode_output():
    num = int(input("input: "))
    num_str = "{:03d}".format(num)
    return num_str
while True:
    val1 = encode_output()
    val2 = encode_output()
    val3 = encode_output()
    num_str = val1+val2+val3
    num_bytes = num_str.encode()
    ser.write(num_bytes)
