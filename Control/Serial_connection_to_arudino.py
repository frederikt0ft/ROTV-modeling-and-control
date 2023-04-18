import serial
import time
ser = serial.Serial('com9', 115200) # Replace '/dev/ttyACM0' with the port your Arduino is connected to

num = 2
num_str = str(num)

num_bytes = num_str.encode()


while True:
    ser.write(num_bytes)
    time.sleep(0.1)
    print("send")