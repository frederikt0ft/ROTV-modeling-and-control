import serial

# Define the serial port and baud rate
ser = serial.Serial('com9', 115200) # Change the 'COM3' to the appropriate port name for your computer

while True:
    # Read data from the serial port and print it
    if ser.in_waiting > 0:
        data = ser.readline().decode('ascii').rstrip()
        print(data)