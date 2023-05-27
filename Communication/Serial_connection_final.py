import serial
import time
from pymavlink import mavutil

# Connect to the autopilot system
connection = mavutil.mavlink_connection('udp:0.0.0.0:14550')

# Request the ATTITUDE message
connection.mav.request_data_stream_send(
    connection.target_system, connection.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL,
    40, 1)

ser = serial.Serial('com9', 115200) # Replace '/dev/ttyACM0' with the port your Arduino is connected to

#Define signal
#Hz = 100
#Duty cycle = DC

#Arduino recieve message from with three integers in the range 0-100
hall_data = 0
var_l = []
data = 0
def encode_output():
    num = int(input("Val: "))
    #num = 100
    num_str = "{:03d}".format(num)
    return num_str
def read_hall_sensor():
    global data
    if ser.in_waiting > 0:
        data = ser.readline().decode('ascii').rstrip()
    return data
i =0
while True:
    val1 = encode_output()
    val2 = encode_output()
    val3 = "000"
    num_str = val1 + val2 + val3
    num_bytes = num_str.encode()
    ser.write(num_bytes)
    i += 1
    print(i)
    #-----------------------HALL SENSOR---------------

    if ser.in_waiting > 0:
        hall_data = ser.readline().decode('ascii').rstrip()
    print(f"hall sensor: {hall_data}")




    msg = connection.recv_match(type='ATTITUDE', blocking=True)
    if msg is not None:

        roll = msg.roll
        pitch = msg.pitch
        yaw = msg.yaw
        roll_vel = msg.rollspeed
        pitch_vel = msg.pitchspeed
        yaw_vel = msg.yawspeed
        """     
        print('Roll:', roll*180/3.14)
        print('Pitch:', pitch*180/3.14)
        print('Yaw:', yaw*180/3.14)
        print('roll_vel', roll_vel)
        print('pitch_vel', pitch_vel)
        print('yaw_vel', yaw_vel)
        print()
        """
    time.sleep(0.1)



