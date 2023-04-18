from pymavlink import mavutil

# Connect to the autopilot system
connection = mavutil.mavlink_connection('udp:0.0.0.0:14550')

# Request the ATTITUDE message
connection.mav.request_data_stream_send(
    connection.target_system, connection.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL,
    40, 1)

# Wait for the ATTITUDE message
i = 0


while True:
    msg = connection.recv_match(type='ATTITUDE', blocking=True)
    if msg is not None:
        print(i)
        roll = msg.roll
        pitch = msg.pitch
        yaw = msg.yaw
        roll_vel = msg.rollspeed
        pitch_vel = msg.pitchspeed
        yaw_vel = msg.yawspeed
        print('Roll:', roll*180/3.14)
        print('Pitch:', pitch*180/3.14)
        print('Yaw:', yaw*180/3.14)
        print('roll_vel', roll_vel)
        print('pitch_vel', pitch_vel)
        print('yaw_vel', yaw_vel)
        print()


        i += 1