"""
Example of how to read all the parameters from the Autopilot with pymavlink
"""

# Disable "Broad exception" warning
# pylint: disable=W0703

import time
import sys

# Import mavutil
from pymavlink import mavutil


# Create the connection
master = mavutil.mavlink_connection('udpin:0.0.0.0:14550')
# Wait a heartbeat before sending commands
master.wait_heartbeat()

# Request all parameters
master.mav.param_request_read_send(
    master.target_system, master.target_component, b"SURFACE_DEPTH", -1)

while True:
    time.sleep(0.01)
    try:
        msg =  master.recv_match()

        # Extract the roll, pitch, and yaw angles from the attitude message
        if msg.get_type() == 'ATTITUDE':
            roll = msg.roll
            pitch = msg.pitch
            yaw = msg.yaw

            # Print the angles
            print(f"Roll: {roll:.2f} degrees, Pitch: {pitch:.2f} degrees, Yaw: {yaw:.2f} degrees")
    except Exception as error:
        print(error)
        sys.exit(0)