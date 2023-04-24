
import time
# Import mavutil
from pymavlink import mavutil


master = mavutil.mavlink_connection('udpin:0.0.0.0:15000')


# Make sure the connection is valid
master.wait_heartbeat()

# Get some information !
i = 0
while True:
    try:
        k = master.recv_match().to_dict()
        print(k)
    except:
        pass
    time.sleep(0.1)
