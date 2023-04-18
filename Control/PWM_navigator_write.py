# Import mavutil
from pymavlink import mavutil
import time
# Create the connection
master = mavutil.mavlink_connection('udpin:0.0.0.0:14550')
# Wait a heartbeat before sending commands
master.wait_heartbeat()

# Create a function to send RC values
# More information about Joystick channels
# here: https://www.ardusub.com/operators-manual/rc-input-and-output.html#rc-inputs
def set_rc_channel_pwm(channel_id, pwm=1500):
    """ Set RC channel pwm value
    Args:
        channel_id (TYPE): Channel ID
        pwm (int, optional): Channel pwm value 1100-1900
    """
    if channel_id < 1 or channel_id > 18:
        print("Channel does not exist.")
        return

    # Mavlink 2 supports up to 18 channels:
    # https://mavlink.io/en/messages/common.html#RC_CHANNELS_OVERRIDE
    rc_channel_values = [16384 for _ in range(18)]

    rc_channel_values[channel_id - 1] = pwm
    print(pwm)
    master.mav.rc_channels_override_send(
        master.target_system,                # target_system
        master.target_component,             # target_component
        *rc_channel_values)                  # RC channel list, in microseconds.



def map_range(value, from_min, from_max, to_min, to_max):
    """
    Maps a value from one range to another range.
    """
    return int((value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min)

value = 100 # replace with the value you want to map

set_rc_channel_pwm(10,map_range(value, 0, 100, 2, 16384))

for x in range(101):
    set_rc_channel_pwm(10,map_range(x, 0, 100, 2, 16384))
    print(x)
    time.sleep(0.1)