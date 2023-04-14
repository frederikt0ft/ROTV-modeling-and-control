import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pymavlink import mavutil

# Create lists to hold roll, pitch, and yaw values
roll_values = []
pitch_values = []
yaw_values = []

# Define function to update plot
def update_plot(frame):
    plt.cla()
    plt.plot(roll_values, label='Roll')
    plt.plot(pitch_values, label='Pitch')
    plt.plot(yaw_values, label='Yaw')
    plt.legend()

# Connect to the autopilot system
connection = mavutil.mavlink_connection('udp:0.0.0.0:14550')

# Request the ATTITUDE message
connection.mav.request_data_stream_send(
    connection.target_system, connection.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL,
    100, 1)

# Define function to process received message
def process_message(msg):
    if msg.get_type() == 'ATTITUDE':
        roll_values.append(msg.roll)
        pitch_values.append(msg.pitch)
        yaw_values.append(msg.yaw)

# Create animation object
ani = FuncAnimation(plt.gcf(), update_plot, interval=100)

# Set up message handler
connection.mav.set_callback(process_message)

# Start plot
plt.show()

# Wait for the ATTITUDE message
while not roll_values:
    pass

# Disconnect from autopilot system
connection.close()