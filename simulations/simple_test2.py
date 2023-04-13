import holoocean
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from holoocean.agents import HoveringAUV
import time
a = holoocean.util.get_os_key()
print(a)
env = holoocean.make("Openwater-Torpedo")

bottom = 0


# The hovering AUV takes a command for each thruster
ticks = 8000
Left = -4
Right = 4
tail = 0
thrust = 50  #37 rent flad

command = np.array([Left,Right,tail,bottom,thrust])

x = []
y = []
z = []

accel_x = []
accel_y = []
accel_z = []

ang_vel_roll = []
ang_vel_pitch = []
ang_vel_yaw = []

Depth_sensor_log = []
IMU_sensor_log = []
""" Inertial Measurement Unit sensor. Returns a 2D numpy array of:
[ [accel_x, accel_y, accel_z],
  [ang_vel_roll,  ang_vel_pitch, ang_vel_yaw],
  [accel_bias_x, accel_bias_y, accel_bias_z],
  [ang_vel_bias_roll,  ang_vel_bias_pitch, ang_vel_bias_yaw]    ]
  where the accleration components are in m/s and the angular velocity is in rad/s """

Depth_sensor_log = []
"""Pressure/Depth Sensor.

Returns a 1D numpy array of:
[position_z]
"""

Location_sensor_log = []
"""Gets the location of the agent in the world.

Returns coordinates in [x, y, z]"""



for _ in range(ticks):
    state = env.step(command)

    if "IMUSensor" in state:
        IMU_data = state["IMUSensor"]
        IMU_sensor_log.append(IMU_data)
        accel_x.append(IMU_data[0,0])
        accel_y.append(IMU_data[0,1])
        accel_z.append(IMU_data[0,2])
        ang_vel_roll.append(IMU_data[1,0])
        ang_vel_pitch.append(IMU_data[1,1])
        ang_vel_yaw.append(IMU_data[1,2])

    if "LocationSensor" in state:
        LOC_data = state["LocationSensor"]
        x.append(LOC_data[0])
        y.append(LOC_data[1])
        z.append(LOC_data[2])
    if "RangeFinderSensor" in state:
        RangeFinderSensor_Data = state["RangeFinderSensor"]
        print(RangeFinderSensor_Data)


    if "DVLSensor" in state:
        dvl = state["DVLSensor"]
        #print("DVL:")w
        #print(dvl)
        #print()

    if "DepthSensor" in state:
        DepthSensor = state["DepthSensor"]
        #print("DepthSensor:")
        #print(DepthSensor)
        Depth_sensor_log.append(DepthSensor)
        #print()

    if "SinglebeamSonar" in state:
        SinglebeamSonar = state["SinglebeamSonar"]
        #print("SinglebeamSonar:")
        #print(SinglebeamSonar)
        #print()


print("Finished Simulation!")


columns = ["Depth", "x", "y", "z","accel_x", "accel_y", "accel_z","ang_vel_roll", "ang_vel_pitch", "ang_vel_yaw"]
lst1 = Depth_sensor_log
lst2 = x
lst3 = y
lst4 = z
lst5 = accel_x
lst6 = accel_y
lst7 = accel_z
lst8 = ang_vel_roll
lst9 = ang_vel_pitch
lst10 = ang_vel_yaw

#Dataframe:
df = pd.DataFrame(list(zip(lst1, lst2, lst3, lst4, lst5, lst6, lst7, lst8, lst9, lst10)),
                  columns =columns)


# For plot
t1 = np.arange(0, len(df["Depth"]))


fig, axs = plt.subplots(3, 3)

print(len(IMU_sensor_log))
print(np.shape(IMU_sensor_log[3]))

# plot [1 1]
axs[0, 0].plot(t1, df["Depth"])
axs[0, 0].set_title("Depth")

# plot [1 1]
param = "accel_x"
axs[1, 0].plot(t1, df[param])
axs[1, 0].set_title(param)

# plot [1 1]
param = "accel_y"
axs[1, 1].plot(t1, df[param])
axs[1, 1].set_title(param)

# plot [1 1]
param = "accel_z"
axs[1, 2].plot(t1, df[param])
axs[1, 2].set_title(param)

# plot [1 1]
param = "ang_vel_roll"
axs[2, 0].plot(t1, df[param])
axs[2, 0].set_title(param)

# plot [1 1]
param = "ang_vel_pitch"
axs[2, 1].plot(t1, df[param])
axs[2, 1].set_title(param)

# plot [1 1]
param = "ang_vel_yaw"
axs[2, 2].plot(t1, df[param])
axs[2, 2].set_title(param)


plt.show()
ax = plt.axes(projection='3d')
ax.plot3D(x,y,z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
plt.show()
print(df)