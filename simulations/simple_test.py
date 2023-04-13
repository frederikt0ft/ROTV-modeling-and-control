import numpy as np
import holoocean
from holoocean.agents import HoveringAUV
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import time
import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Process
import multiprocessing as mp
import ctypes
plt.style.use("fivethirtyeight")

import os

scenario = {
    "name": "hovering_dynamics",
    "package_name": "Ocean",
    "world": "SimpleUnderwater",
    "main_agent": "auv0",
    "ticks_per_sec": 200,
    "lcm_provider": "file:///home/lcm.log",
    "agents": [
        {
            "agent_name": "auv0",
            "agent_type": "TorpedoAUV",
            "sensors": [
                {
                    "sensor_type": "DynamicsSensor",
                    "configuration":{
                        "UseCOM": True,
                        "UseRPY": False # Dont use quaternion
                    }
                },
            ],
            "control_scheme": 1, # this is the custom dynamics control scheme
            "location": [-80,0,-28],
            "rotation": [0,0,150]
        }
    ]
}

g = 9.81 # gravity
b = 0.1 # linear damping
c = 0.1 # angular damping
# HoveringAUV.mass += 1 # alternatively make it sink

def build_df(data):
    columns = ["acc_x", "acc_y", "acc_z","vel_x", "vel_y", "vel_z","ang_acc_roll", "ang_acc_pitch", "ang_acc_yaw","ang_vel_roll", "ang_vel_pitch", "ang_vel_yaw","x","y","z","roll","pitch","yaw"]
    lst1 = data[0][:,0]
    lst2 = data[0][:,1]
    lst3 = data[0][:,2]
    lst4 = data[1][:,0]
    lst5 = data[1][:,1]
    lst6 = data[1][:,2]
    lst7 = data[2][:,0]
    lst8 = data[2][:,1]
    lst9 = data[2][:,2]
    lst10 = data[3][:,0]
    lst11 = data[3][:,1]
    lst12 = data[3][:,2]
    lst13 = data[4][:,0]
    lst14 = data[4][:,1]
    lst15 = data[4][:,2]
    lst16 = data[5][:,0]
    lst17 = data[5][:,1]
    lst18 = data[5][:,2]
    df = pd.DataFrame(list(zip(lst1, lst2, lst3, lst4, lst5, lst6, lst7, lst8, lst9, lst10, lst11, lst12, lst13, lst14, lst15, lst16, lst17, lst18)),
                      columns =columns)
    df.to_csv('Simulation_data.csv', index = False)
    return df

def extract_df(x):
    # Extract all info from state
    quat = x[15:19]
    R = Rotation.from_quat(quat).as_matrix()
    acc = R@x[:3]
    vel = R@x[3:6]
    pos = x[6:9]
    ang_acc = R@x[9:12]
    ang_vel = R@x[12:15]
    rpy = x[15:18]
    sensor_data = [acc, vel, ang_acc, ang_vel, pos, rpy, R]
    return sensor_data

def main_program():

    # Create list for storing data
    acc_d = np.array([])
    vel_d= np.array([])
    ang_acc_d = np.array([])
    ang_vel_d = np.array([])
    pos_d = np.array([])
    rpy_d = np.array([])



    # List of lists
    data = np.zeros((6, tick, 3))
    R = np.zeros((3,3))
    u = np.zeros(6)
    # Make environment
    u = 1 # positiv er frem
    v = 0.1 # positiv er til venstre
    w = 0 # positiv er op
    p = 0.05 # positiv er clockwise
    q = -0.1 # positiv er snuden kører nedad
    r = 0.002 # positiv er anticlockwise fra toppen
    lin_accel = R@np.array([[u], [v], [w]])




    build_df(data)
    with holoocean.make(scenario_cfg=scenario) as env:
        for i in range(tick):
            lin_accel = R@np.array([[u], [v], [w]])
            rot_accel = R@np.array([[p], [q], [r]])
            acc = np.array([lin_accel,rot_accel])
            # Step simulation
            state = env.step(acc)
            # Get accelerations to pass to HoloOcean
            sensor_data = extract_df(state["DynamicsSensor"])

            R = (sensor_data[-1])

            for j in range(6):
                for k in range(3):
                    data[j,i,k] = float(sensor_data[j][k])
            if i%100 == 0:
                print("_------------------------------------DF----------------------------------------------")
                df = build_df(data)
                k = df
                print(k)


                # wait for a short time
                time.sleep(1)

    print("Finished Simulation!")

    #Dataframe:
    df = build_df(data)


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x_vals = [0]
y_vals = [0]



def animate(i):
    df = k
    print(df)
    print(np.shape(df))
    x_vals = np.arange(0,1000,1)
    y_vals = data[0][:,0]
    plt.cla()
    plt.plot(x_vals,y_vals)


def plot():

    ani = FuncAnimation(plt.gcf(), animate, interval = 1050)
    plt.tight_layout()
    plt.show()

tick = 1000
data = np.zeros((6, tick, 3))
df = build_df(data)

k = mp.Value(ctypes.py_object)
k.value = df

if __name__ == '__main__':
    # create a shared dictionary to hold the dataframe reference

    p = Process(target=main_program)
    p1 = Process(target=plot)
    p1.start()
    p.start()
    p.join()
    p1.join()