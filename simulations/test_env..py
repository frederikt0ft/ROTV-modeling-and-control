import numpy as np
import holoocean
from scipy.spatial.transform import Rotation
import control as ct
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess
import scipy as sp
import datetime
import json
data = np.zeros((11, 5000, 3))   #Data matrix for logging all data in simulation


#Initial position
x_i = 0
y_i = 0
z_i = 1

#initial orientation
phi_i = 0
theta_i = -10
psi_i =   0 #-20

tick1 = 1000
tick_rate = 200
#U_val
u_val = 2
scenario = {
    "name": "hovering_dynamics",
    "package_name": "Ocean",
    "world": "SimpleUnderwater",
    "main_agent": "auv0",
    "ticks_per_sec": tick_rate,
    "lcm_provider": "file:///home/lcm.log",
    "agents": [
        {
            "agent_name": "auv0",
            "agent_type": "TorpedoAUV",
            "sensors": [
                {
                    "sensor_type": "DynamicsSensor",
                    "socket": "RotationSocket",
                    "configuration":{
                        "UseCOM": False,
                        "UseRPY": False # Dont use quaternion
                    },
                },
                {
                    "sensor_type": "RotationSensor",
                    "socket": "RotationSocket"

                },
                {
                    "sensor_type": "IMUSensor",
                    "socket": "RotationSocket"

                },
                {
                    "sensor_type": "RangeFinderSensor",
                    "socket": "SonarSocket",
                    "configuration":{
                        "LaserMaxDistance": 80,
                        "LaserCount": 1,
                        "LaserAngle": -90 ,
                        "LaserDebug": True,
                    },
                },
            ],
            "control_scheme": 1, # this is the custom dynamics control scheme
            "location": [x_i,y_i,z_i],
            "rotation": [phi_i,theta_i,psi_i]
        }
    ],
    "window_width":  800,
    "window_height": 300
}

def extract_sensor_info(x, a):
    # Extract all info from state
    quat = x[15:19]
    R = Rotation.from_quat(quat).as_matrix()
    acc = np.linalg.inv(R)@x[:3]
    vel = np.linalg.inv(R)@x[3:6]
    pos = x[6:9]
    ang_acc = x[9:12]
    ang_vel = x[12:15]
    rot = Rotation.from_quat(quat)
    #    rot_euler = rot.as_euler('xyz', degrees=True)
    #    rot_euler[0] -= phi_i
    #    rot_euler[1] -= theta_i
    #    rot_euler[2] -= psi_i
    rpy = a
    rpy[1] *= -1

    if i == 200:
        rpy[1] *= -1

    print(f"Depth: {state['RangeFinderSensor'][0]:.2f}")
    print(f"Roll: {rpy[0]:.2f}")
    print(f"Pitch: {rpy[1]:.2f}")

    sensor_data = [acc, vel, ang_acc, ang_vel, pos, rpy, R]
    for j in range(6):
        for k in range(3):
            if (float(sensor_data[j][k]) <= 0.001 and float(sensor_data[j][k]) >= 0) or (float(sensor_data[j][k]) >= -0.001 and float(sensor_data[j][k]) <= 0) :
                data[j,i,k] = 0
            else:
                data[j,i,k] = float(sensor_data[j][k])
    return sensor_data
R = np.zeros((3,3))

with holoocean.make(scenario_cfg=scenario) as env:
    lin_accel = np.array([0, 0, 0])   # 5 m/s
    rot_accel = np.array([1, 0, 0])
    for i in range(200*3):
        print(i)
        acc = np.array([R@lin_accel,R@rot_accel])
        state = env.step(acc)
        sensor_data = extract_sensor_info(state["DynamicsSensor"], state["RotationSensor"])
        R = (sensor_data[-1])
        print(R)
    lin_accel = np.array([0, 0, 0])   # 5 m/s
    rot_accel = np.array([-1, 0, 0])
    for i in range(200*3,400*3):
        print(i)
        acc = np.array([R@lin_accel,R@rot_accel])
        state = env.step(acc)
        sensor_data = extract_sensor_info(state["DynamicsSensor"], state["RotationSensor"])
        R = (sensor_data[-1])
        print(R)
    lin_accel = np.array([0, 0, 0])   # 5 m/s
    rot_accel = np.array([0, 1, 0])
    for i in range(400*3,600*3):
        print(i)
        acc = np.array([R@lin_accel,R@rot_accel])
        state = env.step(acc)
        sensor_data = extract_sensor_info(state["DynamicsSensor"], state["RotationSensor"])
        R = (sensor_data[-1])
        print(R)
