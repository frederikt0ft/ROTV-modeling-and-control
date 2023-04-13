import numpy as np
import holoocean
from holoocean.agents import HoveringAUV
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from multiprocessing import Process
import control as ct
from pyquaternion import Quaternion
import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use("dark_background")

import os
#---------------------------------- INITIAL ROLL PITCH YAW -------------------------------------------#
phi_i = 3
theta_i = 0
psi_i = -20


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
            "location": [0,0,-28.5],
            "rotation": [phi_i,theta_i,psi_i]
        }
    ]
}


# Create list for storing data
acc_d = np.array([])
vel_d= np.array([])
ang_acc_d = np.array([])
ang_vel_d = np.array([])
pos_d = np.array([])
rpy_d = np.array([])
tick1 = 200
tick2 = 6000 +tick1

# List of lists
data = np.zeros((9, tick2, 3))
R = np.zeros((3,3))

#Initial conditions:
u1 = 0
u2 = 0
u3 = 0
x1 = 0
x2 = 0
x3 = 0
x4 = 0
x5 = 0
x6 = 0

u_list = []
x_list1 = []
x_list2 = []
sonar_list = [0]
acc_list = [u_list,x_list1,x_list2]

#---------------------------------------STATE SPACE -------------------------------------------

u = 0
v = 0
w = 0
p = 0
q = 0
r = 0
# x' = Ax + Bu
x_dot = np.array([u, v, w, p, q, r]) [:,np.newaxis]
""" - ORGINAL

A = np.array([[0,    1,    0,    0,    0,    0],
              [0, 0.58,    0,    0, 4.35,-0.80],
              [0,    0,    0,    1,    0,    0],
              [0,    0,    0,    0,    0,    0],
              [0,    0,    0,    0,    0,    1],
              [0,36.03,    0,    0,-4.12, 0.58]])

B = np.array([[0,        0,       0],
              [-0.73,-0.73,    -0.8],
              [0,        0,       0],
              [-2.39, 2.39,       0],
              [0,        0,       0],
              [-0.41,-0.41,    4.19]])

"""

A = np.array([[0,     1,    0,    0,    0,     0],
              [0,-0.005,    0,    0, 3.81, -0.80],
              [0,     0,    0,    1,    0,     0],
              [0,-0.048,    0,-0.33,    0,     0],
              [0,     0,    0,    0,    0,     1],
              [0, 19.36,    0,    0,-3.55,-0.023]])

B = np.array([[0,        0,       0],
              [-0.73,-0.73,   -0.87],
              [0,        0,       0],
              [ -1.7,  1.7,   -0.06],
              [0,        0,       0],
              [-0.165,-0.17,    2.32]])

K = np.array([[-5.237, -12.348, -7.073, -7.354, -6.508, -4.548],
              [-5.246, -12.362, 7.069, 7.349, -6.504, -4.542],
              [-6.712, -2.030, -0.006, -0.006, 5.945, 7.824]])

def build_df(data):
    columns = ["acc_x", "acc_y", "acc_z","vel_x", "vel_y", "vel_z","ang_acc_roll", "ang_acc_pitch", "ang_acc_yaw","ang_vel_roll", "ang_vel_pitch", "ang_vel_yaw","x","y","z","roll","pitch","yaw","u1","u2","u3","x1","x2","x3","x4","x5","x6"]
    lst1 = data[0][:,0]     #acc_x
    lst2 = data[0][:,1]     #acc_y
    lst3 = data[0][:,2]     #acc_z
    lst4 = data[1][:,0]     #vel_x
    lst5 = data[1][:,1]     #vel_y
    lst6 = data[1][:,2]     #vel_z
    lst7 = data[2][:,0]     #ang_acc_roll
    lst8 = data[2][:,1]     #ang_acc_pitch
    lst9 = data[2][:,2]     #ang_acc_yaw
    lst10 = data[3][:,0]    #ang_vel_roll
    lst11 = data[3][:,1]    #ang_vel_pitch
    lst12 = data[3][:,2]    #ang_vel_yaw
    lst13 = data[4][:,0]    #x
    lst14 = data[4][:,1]    #y
    lst15 = data[4][:,2]    #z
    lst16 = data[5][:,0]    #roll
    lst17 = data[5][:,1]    #pitch
    lst18 = data[5][:,2]    #yaw
    lst19 = data[6][:,0]    #u1
    lst20 = data[6][:,1]    #u2
    lst21 = data[6][:,2]    #u3
    lst22 = data[7][:,0]    #x1
    lst23 = data[7][:,1]    #x2
    lst24 = data[7][:,2]    #x3
    lst25 = data[8][:,0]    #x4
    lst26 = data[8][:,1]    #x5
    lst27 = data[8][:,2]    #x6
    df = pd.DataFrame(list(zip(lst1, lst2, lst3, lst4, lst5, lst6, lst7, lst8, lst9, lst10, lst11, lst12, lst13, lst14, lst15, lst16, lst17, lst18, lst19, lst20, lst21, lst22, lst23, lst24, lst25, lst26, lst27)),
                      columns =columns)
    print(len(df["x"]))
    df.to_csv('Simulation_data_1.csv', index = False)

    return df
def extract_sensor_info(x, a):
    # Extract all info from state
    quat = x[15:19]
    R = Rotation.from_quat(quat).as_matrix()
    acc = x[:3]
    vel = x[3:6]
    pos = x[6:9]
    ang_acc = x[9:12]
    ang_vel = x[12:15]
    rot = Rotation.from_quat(quat)
    rot_euler = rot.as_euler('xyz', degrees=True)
    rot_euler[0] -= phi_i
    rot_euler[1] -= theta_i
    rot_euler[2] -= psi_i
    rpy = a
    rpy[1] *= -1

    print(f"Pitch: {rpy[1]}")
    sensor_data = [acc, vel, ang_acc, ang_vel, pos, rpy, R] #vel[2]  =x2, rpy[0] = x3, ang_vel[0] = x4, rpy[1] = x5, ang_vel[1] = x6
    for j in range(6):
        for k in range(3):
            if (float(sensor_data[j][k]) <= 0.001 and float(sensor_data[j][k]) >= 0) or (float(sensor_data[j][k]) >= -0.001 and float(sensor_data[j][k]) <= 0) :
                data[j,i,k] = 0
            else:
                data[j,i,k] = float(sensor_data[j][k])
    return sensor_data
def extract_acc_terms(sensor_data_var, u1_var,u2_var,u3_var, tick, sonar_sensor, imu_sensor_var):
    print(imu_sensor_var[1,0])
    if sonar_sensor[0] == 80:
        sonar_sensor[0] = sonar_list[-1]

    u_list_temp = [float(u1_var),float(u2_var),float(u3_var)]
    x_list1_temp = [sonar_sensor[0],float(sensor_data_var[1][2]),float(sensor_data_var[5][0])]
    x_list2_temp = [float(imu_sensor_var[1,0])*57.3, float(sensor_data_var[5][1]), -float(sensor_data_var[3][1])*57.3]
    u_list.append(u_list_temp)
    x_list1.append(x_list1_temp)
    x_list2.append(x_list2_temp)
    sonar_list.append(sonar_sensor[0])


    for j in range(6,9):
        for k in range(3):
            if (float(acc_list[j-6][i][k]) <= 0.001 and float(acc_list[j-6][i][k]) >= 0) or (float(acc_list[j-6][i][k]) >= -0.001 and float(acc_list[j-6][i][k]) <= 0):
                data[j,i,k] = 0
            else:
                data[j,i,k] = float(acc_list[j-6][i][k])
    return [x_list1_temp[0],x_list1_temp[1],x_list1_temp[2],x_list2_temp[0],x_list2_temp[1],x_list2_temp[2]]
def compute_x_dot(x_states_var, u1_var, u2_var, u3_var):

    x_states1 = np.array([x_states_var[0],x_states_var[1],x_states_var[2],x_states_var[3],x_states_var[4],x_states_var[5]]) [:,np.newaxis]
    u_input = np.array([u1_var, u2_var, u3_var]) #[:,np.newaxis]
    x_dot = A @ x_states1 + B @ u_input
    return x_dot
def compute_acc(x_dot_var):
    heave_vel1   = x_dot_var[0][0] # positiv er frem
    heave_acc1   = x_dot_var[1][0] # positiv er til venstre
    roll_vel1    = x_dot_var[2][0] # positiv er op
    roll_acc1    = x_dot_var[3][0] # positiv er clockwise
    pitch_vel1   = x_dot_var[4][0] # positiv er snuden kører nedad
    pitch_acc1   = -x_dot_var[5][0] # positiv er anticlockwise fra toppen

    lin_accel = R@np.array([[0], [0], [heave_acc1]])/200
    rot_accel = R@np.array([[roll_acc1], [pitch_acc1], [0]])/200


    return np.array([lin_accel,rot_accel])
#-------------------Controllers---------------------#
error_h_prev = 0
error_p_prev = 0

def pd_controller(states_var):
    #Error dynamics ()
    global error_h_prev
    global error_p_prev
    ref_h = 1
    ref_p = 0

    p_h = -5
    d_h = 10

    p_p = 4
    d_p = 100

    error_h = ref_h - states_var[0]
    error_p = ref_p - states_var[4]


    u1 = error_h*p_h + (error_h-error_h_prev)*d_h
    u2 = error_h*p_h + (error_h-error_h_prev)*d_h
    u3 = error_p*p_p + (error_p-error_p_prev)*d_p

    #print(f"input P: {error_h*p_h}")
    #print(f"input D: {(error_h-error_h_prev)*d_h}")
    #print(f"e_h: {error_h}")
    #print(f"e_h_prev: {error_h_prev}")


    error_h_prev = error_h
    error_p_prev = error_p

    print()

    return u1, u2, u3
def state_feedback_controller(states_var, ref_h, ref_r, ref_p):
    desired_poles1 = [0, -2, 0,0, (-2+1j),(-2-1j)]
    K=ct.place(A,B,desired_poles1)
    print(np.shape(K))
    feedback = -K @ states_var
    print(np.shape(feedback))
    u1 = ref_h + feedback[0]
    u2 = ref_r + feedback[1]
    u3 = ref_p + feedback[2]

    print(u1, u2, u3)
    return u1, u2, u3
def LQR(states_var, ref0, ref1,ref2):
    state_vector = np.array([states_var[0],states_var[1],states_var[2],states_var[3],states_var[4],states_var[5]])[:,np.newaxis]
    ref = np.array([[ref0],[ref1],[ref2]])
    u = -K @ state_vector + ref
    u1 = u[0]
    u2 = u[1]
    u3 = u[2]
    return u1,u2, u3

ref = np.array([1,0,0])[:,np.newaxis]

# Make environment
def extract_acc_terms(sensor_data, u1, u2, u3, param, param1, param2):
    pass


with holoocean.make(scenario_cfg=scenario) as env:

    lin_accel = np.array([0, 0, 0])
    rot_accel = np.array([0, 0, 0])
    for i in range(tick1):
        acc = np.array([R@lin_accel,R@rot_accel])
        # Step simulation
        state = env.step(acc)
        sensor_data = extract_sensor_info(state["DynamicsSensor"], state["RotationSensor"])
        states = extract_acc_terms(sensor_data,u1,u2,u3, 0, state["RangeFinderSensor"], state["IMUSensor"])
        print(state["RangeFinderSensor"])
        R = (sensor_data[-1])

    for i in range(tick1,tick2):
        print(f"Depth: {state['RangeFinderSensor'][0]}")
        sensor_data = extract_sensor_info(state["DynamicsSensor"], state["RotationSensor"])
        states = extract_acc_terms(sensor_data,u1,u2,u3, tick1, state["RangeFinderSensor"], state["IMUSensor"])

        #u1, u2, u3 = pd_controller(states)
        #u1, u2, u3 = state_feedback_controller(states, 5, 0 ,0)
        u1, u2, u3 = LQR(states,1,0,0)
        R = (sensor_data[-1])
        x_dot = compute_x_dot(states, u1, u2,u3)
        acc = compute_acc(x_dot)



        # Step simulation
        state = env.step(acc)
    print("Finished Simulation!")

#Dataframe:
df = build_df(data)
print(df)
exec(open("plots.py").read())
exec(open("plots2.py").read())

