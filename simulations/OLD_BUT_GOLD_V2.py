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

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
os.chdir("..")
#---------------------------------- INITIAL ROLL PITCH YAW -------------------------------------------#


#Initial position
x_i = 0
y_i = 0
z_i = -27.84 + 10

#initial orientation
phi_i = 0
theta_i = 0
psi_i = -20   #-20

al = 20         # Angle limit
u_val = 2      # m/s

#Simulation specifications 1 sec = 200 ticks
tick1 = 200
tick2 = 4200 + tick1
tick_rate = 200

ref_h = 1 + 10
Control = "PID"
logging = False

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

# Create list for storing data
acc_d = np.array([])
vel_d= np.array([])
ang_acc_d = np.array([])
ang_vel_d = np.array([])
pos_d = np.array([])
rpy_d = np.array([])


# List of lists
data = np.zeros((11, tick2, 3))   #Data matrix for logging all data in simulation
R = np.zeros((3,3))

#Initial conditions:
u1 = u2 = u3 = x1 = x2 = x3 = x4 = x5 = x6 = 0
p = d = p_h = d_h = p_r = d_r = p_p = d_p = 0

p_vec = np.array([0,0,0])[:,np.newaxis]
d_vec = np.array([0,0,0])[:,np.newaxis]

u_list = []
x_list1 = []
x_list2 = []
sonar_list = [0]
acc_list = [u_list,x_list1,x_list2]


r1_val = 0.288
S1_val = 0.0051
S2_val = 0.095
S3_val = 0.0693
S4_val = 0.044
d1_val = 0.14
d2_val = 0.055
d3_val = -0.1
d4_val = -0.5

#---------------------------------------STATE SPACE -------------------------------------------

u = v = w = p = q = r = 0
# x' = Ax + Bu
x_dot = np.array([u, v, w, p, q, r]) [:,np.newaxis]

A = np.array([[0, 1.00, 0, 0, 0, 0],
              [0, -0.0344, 0, 0, 0.249*u_val**2, -0.252*u_val],
              [0, 0, 0, 1.00, 0, 0],
              [0, -0.0161*u_val, 0, -0.291, 0.000532*u_val**2, 0.000739],
              [0, 0, 0, 0, 0, 1.00],
              [0, 2.81*u_val, 0, 0.000554, -0.0928*u_val**2, -0.129]])

#gammel added mass

A = np.array([[0, 1.00, 0, 0, 0, 0],
              [0, -0.0222, 0, 0, 0.147*u_val**2, -0.163*u_val],
              [0, 0, 0, 1.00, 0, 0],
              [0, -0.0118*u_val, 0, -0.103, 0.000225*u_val**2, 0.000313],
              [0, 0, 0, 0, 0, 1.00],
              [0, 5.81*u_val, 0, 0.000235, -0.111*u_val**2, -0.154]])


print("A:\n", A)

B = np.array([[0, 0, 0],
              [-0.0509*u_val**2, -0.0509*u_val**2, -0.0471*u_val**2],
              [0, 0, 0],
              [-0.165*u_val**2, 0.166*u_val**2, -0.000507*u_val**2],
              [0, 0, 0],
              [-0.0102*u_val**2, -0.0108*u_val**2, 0.0884*u_val**2]])

#gammel added mass
B = np.array([[0, 0, 0],
              [-0.0328*u_val**2, -0.0328*u_val**2, -0.0304*u_val**2],
              [0, 0, 0],
              [-0.0588*u_val**2, 0.0588*u_val**2, -0.000215*u_val**2],
              [0, 0, 0],
              [-0.0124*u_val**2, -0.0127*u_val**2, 0.105*u_val**2]])


print()
print("B:\n", B)
print()
print(f"Control: ", Control, "\nTicks: ", tick2, "\nSpeed: ", u_val, "m/s")

#--------------------------- LQR --------------------------------#

Q = np.array([[220.000, 0.000, 0.000, 0.000, 0.000, 0.000],
              [0.000,125.000, 0.000, 0.000, 0.000, 0.000],
              [0.000, 0.000, 10.000, 0.000, 0.000, 0.000],
              [0.000, 0.000, 0.000, 1, 0.000, 0.000],
              [0.000, 0.000, 0.000, 0.000, 60.000, 0.000],
              [0.000, 0.000, 0.000, 0.000, 0.000, 45.0]])

LQR_R = np.array([[0.350, 0.000, 0.000],
                  [0.000, 0.350, 0.000],
                  [0.000, 0.000, 1.6]])*2

K, S, E = ct.lqr(A, B, Q, LQR_R)

#-------------------Functions------------------------------------#
def log(l, str,u_val_var, df):
    now = datetime.datetime.now()
    tid = now.strftime("%Y-%d-%m-%H-%M-%S")
    string = f"{str}"

    if l == True:
        df = df.round(3)
        df.to_csv(f'Control/logs/{str}/{u_val_var}_{tid}.csv', index = False)

        # Create a dictionary to hold the data
        data1 = {
            "u_val": u_val,
            "ticks": tick2,
            "Q": Q.tolist(),  # Convert NumPy array to list
            "LQR_R": LQR_R.tolist(),
            "K": K.tolist(),
            "A": A.tolist(),
            "B": B.tolist(),
            "p_vec": p_vec.tolist(),
            "d_vec": d_vec.tolist(),
            "wing_p": p,
            "wing_d": d,
            "Angle limit": al,
            "x_i": x_i,
            "y_i": y_i,
            "z_i": z_i,
            "phi_i":phi_i,
            "theta_i":theta_i,
            "psi_i":psi_i
        }

        # Define the filename for the JSON log file
        filename = f"Control/logs/{str}/{u_val_var}_{tid}_config.json"

        # Write the data to the JSON file
        with open(filename, "w") as f:
            json.dump(data1, f, indent=4)
def build_df(data):

    global df
    columns = ["acc_x", "acc_y", "acc_z","vel_x", "vel_y", "vel_z","ang_acc_roll", "ang_acc_pitch", "ang_acc_yaw","ang_vel_roll", "ang_vel_pitch", "ang_vel_yaw","x","y","z","roll","pitch","yaw","u1","u2","u3","x1","x2","x3","x4","x5","x6", "u1_d","u2_d","u3_d", "pwm_1", "pwm_2", "pwm_3"]
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
    lst28 = data[9][:,0]    #u1_d
    lst29 = data[9][:,1]    #u2_d
    lst30 = data[9][:,2]    #u3_d
    lst31 = data[10][:,0]    #u1_d
    lst32 = data[10][:,1]    #u2_d
    lst33 = data[10][:,2]    #u3_d
    df = pd.DataFrame(list(zip(lst1, lst2, lst3, lst4, lst5, lst6, lst7, lst8, lst9, lst10, lst11, lst12, lst13, lst14, lst15, lst16, lst17, lst18, lst19, lst20, lst21, lst22, lst23, lst24, lst25, lst26, lst27, lst28, lst29,lst30, lst31, lst32,lst33)),
                      columns =columns)
    print(len(df["x"]))

    df.to_csv('Control/Simulation_data.csv', index = False)

    return df
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
    rot_euler = rot.as_euler('xyz', degrees=True)
    rot_euler[0] -= phi_i
    rot_euler[1] -= theta_i
    rot_euler[2] -= psi_i
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
def extract_acc_terms(sensor_data_var, u1_var,u2_var,u3_var, tick, sonar_sensor, imu_sensor_var):
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

    lin_accel = R@np.array([[0], [0], [heave_acc1]])
    rot_accel = R@np.array([[roll_acc1], [pitch_acc1], [0]])
    lin_accel[0] = 0
    lin_accel[1] = 0
    rot_accel[2] = 0

    return np.array([lin_accel,rot_accel])
#-------------------Controllers---------------------#

diff = error_prev = error = np.array([0,0,0])[:,np.newaxis]
ref_pid = np.array([ref_h,0,0])[:,np.newaxis]

flag = False

def clamp(arr, minimum, maximum):
    return np.clip(arr, minimum, maximum)
def pid_controller(states_var):
    #Error dynamics ()
    global flag, p_h, d_h, p_r, d_r, p_p, d_p, error,error_prev, diff
    state_vec = np.array([states_var[0],states_var[2],states_var[4]])[:,np.newaxis]
    p_vec = np.array([85, 1, 450]) [:,np.newaxis] #250 1 450
    d_vec = np.array([5500, 1, 700]) [:,np.newaxis] #7000 1 700
    error = ref_pid - state_vec

    if i%20 == 0:
        diff = error - error_prev
        error_prev = error

    if flag:
        force_vec = error * p_vec + diff * d_vec
    else:
        force_vec = error * p_vec
        flag = True

    lift_h = (1/2)*997*1/8*u_val**2
    lift_r = (1/2)*997*1/8*u_val**2*r1_val
    lift_p = (1/2)*997*1/8*u_val**2

    T = -np.array([[lift_h*S2_val, lift_h*S2_val, lift_h*S4_val],
                   [lift_r*S2_val, -lift_r*S2_val, 0],
                   [lift_p*d2_val*S2_val, lift_p*d2_val*S2_val, lift_p*d4_val*S4_val]])

    u = np.linalg.pinv(T) @ force_vec

    u1 = u[0]
    u2 = u[1]
    u3 = u[2]

    return clamp(u1, -al, al), clamp(u2, -al, al), clamp(u3, -al, al)
def LQR(states_var):
    state_vector = np.array([states_var[0],states_var[1],states_var[2],states_var[3],states_var[4],states_var[5]])[:,np.newaxis]
    ref_vec = np.array([ref_h, 0, 0, 0, 0, 0])[:,np.newaxis]
    u = - K @ (state_vector - ref_vec)

    u1 = u[0]
    u2 = u[1]
    u3 = u[2]
    return clamp(u1, -al, al),clamp(u2, -al, al), clamp(u3, -al, al)

prev_angles = np.array([0,0,0])[:,np.newaxis]
real_angles = np.array([0,0,0])[:,np.newaxis]

if Control == "PID":
    p = 10
    d = 0

if Control == "LQR":
    p = 10
    d = 0
def wing_model(da1,da2,da3):
    global prev_angles, real_angles, pwm, p, d, p_vec, d_vec
    desired_angles = np.vstack((da1, da2, da3))
    error_angles = desired_angles - real_angles

    if i%20 == 0:
        pwm = error_angles * p - (real_angles - prev_angles) * d

    if np.any(pwm > 100):
        pwm[pwm > 100] = 100
    if np.any(pwm < -100):
        pwm[pwm < -100] = -100

    aps = pwm/10 # assuming linearity with 100 pwm = 10 aps. #aps = angle per second

    prev_angles = real_angles
    real_angles = prev_angles + aps/tick_rate #angle per second -> angles per tick



    #Log Desird angle
    j = 9
    for k in range(3):
        data[j,i,k] = desired_angles[k]

    #Log pwm
    j = 10
    for k in range(3):
        data[j,i,k] = pwm[k]

    return real_angles[0], real_angles[1], real_angles[2]


# Make environment
with holoocean.make(scenario_cfg=scenario) as env:
    lin_accel = np.array([u_val, 0, 0])   # 5 m/s
    rot_accel = np.array([0, 0, 0])
    for i in range(tick1):
        acc = np.array([R@lin_accel,R@rot_accel])
        # Step simulation
        state = env.step(acc)
        sensor_data = extract_sensor_info(state["DynamicsSensor"], state["RotationSensor"])
        states = extract_acc_terms(sensor_data,u1,u2,u3, 0, state["RangeFinderSensor"], state["IMUSensor"])
        R = (sensor_data[-1])
    print()
    for i in range(tick1,tick2):
        print(i)
        sensor_data = extract_sensor_info(state["DynamicsSensor"], state["RotationSensor"])
        states = extract_acc_terms(sensor_data,u1,u2,u3, tick1, state["RangeFinderSensor"], state["IMUSensor"])
        if i%20 == 0:
            states_10 = states


        if Control == "PID":
            u1_d, u2_d, u3_d = pid_controller(states_10)
        if Control == "LQR":
            u1_d, u2_d, u3_d = LQR(states_10)
        u1,u2,u3 = wing_model(u1_d,u2_d,u3_d)

        R = (sensor_data[-1])

        x_dot = compute_x_dot(states, u1, u2,u3)   #u1 u2 u3
        acc = compute_acc(x_dot)
        # Step simulation
        state = env.step(acc)
        print()
    print("Finished Simulation!")

#Dataframe:
df = build_df(data)
log(logging, Control, u_val, df)
print(df)


print("Generating plots with following parameters: ")
print(f"Ref:   {ref_h}")
print(f"Ticks: {tick2}")

arg1_value = ref_h
arg2_value = tick_rate

subprocess.run(["python", "Simulations/plots.py", str(arg1_value), str(arg2_value)])
