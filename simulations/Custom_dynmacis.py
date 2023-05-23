import numpy as np
import holoocean
from scipy.spatial.transform import Rotation
import control as ct
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess
import scipy as sp
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
os.chdir("..")
#---------------------------------- INITIAL ROLL PITCH YAW -------------------------------------------#
phi_i = 0
theta_i = 0
psi_i = -20   #-20

u_val = 2

tick_rate = 200

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
            "location": [0,0,-27.84],
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
tick1 = 200
tick2 = 2800 + tick1

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

r1_val = 0.288
S1_val = 0.052
S2_val = 0.084
S3_val = 0.069
S4_val = 0.05
d1_val = 0.14
d2_val = 0.055
d3_val = -0.1
d4_val = -0.5

#---------------------------------------STATE SPACE -------------------------------------------

u = 0
v = 0
w = 0
p = 0
q = 0
r = 0
# x' = Ax + Bu
x_dot = np.array([u, v, w, p, q, r]) [:,np.newaxis]

A = np.array([[0, 1.00, 0, 0, 0, 0],
              [0, -0.370, 0, 0, 0.229*u_val**2, -0.252*u_val],
              [0, 0, 0, 1.00, 0, 0],
              [0, -0.0161*u_val, 0, -0.146, 0.000532*u_val**2, 0.00120],
              [0, 0, 0, 0, 0, 1.00],
              [0, 2.81*u_val, 0, 0.000277, -0.0928*u_val**2, -0.210]])
print(A)

B = np.array([[0, 0, 0],
              [-0.0509*u_val**2, -0.0509*u_val**2, -0.0471*u_val**2],
              [0, 0, 0],
              [-0.165*u_val**2, 0.166*u_val**2, -0.000507*u_val**2],
              [0, 0, 0],
              [-0.0102*u_val**2, -0.0108*u_val**2, 0.0884*u_val**2]])
print(B)

#--------------------------- LQR --------------------------------#

Q = np.array([[900.000, 0.000, 0.000, 0.000, 0.000, 0.000],
              [0.000, 30.000, 0.000, 0.000, 0.000, 0.000],
              [0.000, 0.000, 10.000, 0.000, 0.000, 0.000],
              [0.000, 0.000, 0.000, 1.00, 0.000, 0.000],
              [0.000, 0.000, 0.000, 0.000, 10.000, 0.000],
              [0.000, 0.000, 0.000, 0.000, 0.000, 5.00]])

LQR_R = np.array([[0.20, 0.000, 0.000],
                  [0.000, 0.200, 0.000],
                  [0.000, 0.000, 1.0]])

K, S, E = ct.lqr(A, B, Q, LQR_R)

#-------------------Functions------------------------------------#
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

    print(f"Depth: {state['RangeFinderSensor'][0]}")
    print(f"Roll: {rpy[0]}")
    print(f"Pitch: {rpy[1]}")

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
error_h_prev = 0
error_r_prev = 0
error_p_prev = 0

flag = False
def clamp(arr, minimum, maximum):
    return np.clip(arr, minimum, maximum)
def pid_controller(states_var, ref_h):
    #Error dynamics ()
    global flag
    global error_h_prev
    global error_r_prev
    global error_p_prev
    ref_r = 0
    ref_p = 0


    p_h = 3000 #10000 for god #1000 realistisk
    d_h = 500000 #8000000 for god #500000 realistisk

    p_r = 250 #1000 #1 realistisk
    d_r = 10000 #1 for god #1 realistisk

    p_p = 500 #1000 for god #500 realistisk
    d_p = 10000 #50000 for god #10000 realistisk

    error_h = ref_h - states_var[0]
    error_r = ref_r - states_var[2]
    error_p = ref_p - states_var[4]

    if flag:
        LF = error_h*p_h + (error_h-error_h_prev)*d_h
        RF = error_r*p_r + (error_r-error_r_prev)*d_r
        PF = error_p*p_p + (error_p-error_p_prev)*d_p


    else:
        LF = error_h*p_h
        RF = error_r*p_r
        PF = error_p*p_p

        flag = True

    force_vector = np.array([LF, RF, PF]) [:,np.newaxis]

    lift_h = (1/2)*997*1/8*u_val**2
    lift_r = (1/2)*997*1/8*u_val**2*r1_val
    lift_p = (1/2)*997*1/8*u_val**2


    T = -np.array([[lift_h*S2_val, lift_h*S2_val, lift_h*S4_val],
                  [lift_r*S2_val, -lift_r*S2_val, 0],
                  [lift_p*d2_val*S2_val, lift_p*d2_val*S2_val, lift_p*d4_val*S4_val]])

    #print("force vector")
    #print(force_vector)
    #print("inv T")
    #print(np.linalg.pinv(T))

    u = np.linalg.pinv(T) @ force_vector


    #print(f"input P: {error_h*p_h}")
    #print(f"input D: {(error_h-error_h_prev)*d_h}")
    #print(f"e_h: {error_h}")
    #print(f"e_h_prev: {error_h_prev}")


    error_h_prev = error_h
    error_r_prev = error_r
    error_p_prev = error_p

    print()
    u1 = u[0]
    u2 = u[1]
    u3 = u[2]

    return clamp(u1, -30, 30), clamp(u2, -30, 30), clamp(u3, -30, 30)

def LQR(states_var, z_ref):
    state_vector = np.array([states_var[0],states_var[1],states_var[2],states_var[3],states_var[4],states_var[5]])[:,np.newaxis]
    #print("ref")
    #print(ref)
    #print()
    #print("states")
    #print(state_vector)
    #print()
    #print("-K")
    #print(-K)
    ref_vec = np.array([z_ref, 0, 0, 0, 0, 0])[:,np.newaxis]
    u = - K @ (state_vector - ref_vec)
    #print(u)
    #print(u)
    u1 = u[0]
    u2 = u[1]
    u3 = u[2]
    return clamp(u1, -30, 30),clamp(u2, -30, 30), clamp(u3, -30, 30)

#ref = np.array([0,0,0])[:,np.newaxis]
prev_angles = np.array([0,0,0])[:,np.newaxis]
real_angles = np.array([0,0,0])[:,np.newaxis]
u1_true_prev = 0
u2_true_prev = 0
u3_true_prev = 0
def wing_model(da1,da2,da3):
    global prev_angles, real_angles, pwm, p, d, p_vec, d_vec
    desired_angles = np.vstack((da1, da2, da3))
    error_angles = desired_angles - real_angles

    if i%20 == 0:
        pwm = error_angles * 10 - (real_angles - prev_angles) * 0

    if np.any(pwm > 100):
        pwm[pwm > 100] = 100
    if np.any(pwm < -100):
        pwm[pwm < -100] = -100

    aps = pwm/10 # assuming linearity with 100 pwm = 10 aps. #aps = angle per second

    prev_angles = real_angles
    real_angles = prev_angles + aps/tick_rate #angle per second -> angles per tick





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

    for i in range(tick1,tick2):
        sensor_data = extract_sensor_info(state["DynamicsSensor"], state["RotationSensor"])
        states = extract_acc_terms(sensor_data,u1,u2,u3, tick1, state["RangeFinderSensor"], state["IMUSensor"])
        ref = 1    #Target above seabed
        u1_d, u2_d, u3_d = pid_controller(states,ref)
        #u1, u2, u3 = state_feedback_controller(states, 5, 0 ,0)
        #u1, u2, u3 = LQR(states,ref)

        u1,u2,u3 = wing_model(u1_d,u2_d,u3_d)

        R = (sensor_data[-1])
        x_dot = compute_x_dot(states, u1, u2,u3)   #u1 u2 u3
        acc = compute_acc(x_dot)

        print()
        # Step simulation
        state = env.step(acc)
    print("Finished Simulation!")

#Dataframe:
df = build_df(data)
print(df)


print("Generating plots with following parameters: ")
print(f"Ref:   {ref}")
print(f"Ticks: {tick2}")

arg1_value = ref
arg2_value = tick_rate

subprocess.run(["python", "Simulations/plots.py", str(arg1_value), str(arg2_value)])






