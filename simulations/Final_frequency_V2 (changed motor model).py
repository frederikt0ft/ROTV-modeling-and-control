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
x_i = -10
y_i = -0.1
z_i = -28.34

#initial orientation
phi_i = 0
theta_i = 0
psi_i = 0 #-20

al = 20         # Angle limit
u_val = 2      # m/s
distance = 20  # in meters

#Simulation specifications 1 sec = 200 ticks
tick1 = 200
tick2 = int(distance/u_val*tick1 + tick1)
tick_rate = 200

ref_h = 1

Control = "PID"
logging = False
logging_name = "Final"

frequency = 50
motor_model = True
plots_single = False
period = tick_rate/frequency

scenario = {
    "name": "hovering_dynamics",
    "package_name": "Ocean",
    "world": "ExampleLevel",
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

p_vector = np.array([0,0,0])[:,np.newaxis]
d_vector = np.array([0,0,0])[:,np.newaxis]

u_list = []
x_list1 = []
x_list2 = []
sonar_list = [0]
acc_list = [u_list,x_list1,x_list2]



#---------------------------------------STATE SPACE -------------------------------------------

u = v = w = p = q = r = 0
# x' = Ax + Bu
x_dot = np.array([u, v, w, p, q, r]) [:,np.newaxis]



if logging_name == "Final":
    A = np.array([[0, 1.00, 0, 0, 0, 0],
                  [0, -0.0344, 0, 0, 0.228*u_val**2, -0.252*u_val],
                  [0, 0, 0, 1.00, 0, 0],
                  [0, -0.0161*u_val, 0, -0.291, 0.000577*u_val**2, 0.000739],
                  [0, 0, 0, 0, 0, 1.00],
                  [0, 2.81*u_val, 0, 0.000554, -0.101*u_val**2, -0.129]])
    B = np.array([[0, 0, 0],
                  [-0.0506*u_val**2, -0.0506*u_val**2, -0.0471*u_val**2],
                  [0, 0, 0],
                  [-0.165*u_val**2, 0.165*u_val**2, -0.000519*u_val**2],
                  [0, 0, 0],
                  [-0.00729*u_val**2, -0.00791*u_val**2, 0.0906*u_val**2]])
if logging_name == "Final_split":
    A = np.array([[0, 1.00, 0, 0, 0, 0],
                  [0, -0.0344, 0, 0, 0.228*u_val**2, -0.252*u_val],
                  [0, 0, 0, 1.00, 0, 0],
                  [0, -0.0161*u_val, 0, -0.291, 0.000431*u_val**2, 0.000739],
                  [0, 0, 0, 0, 0, 1.00],
                  [0, 2.81*u_val, 0, 0.000554, -0.0752*u_val**2, -0.129]])
    B = np.array([[0, 0, 0],
                  [-0.0104*u_val**2, -0.0104*u_val**2, -0.0293*u_val**2],
                  [0, 0, 0],
                  [-0.034*u_val**2, 0.034*u_val**2, -0.000299*u_val**2],
                  [0, 0, 0],
                  [-0.00346*u_val**2, -0.00359*u_val**2, 0.0522*u_val**2]])


print("A:\n", A)
print()
print("B:\n", B)
print()
print(f"Control: ", Control, "\nTicks: ", tick2, "\nSpeed: ", u_val, "m/s\nMatrice:",logging_name, "\nMotor Model:", motor_model, "\nLogging:", logging)

#--------------------------- LQR --------------------------------#
Q_05 = np.diag([250,50,5,1,60,30])
LQR_R_05 = np.diag([0.7,0.7,3.2])

Q_10 = np.diag([250,50,5,1,60,30])
LQR_R_10 = np.diag([0.7,0.7,3.2])

Q_15 = np.diag([250,50,5,1,60,30])
LQR_R_15 = np.diag([0.7,0.7,3.2])

Q_20 = np.diag([250,50,5,1,60,30])
LQR_R_20 = np.diag([0.7,0.7,3.2])

Q_25 = np.diag([250,50,5,1,60,30])
LQR_R_25 = np.diag([0.7,0.7,3.2])

Q_30 = np.diag([250,50,5,1,60,30])
LQR_R_30 = np.diag([0.7,0.7,3.2])

Q_35 = np.diag([250,50,5,1,60,30])
LQR_R_35 = np.diag([0.7,0.7,3.2])

Q_40 = np.diag([250,50,5,1,60,30])
LQR_R_40 = np.diag([0.7,0.7,3.2])

Q_45 = np.diag([250,50,5,1,60,30])
LQR_R_45 = np.diag([0.7,0.7,3.2])

Q_50 = np.diag([250,50,5,1,60,30])
LQR_R_50 = np.diag([0.7,0.7,6])

Q_list = [Q_05,Q_10,Q_15,Q_20,Q_25,Q_30,Q_35,Q_40,Q_45,Q_50]
R_list = [LQR_R_05,LQR_R_10,LQR_R_15,LQR_R_20,LQR_R_25,LQR_R_30,LQR_R_35,LQR_R_40,LQR_R_45,LQR_R_50]

K_list = []

for x in range(10):
    K, S, E = ct.lqr(A, B, Q_list[x], R_list[x])
    K_list.append(K)


#-------------------Functions------------------------------------#
def map_argument_to_output(argument):
    ranges = [(0, 0.75), (0.75, 1.25), (1.25, 1.75), (1.75, 2.25), (2.25, 2.75), (2.75, 3.25),(3.25, 3.75),(3.75,4.25),(4.25,4.75),(4.75,100000)]
    for index, (lower, upper) in enumerate(ranges):
        if lower <= argument < upper:
            return index


def log(l, str,u_val_var, df):
    now = datetime.datetime.now()
    tid = now.strftime("%Y-%d-%m-%H-%M-%S")
    string = f"{str}"

    if l == True:
        df = df.round(3)
        df.to_csv(f'Control/logs/{str}/{logging_name}_{motor_model}_{u_val_var}_{frequency}.csv', index = False)

        # Create a dictionary to hold the data
        data1 = {
            "u_val": u_val,
            "Distance": distance,
            "Q": Q.tolist(),  # Convert NumPy array to list
            "LQR_R": LQR_R.tolist(),
            "K": K.tolist(),
            "A": A.tolist(),
            "B": B.tolist(),
            "p_vec": p_vector.tolist(),
            "d_vec": d_vector.tolist(),
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
        filename = f"Control/logs/{str}/{logging_name}_{motor_model}_{u_val_var}_{frequency}_config.json"

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

ref_pid = np.array([ref_h, 0, 0]) [:,np.newaxis]
error_prev = np.array([0, 0, 0]) [:,np.newaxis]
diff = np.array([0,0,0]) [:, np.newaxis]
sum_error = np.array([0,0,0], dtype=float) [:, np.newaxis]


flag = False


r1_val = 0.288
S1_val = 0.0051
S2_val = 0.2*0.053+0.089*0.2*0.5
S3_val = 0.0693
S4_val = 2*0.19*0.072
d1_val = 0.14
d2_val = 0.055
d3_val = -0.1
d4_val = -0.5

def clamp(arr, minimum, maximum):
    return np.clip(arr, minimum, maximum)

for x in range(1,11):




p_vector_05 = np.array([40, 1, 100]) [:,np.newaxis]
i_vector_05 = np.array([0, 0, 0]) [:,np.newaxis]
d_vector_05 = np.array([220, 0, 40]) [:,np.newaxis]

p_vector_10 = np.array([40, 1, 100]) [:,np.newaxis]
i_vector_10 = np.array([0, 0, 0]) [:,np.newaxis]
d_vector_10 = np.array([220, 0, 40]) [:,np.newaxis]

p_vector_15 = np.array([40, 1, 100]) [:,np.newaxis]
i_vector_15 = np.array([0, 0, 0]) [:,np.newaxis]
d_vector_15 = np.array([220, 0, 40]) [:,np.newaxis]

p_vector_20 = np.array([40, 1, 100]) [:,np.newaxis]
i_vector_20 = np.array([0, 0, 0]) [:,np.newaxis]
d_vector_20 = np.array([220, 0, 40]) [:,np.newaxis]

p_vector_25 = np.array([40, 1, 100]) [:,np.newaxis]
i_vector_25 = np.array([0, 0, 0]) [:,np.newaxis]
d_vector_25 = np.array([220, 0, 40]) [:,np.newaxis]

p_vector_30 = np.array([40, 1, 100]) [:,np.newaxis]
i_vector_30 = np.array([0, 0, 0]) [:,np.newaxis]
d_vector_30 = np.array([220, 0, 40]) [:,np.newaxis]

p_vector_35 = np.array([40, 1, 100]) [:,np.newaxis]
i_vector_35 = np.array([0, 0, 0]) [:,np.newaxis]
d_vector_35 = np.array([220, 0, 40]) [:,np.newaxis]

p_vector_40 = np.array([40, 1, 100]) [:,np.newaxis]
i_vector_40 = np.array([0, 0, 0]) [:,np.newaxis]
d_vector_40 = np.array([220, 0, 40]) [:,np.newaxis]

p_vector_45 = np.array([40, 1, 100]) [:,np.newaxis]
i_vector_45 = np.array([0, 0, 0]) [:,np.newaxis]
d_vector_45 = np.array([220, 0, 40]) [:,np.newaxis]

p_vector_50 = np.array([40, 1, 100]) [:,np.newaxis]
i_vector_50 = np.array([0, 0, 0]) [:,np.newaxis]
d_vector_50 = np.array([220, 0, 40]) [:,np.newaxis]

p_vector_list = [p_vector_05,p_vector_10,p_vector_15,p_vector_20,p_vector_25,p_vector_30,p_vector_35,p_vector_40,p_vector_45,p_vector_50]
i_vector_list = [i_vector_05,i_vector_10,i_vector_15,i_vector_20,i_vector_25,i_vector_30,i_vector_35,i_vector_40,i_vector_45,i_vector_50]
d_vector_list = [d_vector_05,d_vector_10,d_vector_15,d_vector_20,d_vector_25,d_vector_30,d_vector_35,d_vector_40,d_vector_45,d_vector_50]
def pid_controller(states_var, u_val_var):
    global error_prev, diff, flag, sum_error, p_vector, d_vector
    l = map_argument_to_output(u_val_var)
    state_vector = np.array([states_var[0], states_var[2], states_var[4]]) [:,np.newaxis]
    p_vector = p_vector_list[l]
    i_vector = i_vector_list[l]
    d_vector = d_vector_list[l]
    #p_error
    error = ref_pid - state_vector


    #i_error
    if i%period == 0:
        sum_error += error

    if error[0] < 0 and error_prev[0] > 0 or error[0] > 0 and error_prev[0] < 0:
        sum_error[0] = 0
    if error[1] < 0 and error_prev[1] > 0 or error[1] > 0 and error_prev[1] < 0:
        sum_error[1] = 0
    if error[2] < 0 and error_prev[2] > 0 or error[2] > 0 and error_prev[2] < 0:
        sum_error[2] = 0


    #d_error
    if i%period == 0 and flag:
        diff = error - error_prev
    error_prev = error

    if flag:
        force_vector = error * p_vector + sum_error * i_vector + diff * d_vector

    else:
        force_vector = error * p_vector + sum_error * i_vector
        flag = True

    r1_val = 0.288



    lift_h = (1/2)*997*1/8*u_val**2
    lift_r = (1/2)*997*1/8*u_val**2*r1_val
    lift_p = (1/2)*997*1/8*u_val**2



    T = -np.array([[lift_h*S2_val,        lift_h*S2_val, lift_h*S4_val],
                   [lift_r*S2_val,       -lift_r*S2_val, 0],
                   [lift_p*d2_val*S2_val, lift_p*d2_val*S2_val, lift_p*d4_val*S4_val]])


    T_god = np.array([[-0.1, -0.357, -0.195],
                      [-0.1, 0.357, -0.195],
                      [-0.014, 0.000, 0.165]])



    MA = np.array([[-1.7,  -1, -0.8],
                   [-1.7,   1, -0.8],
                   [-0.005, 0,  3]]) / 10

    MA = np.linalg.pinv(T)

    u = T_god @ force_vector


    u1 = u[0]
    u2 = u[1]
    u3 = u[2]



    return clamp(u1, -al, al), clamp(u2, -al, al), clamp(u3, -al, al)
def LQR(states_var, u_val_var):
    l = map_argument_to_output(u_val_var)
    state_vector = np.array([states_var[0],states_var[1],states_var[2],states_var[3],states_var[4],states_var[5]])[:,np.newaxis]
    ref_vec = np.array([ref_h, 0, 0, 0, 0, 0])[:,np.newaxis]


    u = - K_list[l] @ (state_vector - ref_vec)

    u1 = u[0]
    u2 = u[1]
    u3 = u[2]
    return clamp(u1, -al, al),clamp(u2, -al, al), clamp(u3, -al, al)

prev_angles = np.array([0,0,0])[:,np.newaxis]
real_angles = np.array([0,0,0])[:,np.newaxis]

if Control == "PID":
    p = 7
    d = 0

if Control == "LQR":
    p = 7
    d = 0

aps = np.array([0.0,0.0,0.0])[:,np.newaxis]
#def APS(pwm_var):

lowest_error_angle = 0.2
b = lowest_error_angle*p

slope = (100-17)/(100-b)
intersection = 17- b * slope

def wing_model(da1,da2,da3):
    global prev_angles, real_angles, pwm, p, d
    desired_angles = np.vstack((da1, da2, da3))
    error_angles = desired_angles - real_angles

    p_f = 0.000005 # tjekkes om
    for o in range(len(error_angles)):
        if error_angles[o] > 0 and error_angles[o] < p_f:
            error_angles[o] = 0
        if error_angles[o] < 0 and error_angles[o] > -p_f:
            error_angles[o] = 0

    if i%period == 0:
        pwm = error_angles * p - (real_angles - prev_angles) * d
        if np.any(pwm > 100):
            pwm[pwm > 100] = 100
        if np.any(pwm < -100):
            pwm[pwm < -100] = -100

        #print(pwm)

        if np.any(pwm < 0):
            pwm[pwm < 0] = pwm[pwm < 0] * slope - intersection #
        if np.any(pwm > 0):
            pwm[pwm > 0] = pwm[pwm > 0] * slope + intersection #


        #print(pwm)

        for o in range(len(pwm)):
            if pwm[o] > 0 and pwm[o] < 17: #
                pwm[o] = 0
            if pwm[o] < 0 and pwm[o] > -17: #
                pwm[o] = 0

        for o in range(len(pwm)):
            if pwm[o] == 0:
                aps[o] = 0
            if pwm[o] < 0:
                aps[o] = 0.135 * pwm[o] + 1.5147 # laves om
            if pwm[o] > 0:
                aps[o] = 0.135 * pwm[o] - 1.5147 # laves om


    #print(aps)

    #aps = pwm/10 # assuming linearity with 100 pwm = 10 aps. #aps = angle per second

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
        if i%period == 0:
            states_frequency = states

        if Control == "PID":
            u1, u2, u3 = pid_controller(states_frequency, u_val)
        if Control == "LQR":
            u1, u2, u3 = LQR(states_frequency, u_val)
        if motor_model:
            u1,u2,u3 = wing_model(u1,u2,u3)

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

if plots_single == True:
    subprocess.run(["python", "Simulations/plots_single.py", str(arg1_value), str(arg2_value)])
else:
    subprocess.run(["python", "Simulations/plots.py", str(arg1_value), str(arg2_value)])
