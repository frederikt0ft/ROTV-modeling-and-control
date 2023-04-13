import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
from mpl_toolkits import mplot3d
from multiprocessing import Process
df = pd.read_csv("Simulation_data.csv")
plt.style.use("dark_background")
t1 = np.arange(0, len(df["x"]))/200

# For plot



#------------------------------------------------------------PLOT 1----------------------------------------------------------------------#

fig, axs = plt.subplots(3,3)


param = "roll"
axs[0, 0].plot(t1, df[param])
axs[0, 0].set_title(param)
axs[0, 0].set_ylabel('Angle (degree)')

param = "pitch"
axs[0, 1].plot(t1, df[param])
axs[0, 1].set_title(param)

param = "yaw"
axs[0, 2].plot(t1, df[param])
axs[0, 2].set_title(param)

param = "x"
axs[1, 0].plot(t1, df[param])
axs[1, 0].set_title(param)
axs[1, 0].set_ylabel('Position (m)')


param = "y"
axs[1, 1].plot(t1, df[param])
axs[1, 1].set_title(param)

param = "z"
axs[1, 2].plot(t1, df[param])
axs[1, 2].set_title(param)

param = "ang_vel_roll"
axs[2, 0].plot(t1, df[param])
axs[2, 0].set_title(param)
axs[2, 0].set_xlabel('time (s)')
axs[2, 0].set_ylabel('Angle velocity (degree/s)')


param = "ang_vel_pitch"
axs[2, 1].plot(t1, df[param])
axs[2, 1].set_title(param)

param = "ang_vel_yaw"
axs[2, 2].plot(t1, df[param])
axs[2, 2].set_title(param)

plt.show()
