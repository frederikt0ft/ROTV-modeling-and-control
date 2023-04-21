import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys

ref = sys.argv[1]


print("arg1:", int(ref))



df = pd.read_csv("Simulation_data_1.csv")
plt.style.use("dark_background")

t1 = np.arange(0, len(df["x"]))/200

# For plot
fig1 = plt.figure()


#------------------------------------------------------------PLOT 2----------------------------------------------------------------------#


fig1 = plt.axes(projection='3d')
fig1.plot3D(df["x"], df["y"], df["z"])
fig1.set_xlabel('x')
fig1.set_ylabel('y')
fig1.set_zlabel('z')

#------------------------------------------------------------PLOT 3----------------------------------------------------------------------#

    # For plot
fig2, axs2 = plt.subplots(2,3)

param = "x1"
axs2[0, 0].plot(t1, df[param])
axs2[0, 0].set_title(param)

param = "x2"
axs2[1, 0].plot(t1, df[param])
axs2[1, 0].set_title(param)

param = "x3"
axs2[0, 1].plot(t1, df[param])
axs2[0, 1].set_title(param)

param = "x4"
axs2[1, 1].plot(t1, df[param])
axs2[1, 1].set_title(param)

param = "x5"
axs2[0, 2].plot(t1, df[param])
axs2[0, 2].set_title(param)

param = "x6"
axs2[1, 2].plot(t1, df[param])
axs2[1, 2].set_title(param)
fig2.suptitle('States', fontsize=16)
#------------------------------------------------------------PLOT 2----------------------------------------------------------------------#


plt.figure(3)
plt.axhline(y=int(ref), color='yellow', linestyle='-', label = "ref")
plt.plot(t1,df["x1"], label = "x1")
plt.plot(t1,df["u1"], label = "u1")
plt.plot(t1,df["u2"], label = "u2")
plt.plot(t1,df["u3"], label = "u3")
plt.title("Actuator input + depth + ref")
plt.legend()



plt.show()
