import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os

print(os.system("python -V"))
plt.style.use("fivethirtyeight")

x_vals = [0]
y_vals = [0]

plt.plot(x_vals,y_vals)

plt.axis([0, len(x_vals), max(y_vals), min(y_vals)])

index = count()
def animate(i):
    df = pd.read_csv("Simulation_data.csv")
    x_vals = np.arange(0,len(df["x"]),1)
    y_vals = df["x"]
    plt.cla()
    plt.plot(x_vals,y_vals)
    plt.axis([0, len(x_vals), min(y_vals), max(y_vals)])

while True:
    try:
        ani = FuncAnimation(plt.gcf(), animate, interval = 150)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Error")
        continue