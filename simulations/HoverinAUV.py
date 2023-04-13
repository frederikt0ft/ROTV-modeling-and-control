import holoocean
from holoocean.agents import HoveringAUV
import numpy as np
HoveringAUV.mass = 10
env = holoocean.make("PierHarbor-Hovering")

# The hovering AUV takes a command for each thruster
command = np.array([10,10,10,10,0,0,0,0])

force = np.zeros(3)
torque = np.zeros(3)
HoveringAUV.I = np.eye(3)
lin_accel = force / HoveringAUV.mass
ang_accel = np.linalg.inv(HoveringAUV.I)@torque
u = np.append(lin_accel, ang_accel)
for _ in range(1800):
    state = env.step(u)