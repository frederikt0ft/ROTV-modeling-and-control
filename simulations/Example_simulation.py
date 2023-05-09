import holoocean
from holoocean.agents import HoveringAUV
import numpy as np
HoveringAUV.mass = 10
env = holoocean.make("PierHarbor-Hovering")

# The hovering AUV takes a command for each thruster
"""
An 8-length floating point vector used to specify the control on each thruster.
They begin with the front right vertical thrusters, then goes around counter-clockwise, then repeat the last four with the sideways thrusters.
"""
command = np.array([0,0,0,0,20,20,20,20])

for _ in range(1800):
    state = env.step(command)

    if "IMUSensor" in state:
        print(state["IMUSensor"])
