import holoocean
import numpy as np
import matplotlib
import csv


pose_log = []
dvl_log = []
"""A basic example of how to use the HoveringAUV agent."""
env = holoocean.make("SimpleUnderwater-Hovering")

# This command tells the AUV go forward with a power of "10"
# The last four elements correspond to the horizontal thrusters (see docs for more info)
command = np.array([0, 0, 0, 10,50, 50, 10, 10])
for _ in range(1000):
    state = env.step(command)
    # To access specific sensor data:
    if "PoseSensor" in state:
        pose = state["PoseSensor"]
        #print("Pose:")
        #print(pose)
        #print()
        pose_log.append(pose)
    # Some sensors don't tick every timestep, so we check if it's received.
    if "DVLSensor" in state:
        dvl = state["DVLSensor"]
        print("DVL:")
        print(dvl)
        print()
        dvl_log.append(dvl)
    if "SinglebeamSonar" in state:
        DepthSensor = state["SinglebeamSonar"]
        print("SinglebeamSonar:")
        print(DepthSensor)
        print()

    # This command tells the AUV to go down with a power of "10"
    # The first four elements correspond to the vertical thrusters
command = np.array([-10, -10, -10, -10, 0, 0, 0, 0])
for _ in range(1000):
    # We alternatively use the act function
    env.act("auv0", command)
    state = env.tick()

    # You can control the AgentFollower camera (what you see) by pressing v to toggle spectator
    # mode. This detaches the camera and allows you to move freely about the world.
    # Press h to view the agents x-y-z location
    # You can also press c to snap to the location of the camera to see the world from the perspective of the
    # agent. See the Controls section of the ReadMe for more details.

print(f"len pose: {len(pose_log)}")
print(f"len dvl : {len(dvl_log)}")

fields = ["velocity_x", "velocity_y", "velocity_z", "range_x_forw", "range_y_forw", "range_x_back", "range_y_back"]


with open('GFG.csv', 'w') as f:

    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(dvl_log)


