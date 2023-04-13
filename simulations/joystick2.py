"""
Sample Python/Pygame Programs
Simpson College Computer Science
http://programarcadegames.com/
http://simpson.edu/computer-science/

Show everything we can pull off the joystick
"""
import holoocean
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pygame

env = holoocean.make("Openwater-Torpedo")


# The hovering AUV takes a command for each thruster
ticks = 9000
Left = 8
Right = 8
tail = -7
bottom = 0
thrust = 800

command = np.array([Left,Right,tail,bottom,thrust])

x = []
y = []
z = []

accel_x = []
accel_y = []
accel_z = []

ang_vel_roll = []
ang_vel_pitch = []
ang_vel_yaw = []

Depth_sensor_log = []
IMU_sensor_log = []
""" Inertial Measurement Unit sensor. Returns a 2D numpy array of:
[ [accel_x, accel_y, accel_z],
  [ang_vel_roll,  ang_vel_pitch, ang_vel_yaw],
  [accel_bias_x, accel_bias_y, accel_bias_z],
  [ang_vel_bias_roll,  ang_vel_bias_pitch, ang_vel_bias_yaw]    ]
  where the accleration components are in m/s and the angular velocity is in rad/s """

Depth_sensor_log = []
"""Pressure/Depth Sensor.

Returns a 1D numpy array of:
[position_z]
"""

Location_sensor_log = []
"""Gets the location of the agent in the world.

Returns coordinates in [x, y, z]"""

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class TextPrint(object):
    """
    This is a simple class that will help us print to the screen
    It has nothing to do with the joysticks, just outputting the
    information.
    """
    def __init__(self):
        """ Constructor """
        self.reset()
        self.x_pos = 10
        self.y_pos = 10
        self.font = pygame.font.Font(None, 20)

    def print(self, my_screen, text_string):
        """ Draw text onto the screen. """
        text_bitmap = self.font.render(text_string, True, BLACK)
        my_screen.blit(text_bitmap, [self.x_pos, self.y_pos])
        self.y_pos += self.line_height

    def reset(self):
        """ Reset text to the top of the screen. """
        self.x_pos = 10
        self.y_pos = 10
        self.line_height = 15

    def indent(self):
        """ Indent the next line of text """
        self.x_pos += 10

    def unindent(self):
        """ Unindent the next line of text """
        self.x_pos -= 10


pygame.init()

# Set the width and height of the screen [width,height]
size = [500, 700]
screen = pygame.display.set_mode(size)

pygame.display.set_caption("My Game")

# Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

# Initialize the joysticks
pygame.joystick.init()

# Get ready to print
textPrint = TextPrint()

# -------- Main Program Loop -----------
while not done:
    # EVENT PROCESSING STEP
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

        # Possible joystick actions: JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN
        # JOYBUTTONUP JOYHATMOTION
        if event.type == pygame.JOYBUTTONDOWN:
            print("Joystick button pressed.")
        if event.type == pygame.JOYBUTTONUP:
            print("Joystick button released.")

    # DRAWING STEP
    # First, clear the screen to white. Don't put other drawing commands
    # above this, or they will be erased with this command.
    screen.fill(WHITE)
    textPrint.reset()

    # Get count of joysticks
    joystick_count = pygame.joystick.get_count()

    textPrint.print(screen, "Number of joysticks: {}".format(joystick_count))
    textPrint.indent()

    # For each joystick:
    for i in range(joystick_count):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()

        textPrint.print(screen, "Joystick {}".format(i))
        textPrint.indent()

        # Get the name from the OS for the controller/joystick
        name = joystick.get_name()
        textPrint.print(screen, "Joystick name: {}".format(name))

        # Usually axis run in pairs, up/down for one, and left/right for
        # the other.
        axes = joystick.get_numaxes()
        textPrint.print(screen, "Number of axes: {}".format(axes))
        textPrint.indent()

        axis_values = []
        for i in range(axes):
            axis = joystick.get_axis(i)
            axis_values.append(axis)
            textPrint.print(screen, "Axis {} value: {:>6.3f}".format(i, axis))
        textPrint.unindent()

        buttons = joystick.get_numbuttons()
        textPrint.print(screen, "Number of buttons: {}".format(buttons))
        textPrint.indent()

        for i in range(buttons):
            button = joystick.get_button(i)
            textPrint.print(screen, "Button {:>2} value: {}".format(i, button))
        textPrint.unindent()

        # Hat switch. All or nothing for direction, not like joysticks.
        # Value comes back in an array.
        hats = joystick.get_numhats()
        textPrint.print(screen, "Number of hats: {}".format(hats))
        textPrint.indent()

        for i in range(hats):
            hat = joystick.get_hat(i)
            textPrint.print(screen, "Hat {} value: {}".format(i, str(hat)))
        textPrint.unindent()

        textPrint.unindent()

    # ALL CODE TO DRAW SHOULD GO ABOVE THIS COMMENT

    # Go ahead and update the screen with what we've drawn.
    pygame.display.flip()

    # Limit to 60 frames per second
    print(axis_values)
    Left = -axis_values[0]*20
    Right = axis_values[0]*20
    tail = -axis_values[1]*20
    bottom = 0
    thrust = -axis_values[2]*1000
    command = np.array([Left,Right,tail,bottom,thrust])

    state = env.step(command)
    if "IMUSensor" in state:
        IMU_data = state["IMUSensor"]
        IMU_sensor_log.append(IMU_data)
        accel_x.append(IMU_data[0,0])
        accel_y.append(IMU_data[0,1])
        accel_z.append(IMU_data[0,2])
        ang_vel_roll.append(IMU_data[1,0])
        ang_vel_pitch.append(IMU_data[1,1])
        ang_vel_yaw.append(IMU_data[1,2])

    if "LocationSensor" in state:
        LOC_data = state["LocationSensor"]
        x.append(LOC_data[0])
        y.append(LOC_data[1])
        z.append(LOC_data[2])
    if "RangeFinderSensor" in state:
        RangeFinderSensor_Data = state["RangeFinderSensor"]
        print(RangeFinderSensor_Data)


    if "DVLSensor" in state:
        dvl = state["DVLSensor"]
        #print("DVL:")w
        #print(dvl)
        #print()
    if "DepthSensor" in state:
        DepthSensor = state["DepthSensor"]
        #print("DepthSensor:")
        #print(DepthSensor)
        Depth_sensor_log.append(DepthSensor)
        #print()
    if "SinglebeamSonar" in state:
        SinglebeamSonar = state["SinglebeamSonar"]
        #print("SinglebeamSonar:")
        #print(SinglebeamSonar)
        #print()



    clock.tick(60)

# Close the window and quit.
# If you forget this line, the program will 'hang'
# on exit if running from IDLE.
print("Finished Simulation!")
pygame.quit()