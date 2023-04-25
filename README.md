# ROTV-modeling-and-control
Repository contains the code for modeling and control of towed underwater vehicle. Hereunder it is possible to create custom dynamics 
for your viechle and create your own enviorments.

The simulation software is based on the open source software HoloOcean: https://holoocean.readthedocs.io/en/latest/index.html


## Overview
The project is split into 3 parts:

* HoloOcean
* Holodeck (Unreal Engine 4.27)
* Client (python)

##### HoloOcean: 
HoloOcean is an underwater robotics simulator developed by E. Potokar and S. Ashford and M. Kaess and J. Mangelson,
in may 2022. HoloOcean refers to the complete project containing both the engine and client. 

##### HoloDeck (not incluced in repoistory due to big size)
HoloDeck is the development enviorment for the simulator. It is based on Unreal Engine 4.27. In the engine you
are able to develop custom enviorments and agents. Furthermore you are able to add sockets to your
agent on which you can attach various sensors. HoloDeck can be opened by clicking the "Holodeck.uproject" file.
When done editing in Unreal Engine you go to "File" -> "Package Project" -> "Windows (64-bit)". This will create 
the Holodeck.exe program which the client will run. See "paths.txt" for location of the packaged Holodeck.exe

##### Client (python)
The clients runs Holodeck.exe, using a scenario profile which is a .json file. See paths.txt to see available scenarios
profiles. It is within the python you send commands to your agent, but there are different Control Schemes to each agent(see docs).
The control scheme should be set in the scenario profile. 

## Installation
To install the software you should clone the repository and move all folders/files in the install folder to your respective place on your local pc - see "paths.txt".
Afterwards you should be able to run the "Example_simulation.py". You might need to install a few python modules to make this work

