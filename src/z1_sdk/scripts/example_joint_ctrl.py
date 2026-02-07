#!/usr/bin/python3

import time
import rospkg
import os
import sys

z1_sdk_path = rospkg.RosPack().get_path('z1_sdk')

# Try devel/lib first (for built packages), then fall back to source lib
devel_lib_path = os.path.join(os.path.dirname(os.path.dirname(z1_sdk_path)), 'devel', 'lib')
if os.path.exists(devel_lib_path):
    sys.path.insert(0, devel_lib_path)
sys.path.append(z1_sdk_path+"/lib")

# Try to import the module
try:
    import unitree_arm_interface
except ImportError as e:
    print(f"Error importing unitree_arm_interface: {e}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Looking in: {devel_lib_path}")
    if os.path.exists(devel_lib_path):
        print(f"Files in devel/lib: {[f for f in os.listdir(devel_lib_path) if 'unitree' in f]}")
    print(f"Also looking in: {z1_sdk_path}/lib")
    if os.path.exists(z1_sdk_path+"/lib"):
        print(f"Files in source lib: {os.listdir(z1_sdk_path+'/lib')}")
    raise


'''
example_joint_ctrl.py
An example showing how to control joint velocity and position in python.
Run `roslaunch z1_bringup sim_ctrl.launch` for simulation or 
`roslaunch z1_bringup real_ctrl.launch` for real robot first.

Usage:
    python3 example_joint_ctrl.py [robot_ip]
    
    robot_ip: IP address of the controller (default: 127.0.0.1 for simulation)
              Use 192.168.123.110 for real robot (or your robot's IP)
'''

# Get robot IP from command line or use default
robot_ip = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
print(f"Connecting to robot at {robot_ip}")

# Connect to z1_controller
z1 =  unitree_arm_interface.UnitreeArm(robot_ip, 0)
z1.init()

# Joint velocity control
jntSpeed = 0.5
dq = [jntSpeed, 0., 0., 0., 0., 0.]
for i in range(0, 400):
    z1.armCmd.mode = unitree_arm_interface.ArmMode.JointSpeedCtrl
    z1.armCmd.dq_d = [jntSpeed, 0, 0, 0, 0, 0] 
    # It cannot modify the elements of a bound C++ std::array using direct index assignment.
    # z1.armCmd.dq_d[0] = 0.5
    z1.sendRecv()
    time.sleep(z1.dt)

# Joint position control
# Get current desired q
q_d = z1.armState.q_d
for i in range(0, 400):
    z1.armCmd.mode = unitree_arm_interface.ArmMode.JointPositionCtrl
    q_d[0] -= jntSpeed * z1.dt
    z1.armCmd.q_d = q_d
    z1.sendRecv()
    time.sleep(z1.dt)

# Switch to Passive mode and maintain connection
print("Switching to Passive mode...")
z1.armCmd.mode = unitree_arm_interface.ArmMode.Passive
# Send multiple Passive commands to ensure mode change is received
for i in range(10):
    z1.sendRecv()
    time.sleep(z1.dt)

print("Control loop finished. Maintaining connection...")
print("Press Ctrl+C to exit and close connection.")

# Keep connection alive until user exits
try:
    while True:
        z1.armCmd.mode = unitree_arm_interface.ArmMode.Passive
        z1.sendRecv()
        time.sleep(z1.dt)
except KeyboardInterrupt:
    print("\nExiting... Sending final Passive command...")
    # Send a few more commands before exiting to ensure clean shutdown
    for i in range(5):
        z1.armCmd.mode = unitree_arm_interface.ArmMode.Passive
        z1.sendRecv()
        time.sleep(z1.dt)
    print("Connection closed.")
