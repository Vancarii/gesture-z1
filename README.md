# Z1 Gesture Control - LeapC + MoveIt Pipeline

Lawrence Kim, Sarah Jade Pratt, Yecheng Wang, Flora Jin

Control a Unitree Z1 robotic arm using hand pointing gestures detected by an Ultraleap (Leap Motion) controller, and executed via MoveIt.

## Overview

---

To change from wifi ethernet to the real arm, go to settings and change ipv4 to manual, then set these values:
- 192.168.123.162
- 255.255.255.0
- change ipv4 back to automatic when done with the arm, will need a reboot

`
LeapC hand tracking -> publish world coordinates -> MoveIt Cartesian planning -> Z1 arm moves
`

Everything runs from a single `roslaunch` command.

## Prerequisites

- **ROS Noetic** on Ubuntu 20.04
- **Unitree Z1** arm (for real hardware mode)
- **Ultraleap Gemini** hand tracking service
- **Anaconda/Miniconda** with the `gesture-env` environment

## Setup (new machine)

```bash
# 1. Recreate the conda environment
conda env create -f gesture-env.yml

# 3. Build the workspace
source /opt/ros/noetic/setup.bash
cd z1_ws && catkin_make
source devel/setup.bash
```

## Quick Start

```bash
# Source workspace 
source /opt/ros/noetic/setup.bash
source ~/z1_ws/devel/setup.bash
```

### Simulation (Gazebo)

```bash
roslaunch z1_leap_integration simulation_full.launch
```

| Flag | Default | Description |
|------|---------|-------------|
| `use_gazebo` | `true` | Gazebo physics sim (`false` = MoveIt fake controllers) |
| `rviz` | `false` | Open RViz visualization |
| `show_viz` | `true` | Show CV2 hand skeleton window |

### Real Arm

```bash
roslaunch z1_leap_integration real_full.launch
```

| Flag | Default | Description |
|------|---------|-------------|
| `rviz` | `false` | Open RViz visualization |
| `show_viz` | `true` | Show CV2 hand skeleton window |

## Files (`z1_leap_integration` package)

| File | Description |
|------|-------------|
| `scripts/detect_point.py` | ROS node - LeapC. Publishes world coordinates of user point to `/leap/pointing_target`. Runs inside `gesture-env` |
| `scripts/look_at_point.py` | ROS node - subscribes to `/leap/pointing_target`, transforms velocity to robot frame, executes Cartesian moves via MoveIt. |
| `config/gesture_config.yaml` | Tuneable parameters: step size, velocity scale, confidence threshold, workspace bounds. |
| `launch/simulation_full.launch` | Single launch for simulation (Gazebo + gesture pipeline). |
| `launch/real_full.launch` | Single launch for real Z1 hardware + gesture pipeline. |
| `gesture-env.yml` | Conda environment export - recreate with `conda env create -f gesture-env.yml`. |


## Progress Updates

### March 10
- implement scene objects, including the ground and the frame into the scene for collision detection
- add speaker to the attached-objects of the arm 

### March 9
- printed a lot of pieces, 2 types of speaker to arm mounts, a stronger one of previous version and one new one to reduce weight of previous adapter
- printed new leap camera holder, brought a tripod from home, bolted the mount to the tripod
- wired the speaker power cord to the arm, temporary just to test it
- implemented homing (use 2 hands open)
- mounted a battery plate on the frame so the battery can sit there when moving and more weight to ground the frame

### March 4
- Tested in TASC 7000
- found a wifi adapter haha
- need a longer aux, mount the camera properly, leap camera is too low and too close to the arm, gestures arent recognized further away
- Needs to refine the end joint, mount the speaker, implement collision detection and scene objects

### February 20 
- Subscribed to world coordinates
- Sent as moveit to path plan and move the z1 arm
- Pipeline is now leap -> calculate world coordinates -> publish -> subscribe -> path plan -> move arm

### February 15
- Sarah wanted to scrap the incremental movements
- Removed all ML from pipeline, we are only using pointing first
- Implemented point recognition conversion to world coordinates

### February 8
- integrated leap all into this ROS repo with the z1
- Got a pipeline working with some of Sajan's MLP work (Leap -> ML -> publish ROS Node -> Z1 subscribes -> moves arm)

--- 

#### Previous Repo, without moveit and ROS

### Yecheng Jan 29th Update

- Added the `export ld_library_path...` command to `./bashrc` so we don't need to run it every time anymore
- downloaded vs code to the laptop
- Added a new script: `move-and-reset.py` with just 2 gestures
    - 5 fingers out: reset arm to base position
    - 1 finger out: move and rotate the arm on the x axis
- works with the arm and the simulator
NOTE: be careful when testing and dont over extend the arm 

---

## Flora & Yecheng Jan 28th Update

Before running the leapC-SDK must run 
`export LD_LIBRARY_PATH=/usr/lib/ultraleap-hand-tracking-service:$LD_LIBRARY_PATH`
in order to successfully import leap.

Was able to test the file `test-integrate.py` which moves the arm when a finger point is detected, on the real robotic arm.

--- 

## Yecheng Jan 27th Update

To run a simple example where gestures control the simulation arm:

Run `roslaunch z1_bringup sim_ctrl.launch` in terminal 1 and gazebo will open, keep this terminal running
- Files are in `Desktop/robotic-arm/`
- `test-integrate.py` puts together the gesture and the arm
    - in a second terminal, run this in the robotic-arm directory: `python test-integrate.py`
    - do a pointing hand gesture (one finger out) and watch the simulation on gazebo rotate as long as it detects your gesture
    - Note: currently it only rotates about 160 degrees or so until it stops and doesnt go anymore to protect it
    - run `reset-arm-pos.py` to reset the position of the arm (doesnt work yet) for now just kill the gazebo terminal and run it again to reset the arm simulation


#### I found stuff that Sajan was working on under `z1/z1_sdk/scripts/`
These are also copied to our repo on github too
- there are the files `test_actions.py` for keyboard controlling the arm
- `train_gesture_mlp.py`
- `gesture_controller.py` - uses trained mlp to control the z1 robot based on detected hand gestures

havent been able to test them yet - Don't think Sajan really got much gestures to work, but once Sarah is done the Evaluation Study Procedure then we will know what exact gestures we want. These files will still be beneficial as reference, especially for the ML part
