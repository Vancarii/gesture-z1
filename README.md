# Z1 Gesture Control - LeapC + ML + MoveIt Pipeline

Control a Unitree Z1 robotic arm using hand gestures detected by an Ultraleap (Leap Motion) controller, classified by a CNN, and executed via MoveIt.

## Overview

```
LeapC hand tracking -> CNN gesture classification -> action mapping -> MoveIt Cartesian planning -> Z1 arm moves
```

Everything runs from a single `roslaunch` command.

## Prerequisites

- **ROS Noetic** on Ubuntu 20.04
- **Unitree Z1** arm (for real hardware mode)
- **Ultraleap Gemini** hand tracking service (`leapd` running)
- **Anaconda/Miniconda** with the `gesture-env` environment

## Setup (new machine)

```bash
# 1. Recreate the conda environment
conda env create -f gesture-env.yml

# 2. Start the Ultraleap tracking daemon
sudo leapd

# 3. Build the workspace
source /opt/ros/noetic/setup.bash
cd z1_ws && catkin_make
source devel/setup.bash
```

## Quick Start

```bash
# Source workspace (conda does NOT need to be deactivated)
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

## Gesture -> Arm Movement Mapping

| Gesture | Arm Movement |
|---------|-------------|
| point right | Forward (+X) |
| point left | Backward (-X) |
| swipe towards / point forward | Left (+Y) |
| swipe back / point back | Right (-Y) |
| point up / move hand up | Up (+Z) |
| point down / move hand down | Down (-Z) |
| pull back | Gripper close |
| background / pause | No movement |

Gestures below 85% confidence are ignored.

## Files (`z1_leap_integration` package)

| File | Description |
|------|-------------|
| `scripts/gesture_detector_ml.py` | ROS node - LeapC + CNN gesture classifier. Publishes JSON actions to `/leap/gesture`. Runs inside `gesture-env` via `conda run`. |
| `scripts/gesture_moveit_controller.py` | ROS node - subscribes to `/leap/gesture`, transforms velocity to robot frame, executes Cartesian moves via MoveIt. |
| `config/gesture_config.yaml` | Tuneable parameters: step size, velocity scale, confidence threshold, workspace bounds. |
| `launch/simulation_full.launch` | Single launch for simulation (Gazebo + gesture pipeline). |
| `launch/real_full.launch` | Single launch for real Z1 hardware + gesture pipeline. |
| `gesture-env.yml` | Conda environment export - recreate with `conda env create -f gesture-env.yml`. |

## Architecture

The CNN model (in `Z1-LeapC/src/architecture/`) classifies 90-frame sliding windows of 19-dimensional hand features into 13 gesture classes. A `GestureActionMap` converts gestures to directional velocities (replicating the converged PPO policy from `Z1-LeapC/src/policy.py`, without needing `stable_baselines3`). The MoveIt controller applies a coordinate transform from "Brain space" to the robot base frame and executes small Cartesian steps (4 cm in sim, 3 cm on real hardware).
