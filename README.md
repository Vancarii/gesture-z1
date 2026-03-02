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
| `scripts/detect_point.py` | ROS node - LeapC + CNN gesture classifier. Publishes JSON actions to `/leap/gesture`. Runs inside `gesture-env` via `conda run`. |
| `scripts/look_at_point.py` | ROS node - subscribes to `/leap/gesture`, transforms velocity to robot frame, executes Cartesian moves via MoveIt. |
| `config/gesture_config.yaml` | Tuneable parameters: step size, velocity scale, confidence threshold, workspace bounds. |
| `launch/simulation_full.launch` | Single launch for simulation (Gazebo + gesture pipeline). |
| `launch/real_full.launch` | Single launch for real Z1 hardware + gesture pipeline. |
| `gesture-env.yml` | Conda environment export - recreate with `conda env create -f gesture-env.yml`. |
