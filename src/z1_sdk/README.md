# Z1 SDK Python Interface

This package provides a Python interface for controlling the Unitree Z1 robot arm.

## Building

The package should build automatically with `catkin_make`. If you encounter Python import errors, ensure you're using the system Python (not Anaconda):

```bash
cd ~/z1_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

## Usage

### Simulation

1. **Start the simulation controller:**
   ```bash
   roslaunch z1_bringup sim_ctrl.launch
   ```
   Or start the full simulation with MoveIt:
   ```bash
   roslaunch z1_bringup sim_arm.launch
   ```

2. **Run Python examples:**
   ```bash
   # In a new terminal
   source ~/z1_ws/devel/setup.bash
   rosrun z1_sdk example_joint_ctrl.py
   # or
   rosrun z1_sdk example_lowcmd.py
   ```

### Real Robot

1. **Network Setup:**
   - Connect your PC to the robot via Ethernet cable
   - The robot's default IP address is `192.168.123.110`
   - Ensure your PC is on the same network subnet

2. **Start the real robot controller:**
   ```bash
   roslaunch z1_bringup real_ctrl.launch
   ```
   Or start the full setup with MoveIt:
   ```bash
   roslaunch z1_bringup real_arm.launch
   ```

3. **Modify Python scripts for real robot:**
   
   In the example scripts, change the IP address from `127.0.0.1` to the robot's IP:
   
   ```python
   # For simulation (localhost)
   z1 = unitree_arm_interface.UnitreeArm("127.0.0.1", 0)
   
   # For real robot (default IP)
   z1 = unitree_arm_interface.UnitreeArm("192.168.123.110", 0)
   ```
   
   Or use a command-line argument:
   ```python
   import sys
   robot_ip = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
   z1 = unitree_arm_interface.UnitreeArm(robot_ip, 0)
   ```

4. **Run Python examples with real robot:**
   ```bash
   source ~/z1_ws/devel/setup.bash
   # Edit the script to use robot IP, or modify as shown above
   python3 ~/z1_ws/src/z1_sdk/scripts/example_joint_ctrl.py
   ```

## Configuration

### Controller Configuration

The controller configuration is in `z1_controller/config/config.yaml`:

- `udp/mcu_ip`: Robot MCU IP address (default: `192.168.123.110`)
- `udp/port_to_mcu`: Port to MCU (default: `8881`)
- `udp/port_to_sdk`: Port for SDK communication (default: `8871`)

### Hardware Interface Configuration

For ROS control interface, see `z1_hw/config/config.yaml`:

- `udp_to_controller/controller_ip`: Controller IP (use `127.0.0.1` for local, robot IP for real hardware)
- `udp_to_controller/own_port`: Local port for SDK (default: `8872`)

## Python API

The Python interface provides the following main classes:

- `UnitreeArm(controllerIP, ownPort=0, toPort=8871)`: Main arm interface
- `ArmMode`: Enum for control modes (Passive, LowCmd, JointSpeedCtrl, JointPositionCtrl, etc.)
- `ArmCmd`: Command structure for sending commands
- `ArmState`: State structure for reading robot state

Example usage:
```python
import unitree_arm_interface

# Connect to controller
z1 = unitree_arm_interface.UnitreeArm("127.0.0.1", 0)
z1.init()

# Joint position control
z1.armCmd.mode = unitree_arm_interface.ArmMode.JointPositionCtrl
z1.armCmd.q_d = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
z1.sendRecv()

# Read state
current_position = z1.armState.q
```

## Connection Management

The example scripts maintain the UDP connection to the controller after completing their control sequences. This prevents "Lose connection with UDP sdk" errors. The scripts will:

1. Complete the control sequence (joint velocity/position control or low-level commands)
2. Switch to Passive mode
3. Continue sending Passive commands to maintain the connection
4. Wait for you to press `Ctrl+C` to exit

This ensures a clean shutdown and prevents connection timeouts. If you want the script to exit automatically after the control sequence, you can modify the scripts to remove the connection maintenance loop.

## Troubleshooting

### "Lose connection with UDP sdk" errors

This error occurs when the UDP connection times out (after ~1 second without communication). The updated example scripts now maintain the connection automatically. If you see this error:

1. **Script exits too quickly**: The scripts now maintain the connection until you press Ctrl+C
2. **Network issues**: Check network connectivity and firewall settings
3. **Controller not running**: Ensure `roslaunch z1_bringup sim_ctrl.launch` or `real_ctrl.launch` is running
4. **Wrong IP address**: Verify the robot IP address matches your network configuration

### ModuleNotFoundError: No module named 'unitree_arm_interface'

This usually means:
1. The package wasn't built correctly - run `catkin_make` again
2. You're using the wrong Python interpreter (Anaconda instead of system Python)
   - Solution: Use `/usr/bin/python3` explicitly or ensure system Python is first in PATH
3. The module path isn't set correctly
   - Solution: Always source `devel/setup.bash` before running scripts

### Connection failed errors

- **Simulation**: Make sure `roslaunch z1_bringup sim_ctrl.launch` is running
- **Real robot**: 
  - Check network connection to robot
  - Verify robot IP address matches configuration
  - Ensure robot is powered on and controller is running
  - Check firewall settings

## Notes

- The scripts use `/usr/bin/python3` explicitly to avoid Anaconda Python conflicts
- Always source the workspace setup file: `source ~/z1_ws/devel/setup.bash`
- For real robot, ensure your PC and robot are on the same network subnet

