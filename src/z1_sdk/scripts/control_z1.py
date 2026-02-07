#!/usr/bin/python3
"""
Z1 Robot Arm Controller - Main Entry Point

A modular controller for the Unitree Z1 robot arm with multiple control modes:
- Joint position control
- Joint velocity control  
- Cartesian position control
- Cartesian velocity control
- Trajectory following
- Keyboard control

Usage:
    python3 control_z1.py [--ip IP] [--mode MODE] [--ros]
    
    --ip:   Controller IP (default: 127.0.0.1 for simulation)
    --mode: Initial mode (passive, keyboard, ros)
    --ros:  Enable ROS interface for external control
"""

import sys
import os
import time
import argparse
import numpy as np
import math
import select
import termios
import tty
from threading import Thread, Lock

# Add module path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from z1_core import Z1ArmInterface, Z1Kinematics, ControlMode, ControlModeManager
from z1_core.trajectory import generate_min_jerk_trajectory


# ==================== Keyboard Control ====================

KEYBOARD_HELP = """
======================================================================
               Z1 Robot Arm Keyboard Controller                   
======================================================================
  CONTROL MODES:                                                  
    1 - Joint Velocity Mode                                       
    2 - Cartesian Velocity Mode                                   
    3 - Joint Position Mode                                       
    4 - Cartesian Position Mode                                   
    0 - Passive Mode (release control)                            
                                                                  
  JOINT VELOCITY MODE (1):                                        
    q/a - Joint 1 (+/-)     w/s - Joint 2 (+/-)                  
    e/d - Joint 3 (+/-)     r/f - Joint 4 (+/-)                  
    t/g - Joint 5 (+/-)     y/z - Joint 6 (+/-)                  
                                                                  
  CARTESIAN MODE (2):                                             
    i/k - Move X (+/-)      j/l - Move Y (+/-)                   
    u/o - Move Z (+/-)                                           
                                                                  
  GRIPPER:                                                        
    [ - Open gripper        ] - Close gripper                    
                                                                  
  SPECIAL:                                                        
    h - Go to Home position                                       
    p - Print current state                                       
    c - Clear errors                                              
    SPACE - Emergency stop                                        
    ESC/x - Exit                                                  
                                                                  
  SPEED:                                                          
    + - Increase speed      - - Decrease speed                   
======================================================================
"""


class KeyboardController:
    
    def __init__(self, control_manager: ControlModeManager):
        self.manager = control_manager
        self.arm = control_manager.arm
        self.kinematics = control_manager.kinematics
        
        self.running = True
        self.speed = 0.5
        self.current_mode = ControlMode.JOINT_VELOCITY
        
        self.joint_vel = np.zeros(6)
        self.cart_vel = np.zeros(3)
        self.gripper_target = 0.0
        
        self._old_settings = None
    
    def _setup_terminal(self):
        """Setup terminal for raw input"""
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
    
    def _restore_terminal(self):
        """Restore terminal settings"""
        if self._old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
    
    def _get_key(self) -> str:
        """Non-blocking key read"""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return ''
    
    def _handle_key(self, key: str):
        """Process keyboard input"""
        if not key:
            return
        
        # Mode switching
        if key == '0':
            self.current_mode = ControlMode.PASSIVE
            self.manager.set_mode(ControlMode.PASSIVE)
            print("\n[Mode] PASSIVE")
        elif key == '1':
            self.current_mode = ControlMode.JOINT_VELOCITY
            print("\n[Mode] JOINT VELOCITY")
        elif key == '2':
            self.current_mode = ControlMode.CARTESIAN_VELOCITY
            print("\n[Mode] CARTESIAN VELOCITY")
        elif key == '3':
            self.current_mode = ControlMode.JOINT_POSITION
            print("\n[Mode] JOINT POSITION")
        elif key == '4':
            self.current_mode = ControlMode.CARTESIAN_POSITION
            print("\n[Mode] CARTESIAN POSITION")
        
        # Joint velocity control
        elif self.current_mode == ControlMode.JOINT_VELOCITY:
            if key == 'q': self.joint_vel[0] = self.speed
            elif key == 'a': self.joint_vel[0] = -self.speed
            elif key == 'w': self.joint_vel[1] = self.speed
            elif key == 's': self.joint_vel[1] = -self.speed
            elif key == 'e': self.joint_vel[2] = self.speed
            elif key == 'd': self.joint_vel[2] = -self.speed
            elif key == 'r': self.joint_vel[3] = self.speed
            elif key == 'f': self.joint_vel[3] = -self.speed
            elif key == 't': self.joint_vel[4] = self.speed
            elif key == 'g': self.joint_vel[4] = -self.speed
            elif key == 'y': self.joint_vel[5] = self.speed
            elif key == 'z': self.joint_vel[5] = -self.speed
        
        # Cartesian velocity control
        elif self.current_mode == ControlMode.CARTESIAN_VELOCITY:
            if key == 'i': self.cart_vel[0] = 0.1 * self.speed
            elif key == 'k': self.cart_vel[0] = -0.1 * self.speed
            elif key == 'j': self.cart_vel[1] = 0.1 * self.speed
            elif key == 'l': self.cart_vel[1] = -0.1 * self.speed
            elif key == 'u': self.cart_vel[2] = 0.1 * self.speed
            elif key == 'o': self.cart_vel[2] = -0.1 * self.speed
        
        # Gripper
        if key == '[':
            self.gripper_target = max(-1.0, self.gripper_target - 0.1)
            self.arm.set_gripper(self.gripper_target)
        elif key == ']':
            self.gripper_target = min(0.0, self.gripper_target + 0.1)
            self.arm.set_gripper(self.gripper_target)
        
        # Special commands
        if key == 'h':
            print("\n[Action] Going home...")
            self.manager.go_home()
            self.current_mode = ControlMode.HOME
        elif key == 'p':
            self._print_state()
        elif key == 'c':
            print("\n[Action] Clearing errors...")
            self.arm.clear_errors()
        elif key == ' ':
            print("\n[EMERGENCY STOP]")
            self.joint_vel = np.zeros(6)
            self.cart_vel = np.zeros(3)
            self.manager.stop()
        elif key == '+' or key == '=':
            self.speed = min(2.0, self.speed + 0.1)
            print(f"\n[Speed] {self.speed:.1f}")
        elif key == '-' or key == '_':
            self.speed = max(0.1, self.speed - 0.1)
            print(f"\n[Speed] {self.speed:.1f}")
        elif key == '\x1b' or key == 'x':
            print("\n[Exit] Shutting down...")
            self.running = False
    
    def _print_state(self):
        """Print current robot state"""
        q = self.arm.joint_positions
        q_deg = np.degrees(q)
        pos, _ = self.kinematics.forward_kinematics(q)
        
        print("\n" + "="*50)
        print("CURRENT STATE:")
        print("-"*50)
        print(f"Joint Positions (deg): {[f'{x:.1f}' for x in q_deg]}")
        print(f"Joint Positions (rad): {[f'{x:.3f}' for x in q]}")
        print(f"End-Effector Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m")
        print(f"Gripper: {self.arm.gripper_position:.2f}")
        print(f"Mode: {self.current_mode.name}")
        print(f"Speed: {self.speed:.1f}")
        print("="*50)
    
    def run(self):
        """Main keyboard control loop"""
        print(KEYBOARD_HELP)
        print("\nConnecting to robot...")
        
        if not self.arm.connect():
            print("Failed to connect! Make sure z1_controller is running.")
            return
        
        print("Connected! Starting keyboard control...")
        print("Press 'p' to print state, SPACE for emergency stop, ESC to exit")
        
        self._setup_terminal()
        
        try:
            while self.running:
                key = self._get_key()
                
                self.joint_vel = np.zeros(6)
                self.cart_vel = np.zeros(3)
                
                self._handle_key(key)
                
                if self.current_mode == ControlMode.JOINT_VELOCITY:
                    if np.any(np.abs(self.joint_vel) > 0.001):
                        self.manager.set_joint_velocity(self.joint_vel)
                    else:
                        self.arm.stop()
                
                elif self.current_mode == ControlMode.CARTESIAN_VELOCITY:
                    if np.any(np.abs(self.cart_vel) > 0.001):
                        self.manager.set_cartesian_velocity(self.cart_vel)
                    else:
                        self.arm.stop()
                
                self.manager.update()
                self.arm.send_recv()
                
                time.sleep(self.arm.dt)
        
        except KeyboardInterrupt:
            print("\n[Interrupted]")
        
        finally:
            self._restore_terminal()
            self.arm.shutdown()


# ==================== ROS Interface ====================

class ROSController:
    """ROS interface for external control"""
    
    def __init__(self, control_manager: ControlModeManager):
        self.manager = control_manager
        self.arm = control_manager.arm
        self.kinematics = control_manager.kinematics
        
        self.running = True
        self.lock = Lock()
        
        import rospy
        from geometry_msgs.msg import Vector3, Point, Twist
        from std_msgs.msg import Int32, String, Float64MultiArray
        
        self.rospy = rospy
        
        self._target_joint_vel = np.zeros(6)
        self._target_cart_vel = np.zeros(3)
        self._gripper_cmd = 0.0
        self._mode_cmd = None
    
    def _velocity_callback(self, msg):
        with self.lock:
            self._target_cart_vel = np.array([msg.x, msg.y, msg.z])
    
    def _joint_vel_callback(self, msg):
        with self.lock:
            self._target_joint_vel = np.array(msg.data[:6])
    
    def _gripper_callback(self, msg):
        with self.lock:
            self._gripper_cmd = float(msg.data) / 100.0
    
    def _mode_callback(self, msg):
        with self.lock:
            self._mode_cmd = msg.data.lower()
    
    def run(self):
        from geometry_msgs.msg import Vector3
        from std_msgs.msg import Int32, String, Float64MultiArray
        
        self.rospy.init_node('z1_controller', anonymous=True)
        
        self.rospy.Subscriber('/z1/velocitycmd', Vector3, self._velocity_callback)
        self.rospy.Subscriber('/z1/joint_velocitycmd', Float64MultiArray, self._joint_vel_callback)
        self.rospy.Subscriber('/z1/grippercmd', Int32, self._gripper_callback)
        self.rospy.Subscriber('/z1/modecmd', String, self._mode_callback)
        
        self.state_pub = self.rospy.Publisher('/z1/state', String, queue_size=10)
        self.joint_pub = self.rospy.Publisher('/z1/joint_positions', Float64MultiArray, queue_size=10)
        self.ee_pub = self.rospy.Publisher('/z1/ee_position', Vector3, queue_size=10)
        
        print("ROS Controller initialized")
        print("Connecting to robot...")
        
        if not self.arm.connect():
            print("Failed to connect! Make sure z1_controller is running.")
            return
        
        print("Connected! ROS control active.")
        print("Topics: /z1/velocitycmd, /z1/joint_velocitycmd, /z1/grippercmd, /z1/modecmd")
        
        rate = self.rospy.Rate(1.0 / self.arm.dt)
        
        try:
            while not self.rospy.is_shutdown() and self.running:
                with self.lock:
                    if self._mode_cmd:
                        self._handle_mode_cmd(self._mode_cmd)
                        self._mode_cmd = None
                    
                    if np.any(np.abs(self._target_cart_vel) > 0.001):
                        self.manager.set_cartesian_velocity(self._target_cart_vel)
                    elif np.any(np.abs(self._target_joint_vel) > 0.001):
                        self.manager.set_joint_velocity(self._target_joint_vel)
                    else:
                        self.arm.stop()
                    
                    self.arm.set_gripper(self._gripper_cmd)
                
                self.manager.update()
                self.arm.send_recv()
                self._publish_state()
                
                rate.sleep()
        
        except KeyboardInterrupt:
            print("\n[Interrupted]")
        
        finally:
            self.arm.shutdown()
    
    def _handle_mode_cmd(self, cmd: str):
        if cmd == 'passive':
            self.manager.set_mode(ControlMode.PASSIVE)
        elif cmd == 'home':
            self.manager.go_home()
        elif cmd == 'stop':
            self.manager.stop()
        elif cmd == 'clear':
            self.arm.clear_errors()
    
    def _publish_state(self):
        from geometry_msgs.msg import Vector3
        from std_msgs.msg import String, Float64MultiArray
        import json
        
        joint_msg = Float64MultiArray()
        joint_msg.data = list(self.arm.joint_positions)
        self.joint_pub.publish(joint_msg)
        
        pos, _ = self.kinematics.forward_kinematics(self.arm.joint_positions)
        ee_msg = Vector3(x=pos[0], y=pos[1], z=pos[2])
        self.ee_pub.publish(ee_msg)
        
        state = self.manager.get_status()
        self.state_pub.publish(json.dumps(state))


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='Z1 Robot Arm Controller')
    parser.add_argument('--ip', type=str, default='127.0.0.1',
                        help='Controller IP (default: 127.0.0.1)')
    parser.add_argument('--mode', type=str, default='keyboard',
                        choices=['keyboard', 'ros', 'passive'],
                        help='Control mode (default: keyboard)')
    parser.add_argument('--ros', action='store_true',
                        help='Enable ROS interface')
    
    args = parser.parse_args()
    
    arm = Z1ArmInterface(controller_ip=args.ip)
    manager = ControlModeManager(arm)
    
    if args.mode == 'ros' or args.ros:
        controller = ROSController(manager)
    else:
        controller = KeyboardController(manager)
    
    controller.run()


if __name__ == '__main__':
    main()
