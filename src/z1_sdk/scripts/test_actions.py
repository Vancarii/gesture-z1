#!/usr/bin/python3
"""
Z1 Action Tester - Simple keyboard control to test robot actions

Press keys to trigger robot actions. No ROS, no ML, just direct control.

Usage:
    python3 test_actions.py [--ip 127.0.0.1]
"""

import sys
import os
import time
import argparse
import select
import termios
import tty
import numpy as np

sys.path.insert(0, '/home/tangentlab/z1_ws/devel/lib')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from z1_core import Z1ArmInterface, Z1Kinematics, ControlMode, ControlModeManager


HELP = """
================================================================================
                      Z1 ACTION TESTER
================================================================================
  MOVEMENT:
    1 - Approach (forward)      2 - Retreat (backward)
    3 - Move Left               4 - Move Right
    5 - Move Up                 6 - Move Down
    
  BEHAVIORS:
    w - Wave                    h - Home position
    y - Nod Yes                 n - Nod No
    
  GRIPPER:
    g - Grab (close)            o - Open (release)
    
  CONTROL:
    0/s - Stop                  p - Pause (hold position)
    SPACE - Emergency stop      q - Quit
    
  INFO:
    i - Print current state
    +/- - Adjust speed
================================================================================
"""


class ActionTester:
    
    def __init__(self, controller_ip: str = "127.0.0.1"):
        self.arm = Z1ArmInterface(controller_ip)
        self.manager = ControlModeManager(self.arm)
        self.kinematics = Z1Kinematics()
        
        self.running = True
        self.hertz = 50
        self.cart_speed = 0.1
        self.joint_speed = 1.0
        
        # Animation states
        self._wave_phase = 0
        self._wave_active = False
        self._nod_phase = 0
        self._nod_type = None
        
        self._old_settings = None
    
    def _setup_terminal(self):
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
    
    def _restore_terminal(self):
        if self._old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
    
    def _get_key(self) -> str:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1).lower()
        return ''
    
    def _print_state(self):
        q = self.arm.joint_positions
        pos, _ = self.kinematics.forward_kinematics(q)
        print("\n" + "="*50)
        print(f"EE Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m")
        print(f"Joints (deg): {[f'{np.degrees(x):.1f}' for x in q]}")
        print(f"Gripper: {self.arm.gripper_position:.2f}")
        print(f"Speed: {self.cart_speed:.2f} m/s")
        print("="*50)
    
    def do_approach(self):
        print("[Action] Approach")
        self.manager.set_cartesian_velocity(np.array([self.cart_speed, 0, 0]))
    
    def do_retreat(self):
        print("[Action] Retreat")
        self.manager.set_cartesian_velocity(np.array([-self.cart_speed, 0, 0]))
    
    def do_left(self):
        print("[Action] Move Left")
        self.manager.set_cartesian_velocity(np.array([0, self.cart_speed, 0]))
    
    def do_right(self):
        print("[Action] Move Right")
        self.manager.set_cartesian_velocity(np.array([0, -self.cart_speed, 0]))
    
    def do_up(self):
        print("[Action] Move Up")
        self.manager.set_cartesian_velocity(np.array([0, 0, self.cart_speed]))
    
    def do_down(self):
        print("[Action] Move Down")
        self.manager.set_cartesian_velocity(np.array([0, 0, -self.cart_speed]))
    
    def do_stop(self):
        print("[Action] Stop")
        self.manager.stop()
        self._wave_active = False
        self._nod_type = None
    
    def do_home(self):
        print("[Action] Go Home")
        self.manager.go_home()
    
    def do_pause(self):
        print("[Action] Pause (hold)")
        self.manager.set_joint_position(self.arm.joint_positions)
    
    def do_wave(self):
        print("[Action] Wave")
        self._wave_active = True
        self._wave_phase = 0
    
    def do_nod_yes(self):
        print("[Action] Nod Yes")
        self._nod_type = 'yes'
        self._nod_phase = 0
    
    def do_nod_no(self):
        print("[Action] Nod No")
        self._nod_type = 'no'
        self._nod_phase = 0
    
    def do_grab(self):
        print("[Action] Grab")
        self.arm.set_gripper(0.0)
    
    def do_release(self):
        print("[Action] Release")
        self.arm.set_gripper(-1.0)
    
    def _update_animations(self):
        # Wave animation
        if self._wave_active:
            self._wave_phase += 0.15
            if self._wave_phase > 4 * np.pi:
                self._wave_active = False
                self.manager.stop()
                print("[Wave complete]")
            else:
                vel = np.zeros(6)
                vel[5] = self.joint_speed * 1.5 * np.sin(self._wave_phase * 2)
                self.manager.set_joint_velocity(vel)
        
        # Nod animation
        if self._nod_type:
            self._nod_phase += 0.1
            if self._nod_phase > 2 * np.pi:
                self._nod_type = None
                self.manager.stop()
                print("[Nod complete]")
            else:
                vel = np.zeros(6)
                if self._nod_type == 'yes':
                    vel[3] = self.joint_speed * 0.8 * np.sin(self._nod_phase * 3)
                else:
                    vel[0] = self.joint_speed * 0.6 * np.sin(self._nod_phase * 3)
                self.manager.set_joint_velocity(vel)
    
    def _handle_key(self, key: str):
        if not key:
            return
        
        # Movement
        if key == '1': self.do_approach()
        elif key == '2': self.do_retreat()
        elif key == '3': self.do_left()
        elif key == '4': self.do_right()
        elif key == '5': self.do_up()
        elif key == '6': self.do_down()
        
        # Behaviors
        elif key == 'w': self.do_wave()
        elif key == 'h': self.do_home()
        elif key == 'y': self.do_nod_yes()
        elif key == 'n': self.do_nod_no()
        
        # Gripper
        elif key == 'g': self.do_grab()
        elif key == 'o': self.do_release()
        
        # Control
        elif key == '0' or key == 's': self.do_stop()
        elif key == 'p': self.do_pause()
        elif key == ' ': 
            print("[EMERGENCY STOP]")
            self.do_stop()
        
        # Info
        elif key == 'i': self._print_state()
        elif key == '+' or key == '=':
            self.cart_speed = min(0.3, self.cart_speed + 0.02)
            print(f"[Speed: {self.cart_speed:.2f} m/s]")
        elif key == '-' or key == '_':
            self.cart_speed = max(0.02, self.cart_speed - 0.02)
            print(f"[Speed: {self.cart_speed:.2f} m/s]")
        
        # Quit
        elif key == 'q':
            print("\n[Quit]")
            self.running = False
    
    def run(self):
        print(HELP)
        
        print("Connecting to robot...")
        if not self.arm.connect():
            print("Failed to connect! Make sure z1_controller is running.")
            return
        print("Connected! Press keys to test actions.\n")
        
        self._setup_terminal()
        
        try:
            while self.running:
                key = self._get_key()
                self._handle_key(key)
                
                # Update animations
                self._update_animations()
                
                # Update robot
                self.manager.update()
                self.arm.send_recv()
                
                time.sleep(1.0 / self.hertz)
        
        except KeyboardInterrupt:
            print("\n[Interrupted]")
        
        finally:
            self._restore_terminal()
            self.arm.shutdown()
            print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Z1 Action Tester')
    parser.add_argument('--ip', type=str, default='127.0.0.1',
                        help='Controller IP (default: 127.0.0.1)')
    args = parser.parse_args()
    
    tester = ActionTester(args.ip)
    tester.run()


if __name__ == "__main__":
    main()

