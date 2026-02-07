#!/usr/bin/python3
"""
Z1 Gesture Data Collector

Collects gesture-action pairs for training an MLP for human-robot interaction.

Gestures (input features):
    - Directional: swipe_back, swipe_towards, point_left, point_right, point_up, point_down
    - Control: pull_back, pause, wave, open_hand, closed_fist, thumbs_up, thumbs_down
    
Actions (target labels):
    - Movement: approach, retreat, move_left, move_right, move_up, move_down
    - Behaviors: wave, stop, home, pause, grab, release

Usage:
    python3 collector.py [--output FILE]
    
    Keys during collection:
        r - Toggle recording
        q - Quit and save
        0-9, a-z - Trigger actions (see HELP below)
"""

import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
import json
from datetime import datetime

# ROS imports
import rospy
from std_msgs.msg import Int32, String, Float32
from geometry_msgs.msg import Vector3

from z1_core import Z1ArmInterface, Z1Kinematics, ControlMode, ControlModeManager


# ==================== Gesture & Action Definitions ====================

GESTURE_CLASSES = {
    'none':           0,
    'wave':           1,
    'open_hand':      2,
    'closed_fist':    3,
    'thumbs_up':      4,
    'thumbs_down':    5,
    'point_up':       6,
    'point_down':     7,
    'point_left':     8,
    'point_right':    9,
    'swipe_back':    10,
    'swipe_towards': 11,
    'pull_back':     12,
    'pause':         13,
    'peace':         14,
    'ok':            15,
}

ACTION_CLASSES = {
    'idle':       0,
    'approach':   1,
    'retreat':    2,
    'move_left':  3,
    'move_right': 4,
    'move_up':    5,
    'move_down':  6,
    'wave':       7,
    'stop':       8,
    'home':       9,
    'pause':     10,
    'grab':      11,
    'release':   12,
    'nod_yes':   13,
    'nod_no':    14,
}

# Keyboard to action mapping
KEY_ACTION_MAP = {
    '0': 'idle',
    '1': 'approach',
    '2': 'retreat',
    '3': 'move_left',
    '4': 'move_right',
    '5': 'move_up',
    '6': 'move_down',
    'w': 'wave',
    's': 'stop',
    'h': 'home',
    'p': 'pause',
    'g': 'grab',
    'o': 'release',
    'y': 'nod_yes',
    'n': 'nod_no',
}

HELP_TEXT = """
================================================================================
                        Z1 Gesture Data Collector
================================================================================
CONTROLS:
    r - Toggle recording on/off
    q - Quit and save data
    
ACTION KEYS:
    0 - Idle (stop)           1 - Approach (move forward)
    2 - Retreat (move back)   3 - Move Left
    4 - Move Right            5 - Move Up
    6 - Move Down             w - Wave gesture
    s - Stop                  h - Go Home
    p - Pause (hold)          g - Grab (close gripper)
    o - Open (release grip)   y - Nod Yes
    n - Nod No
    
RECORDING:
    When recording is ON, each frame captures:
    - Current detected gesture (from /z1/gesture topic)
    - Gesture confidence
    - Previous robot action
    - Current action you trigger
    
    This creates training pairs: (gesture, prev_action) -> target_action
================================================================================
"""


class GestureDataCollector:
    
    def __init__(self, output_dir: str = None):
        # Output setup
        if output_dir is None:
            output_dir = os.path.expanduser("~/z1_ws/src/z1_sdk/scripts/data")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(output_dir, f"gesture_data_{timestamp}.csv")
        self.metadata_file = os.path.join(output_dir, f"gesture_data_{timestamp}_meta.json")
        
        # Robot interface
        self.arm = Z1ArmInterface()
        self.manager = ControlModeManager(self.arm)
        self.kinematics = Z1Kinematics()
        
        # State tracking
        self.current_gesture = "none"
        self.current_confidence = 0.0
        self.current_phase = 0
        self.gesture_velocity = np.zeros(3)  # For swipe detection
        
        # Recording state
        self.is_recording = False
        self.buffer = []
        self.prev_action = 'idle'
        self.prev_action_id = 0
        
        # Control parameters
        self.hertz = 10
        self.cart_speed = 0.08  # m/s for Cartesian moves
        self.joint_speed = 0.8  # rad/s for joint moves
        
        # Wave animation state
        self._wave_phase = 0
        self._wave_active = False
        
        # Nod animation state
        self._nod_phase = 0
        self._nod_type = None  # 'yes' or 'no'
    
    # ==================== ROS Callbacks ====================
    
    def _gesture_callback(self, msg):
        gesture = msg.data.lower().replace(' ', '_')
        if gesture in GESTURE_CLASSES:
            self.current_gesture = gesture
        else:
            self.current_gesture = 'none'
    
    def _confidence_callback(self, msg):
        self.current_confidence = msg.data
    
    def _phase_callback(self, msg):
        self.current_phase = msg.data
    
    def _velocity_callback(self, msg):
        self.gesture_velocity = np.array([msg.x, msg.y, msg.z])
    
    def _setup_ros(self):
        rospy.init_node("z1_gesture_collector", anonymous=True)
        
        # Subscribe to gesture detection topics
        rospy.Subscriber("/z1/gesture", String, self._gesture_callback)
        rospy.Subscriber("/z1/conf", Float32, self._confidence_callback)
        rospy.Subscriber("/z1/phase", Int32, self._phase_callback)
        rospy.Subscriber("/z1/gesture_velocity", Vector3, self._velocity_callback)
        
        rospy.loginfo("ROS subscribers initialized")
    
    # ==================== Robot Actions ====================
    
    def execute_action(self, action: str) -> int:
        """Execute robot action and return action ID"""
        action_id = ACTION_CLASSES.get(action, 0)
        
        if action == 'idle' or action == 'stop':
            self.manager.stop()
            self._wave_active = False
            self._nod_type = None
            
        elif action == 'approach':
            self.manager.set_cartesian_velocity(np.array([self.cart_speed, 0, 0]))
            
        elif action == 'retreat':
            self.manager.set_cartesian_velocity(np.array([-self.cart_speed, 0, 0]))
            
        elif action == 'move_left':
            self.manager.set_cartesian_velocity(np.array([0, self.cart_speed, 0]))
            
        elif action == 'move_right':
            self.manager.set_cartesian_velocity(np.array([0, -self.cart_speed, 0]))
            
        elif action == 'move_up':
            self.manager.set_cartesian_velocity(np.array([0, 0, self.cart_speed]))
            
        elif action == 'move_down':
            self.manager.set_cartesian_velocity(np.array([0, 0, -self.cart_speed]))
            
        elif action == 'wave':
            self._wave_active = True
            self._wave_phase = 0
            
        elif action == 'home':
            self.manager.go_home()
            
        elif action == 'pause':
            # Hold current position
            self.manager.set_joint_position(self.arm.joint_positions)
            
        elif action == 'grab':
            self.arm.set_gripper(0.0)  # Close gripper
            
        elif action == 'release':
            self.arm.set_gripper(-1.0)  # Open gripper
            
        elif action == 'nod_yes':
            self._nod_type = 'yes'
            self._nod_phase = 0
            
        elif action == 'nod_no':
            self._nod_type = 'no'
            self._nod_phase = 0
        
        return action_id
    
    def _update_animations(self):
        """Update ongoing animations (wave, nod)"""
        # Wave animation - oscillate joint 6
        if self._wave_active:
            self._wave_phase += 0.3
            if self._wave_phase > 4 * np.pi:  # 2 full waves
                self._wave_active = False
                self.manager.stop()
            else:
                vel = np.zeros(6)
                vel[5] = 1.5 * np.sin(self._wave_phase)
                self.manager.set_joint_velocity(vel)
        
        # Nod animation
        if self._nod_type:
            self._nod_phase += 0.2
            if self._nod_phase > 2 * np.pi:
                self._nod_type = None
                self.manager.stop()
            else:
                vel = np.zeros(6)
                if self._nod_type == 'yes':
                    # Nod up/down using joint 4
                    vel[3] = 0.8 * np.sin(self._nod_phase * 2)
                else:  # 'no'
                    # Shake left/right using joint 1
                    vel[0] = 0.6 * np.sin(self._nod_phase * 2)
                self.manager.set_joint_velocity(vel)
    
    # ==================== Data Collection ====================
    
    def _get_input(self) -> str:
        """Non-blocking keyboard input"""
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1).lower()
        return None
    
    def _collect_sample(self, action: str, action_id: int):
        """Collect one data sample"""
        # Get robot state
        joint_pos = self.arm.joint_positions
        ee_pos, _ = self.kinematics.forward_kinematics(joint_pos)
        
        # Build feature vector
        sample = {
            'timestamp': time.time(),
            # Gesture features
            'gesture': self.current_gesture,
            'gesture_id': GESTURE_CLASSES.get(self.current_gesture, 0),
            'gesture_confidence': self.current_confidence,
            'gesture_phase': self.current_phase,
            'gesture_vel_x': self.gesture_velocity[0],
            'gesture_vel_y': self.gesture_velocity[1],
            'gesture_vel_z': self.gesture_velocity[2],
            # Robot state features
            'ee_x': ee_pos[0],
            'ee_y': ee_pos[1],
            'ee_z': ee_pos[2],
            'joint_0': joint_pos[0],
            'joint_1': joint_pos[1],
            'joint_2': joint_pos[2],
            'joint_3': joint_pos[3],
            'joint_4': joint_pos[4],
            'joint_5': joint_pos[5],
            'gripper': self.arm.gripper_position,
            # Previous action context
            'prev_action': self.prev_action,
            'prev_action_id': self.prev_action_id,
            # Target (label)
            'action': action,
            'action_id': action_id,
        }
        
        self.buffer.append(sample)
    
    def _save_data(self):
        """Save collected data to CSV"""
        if not self.buffer:
            print("No data to save")
            return
        
        df = pd.DataFrame(self.buffer)
        df.to_csv(self.output_file, index=False)
        print(f"\nSaved {len(df)} samples to {self.output_file}")
        
        # Save metadata
        meta = {
            'gesture_classes': GESTURE_CLASSES,
            'action_classes': ACTION_CLASSES,
            'num_samples': len(df),
            'collection_date': datetime.now().isoformat(),
            'gesture_distribution': df['gesture'].value_counts().to_dict(),
            'action_distribution': df['action'].value_counts().to_dict(),
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"Saved metadata to {self.metadata_file}")
    
    # ==================== Main Loop ====================
    
    def run(self):
        print(HELP_TEXT)
        
        # Setup ROS
        self._setup_ros()
        
        # Connect to robot
        print("Connecting to robot...")
        if not self.arm.connect():
            print("Failed to connect to robot!")
            return
        print("Connected!")
        
        # Setup terminal for raw input
        import termios, tty
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        
        try:
            print("\nReady! Press 'r' to start recording, 'q' to quit")
            
            while not rospy.is_shutdown():
                loop_start = time.time()
                
                # Get keyboard input
                key = self._get_input()
                
                if key == 'q':
                    print("\nQuitting...")
                    break
                elif key == 'r':
                    self.is_recording = not self.is_recording
                    status = "ON" if self.is_recording else "OFF"
                    print(f"\n[Recording {status}]")
                    continue
                
                # Determine action
                current_action = self.prev_action
                current_action_id = self.prev_action_id
                
                if key and key in KEY_ACTION_MAP:
                    current_action = KEY_ACTION_MAP[key]
                    current_action_id = self.execute_action(current_action)
                
                # Update animations
                self._update_animations()
                
                # Collect data if recording
                if self.is_recording:
                    self._collect_sample(current_action, current_action_id)
                    
                    # Status display
                    sys.stdout.write(
                        f"\r[REC] Samples: {len(self.buffer):5d} | "
                        f"Gesture: {self.current_gesture:15s} ({self.current_confidence:.2f}) | "
                        f"Action: {current_action:12s}"
                    )
                    sys.stdout.flush()
                
                # Update robot
                self.manager.update()
                self.arm.send_recv()
                
                # Update previous action
                self.prev_action = current_action
                self.prev_action_id = current_action_id
                
                # Maintain loop rate
                elapsed = time.time() - loop_start
                sleep_time = (1.0 / self.hertz) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\n[Interrupted]")
        
        finally:
            # Restore terminal
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            
            # Save data
            self._save_data()
            
            # Shutdown robot
            self.arm.shutdown()
            print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Z1 Gesture Data Collector')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for collected data')
    args = parser.parse_args()
    
    collector = GestureDataCollector(output_dir=args.output)
    collector.run()


if __name__ == "__main__":
    main()
