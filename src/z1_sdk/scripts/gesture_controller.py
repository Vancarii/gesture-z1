#!/usr/bin/python3
"""
Gesture-based Robot Controller

Uses trained MLP to control the Z1 robot based on detected hand gestures.

Usage:
    python3 gesture_controller.py --model gesture_mlp.pt [--ip 127.0.0.1]
"""

import os
import sys
import argparse
import time
import numpy as np

import rospy
from std_msgs.msg import Int32, String, Float32
from geometry_msgs.msg import Vector3

from z1_core import Z1ArmInterface, Z1Kinematics, ControlMode, ControlModeManager

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ==================== Gesture/Action Mappings ====================

GESTURE_CLASSES = {
    'none': 0, 'wave': 1, 'open_hand': 2, 'closed_fist': 3,
    'thumbs_up': 4, 'thumbs_down': 5, 'point_up': 6, 'point_down': 7,
    'point_left': 8, 'point_right': 9, 'swipe_back': 10, 'swipe_towards': 11,
    'pull_back': 12, 'pause': 13, 'peace': 14, 'ok': 15,
}

ACTION_NAMES = [
    'idle', 'approach', 'retreat', 'move_left', 'move_right',
    'move_up', 'move_down', 'wave', 'stop', 'home',
    'pause', 'grab', 'release', 'nod_yes', 'nod_no',
]

NUM_GESTURES = 16
NUM_ACTIONS = 15


# ==================== Model ====================

class GestureActionMLP(nn.Module):
    def __init__(self, num_gestures=16, num_actions=15, hidden_sizes=[128, 64, 32]):
        super().__init__()
        self.num_gestures = num_gestures
        self.num_actions = num_actions
        
        input_size = num_gestures + 1 + 3 + 3 + 6 + num_actions
        
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2),
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_actions))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def load_model(model_path: str):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location='cpu')
    model = GestureActionMLP(
        num_gestures=checkpoint.get('num_gestures', NUM_GESTURES),
        num_actions=checkpoint.get('num_actions', NUM_ACTIONS),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


# ==================== Controller ====================

class GestureController:
    
    def __init__(self, model_path: str, controller_ip: str = "127.0.0.1"):
        # Load model
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for inference!")
        
        self.model = load_model(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Loaded model from {model_path}")
        
        # Robot interface
        self.arm = Z1ArmInterface(controller_ip)
        self.manager = ControlModeManager(self.arm)
        self.kinematics = Z1Kinematics()
        
        # Gesture state
        self.current_gesture = "none"
        self.gesture_confidence = 0.0
        self.gesture_velocity = np.zeros(3)
        
        # Action state
        self.prev_action_id = 0
        self.current_action = 'idle'
        
        # Control parameters
        self.hertz = 10
        self.cart_speed = 0.08
        self.confidence_threshold = 0.6
        
        # Animation state
        self._wave_phase = 0
        self._wave_active = False
        self._nod_phase = 0
        self._nod_type = None
    
    def _gesture_callback(self, msg):
        gesture = msg.data.lower().replace(' ', '_')
        if gesture in GESTURE_CLASSES:
            self.current_gesture = gesture
    
    def _confidence_callback(self, msg):
        self.gesture_confidence = msg.data
    
    def _velocity_callback(self, msg):
        self.gesture_velocity = np.array([msg.x, msg.y, msg.z])
    
    def _setup_ros(self):
        rospy.init_node("z1_gesture_controller", anonymous=True)
        rospy.Subscriber("/z1/gesture", String, self._gesture_callback)
        rospy.Subscriber("/z1/conf", Float32, self._confidence_callback)
        rospy.Subscriber("/z1/gesture_velocity", Vector3, self._velocity_callback)
        
        # Publisher for predicted action
        self.action_pub = rospy.Publisher("/z1/predicted_action", String, queue_size=10)
    
    def predict_action(self) -> str:
        """Use MLP to predict action from current state"""
        # Build feature vector
        gesture_onehot = np.zeros(NUM_GESTURES, dtype=np.float32)
        gesture_id = GESTURE_CLASSES.get(self.current_gesture, 0)
        gesture_onehot[gesture_id] = 1.0
        
        prev_action_onehot = np.zeros(NUM_ACTIONS, dtype=np.float32)
        prev_action_onehot[self.prev_action_id] = 1.0
        
        joint_pos = self.arm.joint_positions
        ee_pos, _ = self.kinematics.forward_kinematics(joint_pos)
        
        features = np.concatenate([
            gesture_onehot,
            [self.gesture_confidence],
            self.gesture_velocity,
            ee_pos,
            joint_pos,
            prev_action_onehot,
        ]).astype(np.float32)
        
        # Inference
        with torch.no_grad():
            x = torch.tensor(features).unsqueeze(0).to(self.device)
            logits = self.model(x)
            action_id = logits.argmax(dim=1).item()
        
        return ACTION_NAMES[action_id], action_id
    
    def execute_action(self, action: str):
        """Execute the predicted action"""
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
            self.manager.set_joint_position(self.arm.joint_positions)
        elif action == 'grab':
            self.arm.set_gripper(0.0)
        elif action == 'release':
            self.arm.set_gripper(-1.0)
        elif action == 'nod_yes':
            self._nod_type = 'yes'
            self._nod_phase = 0
        elif action == 'nod_no':
            self._nod_type = 'no'
            self._nod_phase = 0
    
    def _update_animations(self):
        if self._wave_active:
            self._wave_phase += 0.3
            if self._wave_phase > 4 * np.pi:
                self._wave_active = False
                self.manager.stop()
            else:
                vel = np.zeros(6)
                vel[5] = 1.5 * np.sin(self._wave_phase)
                self.manager.set_joint_velocity(vel)
        
        if self._nod_type:
            self._nod_phase += 0.2
            if self._nod_phase > 2 * np.pi:
                self._nod_type = None
                self.manager.stop()
            else:
                vel = np.zeros(6)
                if self._nod_type == 'yes':
                    vel[3] = 0.8 * np.sin(self._nod_phase * 2)
                else:
                    vel[0] = 0.6 * np.sin(self._nod_phase * 2)
                self.manager.set_joint_velocity(vel)
    
    def run(self):
        print("="*60)
        print("       Z1 Gesture-Based Controller")
        print("="*60)
        print(f"Confidence threshold: {self.confidence_threshold}")
        print("Press Ctrl+C to stop")
        print("="*60)
        
        self._setup_ros()
        
        print("Connecting to robot...")
        if not self.arm.connect():
            print("Failed to connect!")
            return
        print("Connected!")
        
        try:
            while not rospy.is_shutdown():
                loop_start = time.time()
                
                # Only predict if confidence is high enough
                if self.gesture_confidence >= self.confidence_threshold:
                    action, action_id = self.predict_action()
                    
                    if action != self.current_action:
                        print(f"Gesture: {self.current_gesture:15s} -> Action: {action}")
                        self.execute_action(action)
                        self.current_action = action
                        self.prev_action_id = action_id
                        
                        # Publish action
                        self.action_pub.publish(action)
                
                # Update animations
                self._update_animations()
                
                # Update robot
                self.manager.update()
                self.arm.send_recv()
                
                # Maintain loop rate
                elapsed = time.time() - loop_start
                sleep_time = (1.0 / self.hertz) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\n[Stopped]")
        
        finally:
            self.arm.shutdown()


def main():
    parser = argparse.ArgumentParser(description='Gesture-based Robot Controller')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.pt)')
    parser.add_argument('--ip', type=str, default='127.0.0.1',
                        help='Controller IP address')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Gesture confidence threshold')
    args = parser.parse_args()
    
    controller = GestureController(args.model, args.ip)
    controller.confidence_threshold = args.threshold
    controller.run()


if __name__ == "__main__":
    main()

