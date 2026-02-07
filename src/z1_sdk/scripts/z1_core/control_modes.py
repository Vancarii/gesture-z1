import numpy as np
import time
from enum import Enum, auto
from typing import Optional, List, Tuple
from .kinematics import Z1Kinematics


class ControlMode(Enum):
    """Available control modes"""
    PASSIVE = auto()           # Gravity compensation only
    JOINT_VELOCITY = auto()    # Direct joint velocity control
    JOINT_POSITION = auto()    # Joint position control
    CARTESIAN_VELOCITY = auto() # End-effector velocity control
    CARTESIAN_POSITION = auto() # End-effector position control
    TRAJECTORY = auto()        # Follow trajectory
    HOME = auto()              # Go to home position


class TrajectoryPoint:
    def __init__(self, positions: np.ndarray, time: float, velocities: np.ndarray = None):
        self.positions = np.array(positions)
        self.time = time
        self.velocities = np.array(velocities) if velocities is not None else np.zeros(6)


class ControlModeManager:
    
    def __init__(self, arm_interface):
        self.arm = arm_interface
        self.kinematics = Z1Kinematics()
        
        self._current_mode = ControlMode.PASSIVE
        self._target_joint_vel = np.zeros(6)
        self._target_joint_pos = None
        self._target_cart_vel = np.zeros(3)
        self._target_cart_pos = None
        self._trajectory: List[TrajectoryPoint] = []
        self._trajectory_start_time = 0
        self._trajectory_index = 0
        
        self.position_gain = 2.0  # For position control
        self.max_velocity = 1.0   # Max joint velocity (rad/s)
        self.position_threshold = 0.01  # Position reached threshold
        
        # Home position
        self.home_position = np.array([0.0, 1.5, -1.0, 0.0, 0.0, 0.0])
    
    @property
    def mode(self) -> ControlMode:
        return self._current_mode
    
    def set_mode(self, mode: ControlMode):
        self._current_mode = mode
        
        if mode == ControlMode.PASSIVE:
            self.arm.go_passive()
        elif mode == ControlMode.HOME:
            self._target_joint_pos = self.home_position.copy()
    
    # ==================== Setters ====================
    
    def set_joint_velocity(self, velocities: np.ndarray):
        self._target_joint_vel = np.array(velocities).flatten()[:6]
        self._current_mode = ControlMode.JOINT_VELOCITY
    
    def set_joint_position(self, positions: np.ndarray):
        self._target_joint_pos = np.array(positions).flatten()[:6]
        self._current_mode = ControlMode.JOINT_POSITION
    
    def set_cartesian_velocity(self, velocity: np.ndarray):
        self._target_cart_vel = np.array(velocity).flatten()[:3]
        self._current_mode = ControlMode.CARTESIAN_VELOCITY
    
    def set_cartesian_position(self, position: np.ndarray):
        self._target_cart_pos = np.array(position).flatten()[:3]
        self._current_mode = ControlMode.CARTESIAN_POSITION
    
    def set_trajectory(self, trajectory: List[TrajectoryPoint]):
        self._trajectory = trajectory
        self._trajectory_index = 0
        self._trajectory_start_time = time.time()
        self._current_mode = ControlMode.TRAJECTORY
    
    def go_home(self):
        self._target_joint_pos = self.home_position.copy()
        self._current_mode = ControlMode.HOME
    
    def stop(self):
        self._target_joint_vel = np.zeros(6)
        self._target_cart_vel = np.zeros(3)
        self.arm.stop()
    
    # ==================== Update ====================
    
    def update(self) -> bool:
        if self._current_mode == ControlMode.PASSIVE:
            self.arm.go_passive()
            return False
        
        elif self._current_mode == ControlMode.JOINT_VELOCITY:
            return self._update_joint_velocity()
        
        elif self._current_mode == ControlMode.JOINT_POSITION:
            return self._update_joint_position()
        
        elif self._current_mode == ControlMode.CARTESIAN_VELOCITY:
            return self._update_cartesian_velocity()
        
        elif self._current_mode == ControlMode.CARTESIAN_POSITION:
            return self._update_cartesian_position()
        
        elif self._current_mode == ControlMode.TRAJECTORY:
            return self._update_trajectory()
        
        elif self._current_mode == ControlMode.HOME:
            return self._update_joint_position()  # Uses _target_joint_pos
        
        return False
    
    def _update_joint_velocity(self) -> bool:
        vel = np.clip(self._target_joint_vel, -self.max_velocity, self.max_velocity)
        self.arm.set_joint_velocities(vel)
        return np.any(np.abs(vel) > 0.001)
    
    def _update_joint_position(self) -> bool:
        if self._target_joint_pos is None:
            return False
        
        current = self.arm.joint_positions
        error = self._target_joint_pos - current
        
        if np.linalg.norm(error) < self.position_threshold:
            self.arm.stop()
            return False
        
        vel = self.position_gain * error
        vel = np.clip(vel, -self.max_velocity, self.max_velocity)
        
        self.arm.set_joint_velocities(vel)
        return True
    
    def _update_cartesian_velocity(self) -> bool:
        """Update Cartesian velocity control"""
        current_q = self.arm.joint_positions
        
        joint_vel = self.kinematics.cartesian_to_joint_velocity(
            current_q, self._target_cart_vel
        )
        
        joint_vel = np.clip(joint_vel, -self.max_velocity, self.max_velocity)
        
        self.arm.set_joint_velocities(joint_vel)
        return np.any(np.abs(self._target_cart_vel) > 0.001)
    
    def _update_cartesian_position(self) -> bool:
        if self._target_cart_pos is None:
            return False
        
        current_q = self.arm.joint_positions
        current_pos, _ = self.kinematics.forward_kinematics(current_q)
        
        error = self._target_cart_pos - current_pos
        
        if np.linalg.norm(error) < self.position_threshold:
            self.arm.stop()
            return False
        
        cart_vel = self.position_gain * error
        cart_vel = np.clip(cart_vel, -0.2, 0.2) 
        
        joint_vel = self.kinematics.cartesian_to_joint_velocity(current_q, cart_vel)
        joint_vel = np.clip(joint_vel, -self.max_velocity, self.max_velocity)
        
        self.arm.set_joint_velocities(joint_vel)
        return True
    
    def _update_trajectory(self) -> bool:
        if not self._trajectory:
            return False
        
        elapsed = time.time() - self._trajectory_start_time
        
        # Find current trajectory point
        while (self._trajectory_index < len(self._trajectory) - 1 and 
               elapsed > self._trajectory[self._trajectory_index + 1].time):
            self._trajectory_index += 1
        
        if self._trajectory_index >= len(self._trajectory) - 1:
            # Trajectory complete
            self._target_joint_pos = self._trajectory[-1].positions
            return self._update_joint_position()
        
        # Interpolate between points
        p0 = self._trajectory[self._trajectory_index]
        p1 = self._trajectory[self._trajectory_index + 1]
        
        if p1.time > p0.time:
            alpha = (elapsed - p0.time) / (p1.time - p0.time)
            alpha = np.clip(alpha, 0, 1)
        else:
            alpha = 1.0
        
        target_pos = (1 - alpha) * p0.positions + alpha * p1.positions
        target_vel = (1 - alpha) * p0.velocities + alpha * p1.velocities
        
        # Use position control with feedforward velocity
        self.arm.set_joint_positions(target_pos, target_vel)
        return True
    
    
    def get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.kinematics.forward_kinematics(self.arm.joint_positions)
    
    def get_status(self) -> dict:
        pos, rot = self.get_end_effector_pose()
        return {
            'mode': self._current_mode.name,
            'joint_positions': self.arm.joint_positions.tolist(),
            'joint_velocities': self.arm.joint_velocities.tolist(),
            'ee_position': pos.tolist(),
            'gripper': self.arm.gripper_position,
            'has_error': self.arm.has_error()
        }

