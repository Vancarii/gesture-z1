
import numpy as np
from typing import Tuple, Optional
import time


class SafetyMonitor:
    
    # Joint limits (radians)
    JOINT_LIMITS = np.array([
        [-2.618, 2.618],    # joint1
        [0.0, 2.967],       # joint2
        [-2.880, 0.0],      # joint3
        [-1.518, 1.518],    # joint4
        [-1.344, 1.344],    # joint5
        [-2.793, 2.793]     # joint6
    ])
    
    # Safety margins
    SAFETY_MARGIN = 0.1  # radians from limit to start slowing
    CRITICAL_MARGIN = 0.02  # radians from limit to stop
    
    # Velocity limits (rad/s)
    MAX_JOINT_VELOCITY = np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    
    # Torque limits (Nm)
    TORQUE_WARNING = np.array([25.0, 50.0, 25.0, 25.0, 25.0, 25.0])
    TORQUE_CRITICAL = np.array([28.0, 55.0, 28.0, 28.0, 28.0, 28.0])
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._last_warning_time = 0
        self._warning_interval = 0.5  # seconds between warnings
        self._is_stopped = False
    
    def check_joint_limits(self, 
                          positions: np.ndarray, 
                          velocities: np.ndarray = None) -> Tuple[np.ndarray, bool]:
        positions = np.array(positions).flatten()
        if velocities is None:
            velocities = np.zeros(6)
        else:
            velocities = np.array(velocities).flatten()
        
        safe_vel = velocities.copy()
        should_stop = False
        current_time = time.time()
        
        for i in range(6):
            lower, upper = self.JOINT_LIMITS[i]
            pos = positions[i]
            vel = velocities[i]
            
            # Distance to limits
            dist_to_lower = pos - lower
            dist_to_upper = upper - pos
            
            # Check critical zone
            if dist_to_lower < self.CRITICAL_MARGIN:
                if vel < 0:  # Moving toward limit
                    safe_vel[i] = 0
                    should_stop = True
                    self._warn(f"Joint {i+1} at lower limit", current_time)
            elif dist_to_upper < self.CRITICAL_MARGIN:
                if vel > 0:  # Moving toward limit
                    safe_vel[i] = 0
                    should_stop = True
                    self._warn(f"Joint {i+1} at upper limit", current_time)
            
            # Check safety zone - reduce velocity
            elif dist_to_lower < self.SAFETY_MARGIN and vel < 0:
                scale = dist_to_lower / self.SAFETY_MARGIN
                safe_vel[i] = vel * scale
                self._warn(f"Joint {i+1} approaching lower limit", current_time)
            elif dist_to_upper < self.SAFETY_MARGIN and vel > 0:
                scale = dist_to_upper / self.SAFETY_MARGIN
                safe_vel[i] = vel * scale
                self._warn(f"Joint {i+1} approaching upper limit", current_time)
        
        return safe_vel, should_stop
    
    def check_velocity_limits(self, velocities: np.ndarray) -> np.ndarray:
        velocities = np.array(velocities).flatten()
        return np.clip(velocities, -self.MAX_JOINT_VELOCITY, self.MAX_JOINT_VELOCITY)
    
    def check_torques(self, torques: np.ndarray) -> Tuple[float, bool]:
        torques = np.abs(np.array(torques).flatten())
        current_time = time.time()
        
        # Check for critical torque
        for i in range(6):
            if torques[i] > self.TORQUE_CRITICAL[i]:
                self._warn(f"Critical torque on joint {i+1}: {torques[i]:.1f} Nm", current_time)
                return 0.0, True
        
        # Check for warning torque
        scale = 1.0
        for i in range(6):
            if torques[i] > self.TORQUE_WARNING[i]:
                joint_scale = 1.0 - (torques[i] - self.TORQUE_WARNING[i]) / \
                             (self.TORQUE_CRITICAL[i] - self.TORQUE_WARNING[i])
                scale = min(scale, max(0.1, joint_scale))
                self._warn(f"High torque on joint {i+1}: {torques[i]:.1f} Nm", current_time)
        
        return scale, False
    
    def get_safe_velocities(self,
                           positions: np.ndarray,
                           velocities: np.ndarray,
                           torques: np.ndarray = None) -> Tuple[np.ndarray, bool]:
        # Velocity limits
        velocities = self.check_velocity_limits(velocities)
        
        # Joint limits
        velocities, pos_stop = self.check_joint_limits(positions, velocities)
        
        # Torque check
        if torques is not None:
            torque_scale, torque_stop = self.check_torques(torques)
            velocities = velocities * torque_scale
            should_stop = pos_stop or torque_stop
        else:
            should_stop = pos_stop
        
        self._is_stopped = should_stop
        return velocities, should_stop
    
    @property
    def is_stopped(self) -> bool:
        return self._is_stopped
    
    def reset(self):
        self._is_stopped = False
    
    def _warn(self, message: str, current_time: float):
        if self.verbose and current_time - self._last_warning_time > self._warning_interval:
            print(f"[Safety] {message}")
            self._last_warning_time = current_time


class WorkspaceBounds:
    
    def __init__(self,
                 x_range: Tuple[float, float] = (-0.6, 0.6),
                 y_range: Tuple[float, float] = (-0.6, 0.6),
                 z_range: Tuple[float, float] = (0.0, 0.8)):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range
    
    def is_in_bounds(self, position: np.ndarray, margin: float = 0.05) -> bool:
        x, y, z = position[:3]
        return (self.x_min + margin <= x <= self.x_max - margin and
                self.y_min + margin <= y <= self.y_max - margin and
                self.z_min + margin <= z <= self.z_max - margin)
    
    def clip_to_bounds(self, position: np.ndarray, margin: float = 0.05) -> np.ndarray:
        position = np.array(position).flatten()
        position[0] = np.clip(position[0], self.x_min + margin, self.x_max - margin)
        position[1] = np.clip(position[1], self.y_min + margin, self.y_max - margin)
        position[2] = np.clip(position[2], self.z_min + margin, self.z_max - margin)
        return position

