import numpy as np
from typing import Tuple, Optional

class Z1Kinematics:

    # Link lengths from URDF (meters)
    D1 = 0.1035   # Base height (0.0585 + 0.045)
    A2 = 0.35     # Upper arm length
    A3 = 0.225    # Forearm length (approx sqrt(0.218^2 + 0.057^2))
    D3 = 0.057    # Forearm z-offset
    A4 = 0.07     # Wrist length 1
    A5 = 0.0492   # Wrist length 2
    
    # Joint limits (radians)
    JOINT_LIMITS = np.array([
        [-2.618, 2.618],    # joint1
        [0.0, 2.967],       # joint2
        [-2.880, 0.0],      # joint3
        [-1.518, 1.518],    # joint4
        [-1.344, 1.344],    # joint5
        [-2.793, 2.793]     # joint6
    ])
    
    def __init__(self):
        self._last_solution = None
    
    @staticmethod
    def rotation_x(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    
    @staticmethod
    def rotation_y(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    
    @staticmethod
    def rotation_z(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    
    @staticmethod
    def transform_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    def forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array(q).flatten()
        
        # Joint 1: rotation around Z
        T01 = self.transform_matrix(
            self.rotation_z(q[0]),
            np.array([0, 0, self.D1])
        )
        
        # Joint 2: rotation around Y
        T12 = self.transform_matrix(
            self.rotation_y(q[1]),
            np.array([0, 0, 0])
        )
        
        # Link 2 translation + Joint 3
        T23 = self.transform_matrix(
            self.rotation_y(q[2]),
            np.array([-self.A2, 0, 0])
        )
        
        # Link 3 translation + Joint 4
        T34 = self.transform_matrix(
            self.rotation_y(q[3]),
            np.array([self.A3, 0, self.D3])
        )
        
        # Link 4 translation + Joint 5
        T45 = self.transform_matrix(
            self.rotation_z(q[4]),
            np.array([self.A4, 0, 0])
        )
        
        # Link 5 translation + Joint 6
        T56 = self.transform_matrix(
            self.rotation_x(q[5]),
            np.array([self.A5, 0, 0])
        )
        
        T = T01 @ T12 @ T23 @ T34 @ T45 @ T56
        
        position = T[:3, 3]
        orientation = T[:3, :3]
        
        return position, orientation
    
    def get_end_effector_pose(self, q: np.ndarray) -> np.ndarray:
        pos, rot = self.forward_kinematics(q)
        return self.transform_matrix(rot, pos)
    
    def jacobian(self, q: np.ndarray) -> np.ndarray:
        eps = 1e-6
        J = np.zeros((6, 6))
        
        pos0, _ = self.forward_kinematics(q)
        
        for i in range(6):
            q_plus = q.copy()
            q_plus[i] += eps
            pos_plus, _ = self.forward_kinematics(q_plus)
            
            J[:3, i] = (pos_plus - pos0) / eps
        

        T = np.eye(4)
        axes = []
        joint_axes = [
            np.array([0, 0, 1]),  # joint1: Z
            np.array([0, 1, 0]),  # joint2: Y
            np.array([0, 1, 0]),  # joint3: Y
            np.array([0, 1, 0]),  # joint4: Y
            np.array([0, 0, 1]),  # joint5: Z
            np.array([1, 0, 0]),  # joint6: X
        ]
        
        for i, axis in enumerate(joint_axes):
            J[3:6, i] = axis
        
        return J
    
    def inverse_kinematics(self, 
                           target_pos: np.ndarray, 
                           target_rot: Optional[np.ndarray] = None,
                           q_init: Optional[np.ndarray] = None,
                           max_iter: int = 100,
                           tol: float = 1e-4) -> Tuple[np.ndarray, bool]:
        target_pos = np.array(target_pos).flatten()
        
        # Initial guess
        if q_init is not None:
            q = np.array(q_init).copy()
        elif self._last_solution is not None:
            q = self._last_solution.copy()
        else:
            q = np.array([0.0, 1.0, -0.5, 0.0, 0.0, 0.0])
        
        # Damped least squares parameters
        damping = 0.1
        step_size = 0.5
        
        for iteration in range(max_iter):
            # Current position
            current_pos, current_rot = self.forward_kinematics(q)
            
            # Position error
            pos_error = target_pos - current_pos
            error_norm = np.linalg.norm(pos_error)
            
            if error_norm < tol:
                self._last_solution = q.copy()
                return self.clip_to_limits(q), True
            
            J = self.jacobian(q)[:3, :]  
            
            # Damped least squares
            JJT = J @ J.T + damping**2 * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, pos_error)
            
            # Update
            q = q + step_size * dq
            
            # Clip to limits
            q = self.clip_to_limits(q)
        
        # Failed to converge
        return q, False
    
    def clip_to_limits(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.JOINT_LIMITS[:, 0], self.JOINT_LIMITS[:, 1])
    
    def cartesian_to_joint_velocity(self, 
                                    q: np.ndarray, 
                                    cart_vel: np.ndarray) -> np.ndarray:

        cart_vel = np.array(cart_vel).flatten()
        
        if len(cart_vel) == 3:
            # Position only
            J = self.jacobian(q)[:3, :]
            # Damped pseudo-inverse
            damping = 0.05
            JJT = J @ J.T + damping**2 * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, cart_vel)
        else:
            # Full 6-DOF
            J = self.jacobian(q)
            damping = 0.05
            JJT = J @ J.T + damping**2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJT, cart_vel)
        
        return dq

