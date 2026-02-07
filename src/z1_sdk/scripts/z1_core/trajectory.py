
import numpy as np
from typing import List, Tuple
from .control_modes import TrajectoryPoint


def generate_linear_trajectory(start: np.ndarray, 
                               end: np.ndarray, 
                               duration: float,
                               num_points: int = 50) -> List[TrajectoryPoint]:
    trajectory = []
    times = np.linspace(0, duration, num_points)
    
    for t in times:
        alpha = t / duration
        # Smooth interpolation using cosine
        alpha_smooth = 0.5 * (1 - np.cos(np.pi * alpha))
        
        pos = (1 - alpha_smooth) * start + alpha_smooth * end
        
        # Velocity (derivative of cosine interpolation)
        vel_scale = (np.pi / (2 * duration)) * np.sin(np.pi * alpha)
        vel = vel_scale * (end - start)
        
        trajectory.append(TrajectoryPoint(pos, t, vel))
    
    return trajectory


def generate_min_jerk_trajectory(start: np.ndarray,
                                  end: np.ndarray,
                                  duration: float,
                                  num_points: int = 50) -> List[TrajectoryPoint]:
    trajectory = []
    times = np.linspace(0, duration, num_points)
    
    for t in times:
        tau = t / duration
        
        # Minimum jerk position: 10*tau^3 - 15*tau^4 + 6*tau^5
        alpha = 10*tau**3 - 15*tau**4 + 6*tau**5
        
        # Minimum jerk velocity: (30*tau^2 - 60*tau^3 + 30*tau^4) / duration
        alpha_dot = (30*tau**2 - 60*tau**3 + 30*tau**4) / duration
        
        pos = (1 - alpha) * start + alpha * end
        vel = alpha_dot * (end - start)
        
        trajectory.append(TrajectoryPoint(pos, t, vel))
    
    return trajectory


def generate_via_point_trajectory(waypoints: List[np.ndarray],
                                   times: List[float]) -> List[TrajectoryPoint]:
    if len(waypoints) < 2:
        return [TrajectoryPoint(waypoints[0], 0)]
    
    trajectory = []
    
    for i in range(len(waypoints) - 1):
        start = np.array(waypoints[i])
        end = np.array(waypoints[i + 1])
        t_start = times[i]
        t_end = times[i + 1]
        duration = t_end - t_start
        
        num_points = max(10, int(duration / 0.01))  # 100Hz resolution
        
        segment = generate_min_jerk_trajectory(start, end, duration, num_points)
        
        # Offset times
        for point in segment:
            point.time += t_start
            trajectory.append(point)
    
    return trajectory


def generate_circle_trajectory(center: np.ndarray,
                                radius: float,
                                duration: float,
                                kinematics,
                                num_points: int = 100) -> Tuple[List[TrajectoryPoint], bool]:
    trajectory = []
    times = np.linspace(0, duration, num_points)
    
    q_prev = None
    
    for i, t in enumerate(times):
        theta = 2 * np.pi * t / duration
        
        # Circle in XY plane
        target_pos = np.array([
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            center[2]
        ])
        
        q, success = kinematics.inverse_kinematics(target_pos, q_init=q_prev)
        
        if not success:
            return trajectory, False
        
        q_prev = q
        
        # Compute velocity from position difference
        if i > 0:
            dt = times[i] - times[i-1]
            vel = (q - trajectory[-1].positions) / dt
        else:
            vel = np.zeros(6)
        
        trajectory.append(TrajectoryPoint(q, t, vel))
    
    return trajectory, True


def generate_square_trajectory(corner: np.ndarray,
                                size: float,
                                duration: float,
                                kinematics) -> Tuple[List[TrajectoryPoint], bool]:
    # Define corners
    corners = [
        corner,
        corner + np.array([size, 0, 0]),
        corner + np.array([size, size, 0]),
        corner + np.array([0, size, 0]),
        corner  # Return to start
    ]
    
    # Convert to joint space
    joint_waypoints = []
    q_prev = None
    
    for pos in corners:
        q, success = kinematics.inverse_kinematics(pos, q_init=q_prev)
        if not success:
            return [], False
        joint_waypoints.append(q)
        q_prev = q
    
    # Generate times (equal time for each side)
    side_time = duration / 4
    times = [i * side_time for i in range(5)]
    
    return generate_via_point_trajectory(joint_waypoints, times), True

