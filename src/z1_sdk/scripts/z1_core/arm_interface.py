"""
Z1 Arm Interface - Low-level communication wrapper

Provides a clean interface to the unitree_arm_interface SDK.
"""

import os
import sys
import time
import numpy as np

def _setup_sdk_path():
    """Setup path to find unitree_arm_interface module"""
    try:
        import rospkg
        z1_sdk_path = rospkg.RosPack().get_path('z1_sdk')
        devel_lib_path = os.path.join(os.path.dirname(os.path.dirname(z1_sdk_path)), 'devel', 'lib')
        if os.path.exists(devel_lib_path):
            sys.path.insert(0, devel_lib_path)
        sys.path.append(z1_sdk_path + "/lib")
    except ImportError:
        pass
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
    
    potential_paths = [
        os.path.join(workspace_root, 'devel', 'lib'),
        os.path.join(workspace_root, 'src', 'z1_sdk', 'lib'),
        '/home/tangentlab/z1_ws/devel/lib',
        '/home/tangentlab/z1_ws/src/z1_sdk/lib',
    ]
    
    for path in potential_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)

_setup_sdk_path()
import unitree_arm_interface


class ArmMode:
    INVALID = unitree_arm_interface.ArmMode.Invalid
    PASSIVE = unitree_arm_interface.ArmMode.Passive
    LOWCMD = unitree_arm_interface.ArmMode.LowCmd
    JOINT_SPEED = unitree_arm_interface.ArmMode.JointSpeedCtrl
    JOINT_POSITION = unitree_arm_interface.ArmMode.JointPositionCtrl
    TEACH = unitree_arm_interface.ArmMode.Teach
    TEACH_REPEAT = unitree_arm_interface.ArmMode.TeachRepeat
    CALIBRATION = unitree_arm_interface.ArmMode.Calibration
    CLEAR_ERROR = unitree_arm_interface.ArmMode.ClearError


class Z1ArmInterface:

    JOINT_LIMITS = np.array([
        [-2.618, 2.618],    # joint1: -150° to 150°
        [0.0, 2.967],       # joint2: 0° to 170°
        [-2.880, 0.0],      # joint3: -165° to 0°
        [-1.518, 1.518],    # joint4: -87° to 87°
        [-1.344, 1.344],    # joint5: -77° to 77°
        [-2.793, 2.793]     # joint6: -160° to 160°
    ])
    
    HOME_POSITION = np.array([0.0, 1.5, -1.0, 0.0, 0.0, 0.0])
    
    def __init__(self, controller_ip: str = "127.0.0.1"):

        self.controller_ip = controller_ip
        self._arm = None
        self._connected = False
        self._current_mode = None
        
        self._gripper_cmd = unitree_arm_interface.GripperCmd()
        self._gripper_cmd.angle = 0.0
        self._gripper_cmd.speed = 0.5
        self._gripper_cmd.maxTau = 5.0
        self._gripper_cmd.epsilon_inner = 0.01
        self._gripper_cmd.epsilon_outer = 0.01
    
    def connect(self) -> bool:
        try:
            print(f"Connecting to z1_controller at {self.controller_ip}...")
            self._arm = unitree_arm_interface.UnitreeArm(self.controller_ip, 0)
            self._arm.init()
            self._connected = True
            print("Connected successfully!")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            self._connected = False
            return False
    
    @property
    def connected(self) -> bool:
        return self._connected
    
    @property
    def dt(self) -> float:
        return self._arm.dt if self._arm else 0.004
    
    
    @property
    def joint_positions(self) -> np.ndarray:
        if not self._arm:
            return np.zeros(6)
        return np.array(self._arm.armState.q[:6])
    
    @property
    def joint_velocities(self) -> np.ndarray:
        if not self._arm:
            return np.zeros(6)
        return np.array(self._arm.armState.dq[:6])
    
    @property
    def joint_torques(self) -> np.ndarray:
        if not self._arm:
            return np.zeros(6)
        return np.array(self._arm.armState.tau[:6])
    
    @property
    def gripper_position(self) -> float:
        if not self._arm:
            return 0.0
        return self._arm.armState.gripperState.angle
    
    @property
    def mode(self):
        if not self._arm:
            return None
        return self._arm.armState.mode
    
    def has_error(self) -> bool:
        if not self._arm:
            return False
        return self._arm.armState.hasError() if hasattr(self._arm.armState, 'hasError') else False
    
    
    def set_mode(self, mode):
        if self._arm:
            self._arm.armCmd.mode = mode
            self._current_mode = mode
    
    def set_joint_velocities(self, velocities: np.ndarray):
        if self._arm:
            self._arm.armCmd.mode = ArmMode.JOINT_SPEED
            self._arm.armCmd.dq_d = list(velocities[:6])
    
    def set_joint_positions(self, positions: np.ndarray, velocities: np.ndarray = None):
        if self._arm:
            self._arm.armCmd.mode = ArmMode.JOINT_POSITION
            self._arm.armCmd.q_d = list(positions[:6])
            if velocities is not None:
                self._arm.armCmd.dq_d = list(velocities[:6])
    
    def set_gripper(self, angle: float, speed: float = 0.5):
        self._gripper_cmd.angle = angle
        self._gripper_cmd.speed = speed
        if self._arm:
            self._arm.armCmd.gripperCmd = self._gripper_cmd
    
    def stop(self):
        if self._arm:
            self._arm.armCmd.mode = ArmMode.JOINT_SPEED
            self._arm.armCmd.dq_d = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def go_passive(self):
        self.set_mode(ArmMode.PASSIVE)
    
    def clear_errors(self):
        if self._arm:
            self._arm.armCmd.mode = ArmMode.CLEAR_ERROR
            self.send_recv()
            time.sleep(0.01)
            self._arm.armCmd.mode = ArmMode.JOINT_SPEED
            self.send_recv()
    
    def send_recv(self):
        if self._arm:
            self._arm.sendRecv()
    
    def update(self):
        self.send_recv()
        
    def clip_to_limits(self, positions: np.ndarray) -> np.ndarray:
        positions = np.array(positions)
        return np.clip(positions, self.JOINT_LIMITS[:, 0], self.JOINT_LIMITS[:, 1])
    
    def is_within_limits(self, positions: np.ndarray, margin: float = 0.05) -> bool:
        positions = np.array(positions)
        lower = self.JOINT_LIMITS[:, 0] + margin
        upper = self.JOINT_LIMITS[:, 1] - margin
        return np.all((positions >= lower) & (positions <= upper))
    
    def shutdown(self):
        if self._arm:
            print("Shutting down...")
            self.go_passive()
            for _ in range(10):
                self.send_recv()
                time.sleep(self.dt)
            print("Shutdown complete.")

