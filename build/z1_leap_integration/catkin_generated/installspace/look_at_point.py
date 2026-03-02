#!/usr/bin/env python3
"""
look_at_point.py  (v5 — ROS subscriber / continuous mode)
----------------------------------------------------------
Subscribes to /leap/pointing_target published by detect_point.py and
continuously drives the Z1 arm to point at whatever 3D coordinate the
Leap Motion detects.

Topic consumed:
    /leap/pointing_target  (std_msgs/String, JSON)
    {"target_pos_m": [x, y, z], "frame": "arm_base"}

Pipeline:
    detect_point  →  /leap/pointing_target  →  look_at_point  →  MoveIt

Run:
    roslaunch z1_leap_integration simulation_full.launch

(gesture_moveit_controller is no longer needed — this node replaces it.)
"""

import sys
import math
import threading
import json
import numpy as np

import rospy
import moveit_commander
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from tf.transformations import (
    quaternion_from_matrix,
    quaternion_matrix,
    unit_vector,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PLANNING_GROUP  = "manipulator"
EEF_LINK        = "link06"
BASE_FRAME      = "world"

# CONFIRMED from `rosrun tf tf_echo link06 gripperStator`:
# Translation [0.051, 0.000, 0.000]  →  pointing axis is +X
EEF_POINT_AXIS  = np.array([1.0, 0.0, 0.0])

# Skip motion if angular error is already within this threshold
ALREADY_POINTING_THRESHOLD_DEG = 3.0

# Reject targets pointing below this elevation angle (degrees below horizontal).
# The Z1 can tilt slightly downward, but steeply underground targets are
# unreachable and cause planner failures.
MIN_ELEVATION_DEG = -30.0

PLANNING_TIME   = 10.0
MAX_VELOCITY    = 0.3    # base velocity — scaled down for large moves
MAX_ACCEL       = 0.3

MAX_RETRIES     = 2      # retry once on CONTROL_FAILED before giving up

# Settle time after a motion before accepting the next target.
MIN_REPLAN_INTERVAL = 1.0

# ---------------------------------------------------------------------------
# Coordinate frame transform: Leap world → ROS robot world
# ---------------------------------------------------------------------------
# detect_point publishes targets in Leap world frame (after tilt correction):
#   +X = right (from user's perspective)
#   +Y = up
#   +Z = toward the user
#
# look_at_point (MoveIt) uses ROS robot world frame:
#   +X = forward (away from user, toward objects)
#   +Y = left    (same as user's left — user and arm face the same way)
#   +Z = up
#
# Physical setup:  [User] → [Leap] → [Arm] → [Objects]  (all face same direction)
#
# Derived mapping (det = +1, proper rotation):
#   robot_x = -leap_z   (forward = away from user)
#   robot_y = -leap_x   (robot left = user left)
#   robot_z =  leap_y   (up = up)
#
# If lateral pointing is mirrored, flip the sign of row 1.
LEAP_TO_ROBOT = np.array([
    [ 0,  0, -1],
    [-1,  0,  0],
    [ 0,  1,  0],
], dtype=float)

# ---------------------------------------------------------------------------


# ── Maths helpers ──────────────────────────────────────────────────────────


def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = unit_vector(a)
    b = unit_vector(b)
    cross = np.cross(a, b)
    cross_norm = np.linalg.norm(cross)
    dot = float(np.dot(a, b))

    if cross_norm < 1e-9:
        if dot > 0:
            return np.eye(3)
        perp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, perp)) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = unit_vector(np.cross(a, perp))
        K = np.array([[       0, -axis[2],  axis[1]],
                      [ axis[2],        0, -axis[0]],
                      [-axis[1],  axis[0],        0]])
        return np.eye(3) + 2.0 * K @ K

    K = np.array([[        0, -cross[2],  cross[1]],
                  [ cross[2],         0, -cross[0]],
                  [-cross[1],  cross[0],         0]])
    return np.eye(3) + K + K @ K * ((1.0 - dot) / (cross_norm ** 2))


def R_to_quat(R: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    return quaternion_from_matrix(T)   # [x, y, z, w]


def quat_to_R(q: Quaternion) -> np.ndarray:
    return quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]


def current_pointing_direction(group) -> np.ndarray:
    q = group.get_current_pose().pose.orientation
    R = quat_to_R(q)
    return unit_vector(R @ EEF_POINT_AXIS)


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos_a = np.clip(np.dot(unit_vector(a), unit_vector(b)), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


# ── Planning helpers ────────────────────────────────────────────────────────

def _prepare_group(group):
    """
    Always call this immediately before planning.
    Syncs the planner's start state with the live controller state,
    which is the primary cause of CONTROL_FAILED.
    """
    group.set_start_state_to_current_state()


def _execute_and_check(group, plan) -> bool:
    """Execute a plan and return True only if the controller confirms success."""
    result = group.execute(plan, wait=True)
    group.stop()
    group.clear_pose_targets()
    group.clear_path_constraints()
    if not result:
        rospy.logerr("group.execute() returned False — controller rejected trajectory.")
    return result


def plan_orientation_only(group, quat_xyzw: np.ndarray) -> bool:
    """
    Plan an orientation-only goal (position unconstrained).

    NOTE: The previous version added an OrientationConstraint as a *path*
    constraint on top of the orientation goal.  Path constraints force OMPL
    to keep every intermediate state within the tolerance band, which:
      1. Causes the planner to insert dozens of extra waypoints very close
         together along the constraint boundary.
      2. MoveIt's AddTimeParameterization adapter then produces zero- or
         negative-duration segments between those micro-steps, resulting in
         the 'waypoints not strictly increasing in time' controller error.

    Fix: use only set_orientation_target (goal constraint) — no path
    constraint.  The planner is free to take any collision-free path, and the
    resulting trajectory is clean and properly time-parameterised.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        rospy.loginfo(f"Orientation planning — attempt {attempt}/{MAX_RETRIES}")

        # Sync start state to live robot state before every plan
        _prepare_group(group)

        group.set_planning_time(PLANNING_TIME)
        group.set_orientation_target(quat_xyzw.tolist(), EEF_LINK)

        plan_ok, plan, t, err = group.plan()
        group.clear_pose_targets()

        if not plan_ok:
            rospy.logwarn(f"  Planning failed (error {err.val}, {t:.2f}s).")
            continue

        rospy.loginfo(f"  Plan found in {t:.2f}s — executing.")
        if _execute_and_check(group, plan):
            return True

        rospy.logwarn(
            f"  Execution failed on attempt {attempt}. "
            "Re-reading joint state and retrying …"
        )
        rospy.sleep(1.0)   # let controller fully settle before retry

    return False


def plan_joint_fallback(group, quat_xyzw: np.ndarray) -> bool:
    """
    Fallback: set a full Pose goal using the CURRENT EEF position with the
    desired orientation.  This avoids the unreachable-position problem of
    the old ray-sampled approach (negative-Z targets underground).

    Uses the arm's current position so MoveIt only needs to rotate in-place,
    but with a full Pose goal which gives the planner more flexibility than
    a pure orientation target.
    """
    q = Quaternion(x=float(quat_xyzw[0]), y=float(quat_xyzw[1]),
                   z=float(quat_xyzw[2]), w=float(quat_xyzw[3]))

    for attempt in range(1, MAX_RETRIES + 1):
        _prepare_group(group)

        # Get the CURRENT position — only change the orientation
        current = group.get_current_pose().pose
        pose = Pose()
        pose.position    = current.position   # stay where we are
        pose.orientation = q                  # new pointing direction

        rospy.loginfo(
            f"  Joint fallback attempt {attempt}/{MAX_RETRIES}: "
            f"pose at [{current.position.x:.3f}, {current.position.y:.3f}, "
            f"{current.position.z:.3f}] with new orientation"
        )

        group.set_planning_time(PLANNING_TIME)
        group.set_pose_target(pose)

        plan_ok, plan, t, err = group.plan()
        group.clear_pose_targets()

        if not plan_ok:
            rospy.logwarn(f"    Planning failed (error {err.val}, {t:.2f}s).")
            continue

        rospy.loginfo(f"    Plan found in {t:.2f}s — executing.")
        if _execute_and_check(group, plan):
            return True

        rospy.logwarn(f"    Execution failed on attempt {attempt}.")
        rospy.sleep(1.0)   # let controller fully settle

    return False


# ── ROS Node ───────────────────────────────────────────────────────────────

class LookAtPointNode:
    """
    Persistent ROS node that listens to /leap/pointing_target (published by
    detect_point.py) and continuously orients the arm to point at that target.

    All MoveIt calls are made from a dedicated worker thread so the ROS
    subscriber callback never blocks.  The latest target always wins — if the
    hand moves while the arm is executing, the new position is picked up as
    soon as the current motion finishes.
    """

    def __init__(self):
        # ── MoveIt initialisation (once at startup) ───────────────────────
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander(PLANNING_GROUP)
        self.group.set_max_velocity_scaling_factor(MAX_VELOCITY)
        self.group.set_max_acceleration_scaling_factor(MAX_ACCEL)
        self.group.set_pose_reference_frame(BASE_FRAME)
        self.group.set_end_effector_link(EEF_LINK)

        # Wait for joint states to arrive (real arm needs z1_hw_node +
        # controller_spawner to finish, which can take 10+ seconds).
        rospy.loginfo("[look_at_point] Waiting for /joint_states ...")
        try:
            rospy.wait_for_message("/joint_states", JointState, timeout=30.0)
            rospy.loginfo("[look_at_point] Joint states received.")
        except rospy.ROSException:
            rospy.logwarn(
                "[look_at_point] Timed out waiting for /joint_states after 30 s. "
                "Continuing anyway — MoveIt may report stale poses."
            )
        # Give TF one more second to propagate the joint positions
        rospy.sleep(1.0)
        eef = self.group.get_current_pose().pose
        rospy.loginfo(
            "[look_at_point] MoveIt ready.  EEF at [%.3f, %.3f, %.3f] (world frame)",
            eef.position.x, eef.position.y, eef.position.z,
        )

        # ── Shared state (lock-protected) ─────────────────────────────────
        self._lock           = threading.Lock()
        self._pending_target = None   # latest np.ndarray from subscriber
        self._busy           = False  # True while a motion is executing

        # ── ROS subscriber ────────────────────────────────────────────────
        rospy.Subscriber(
            "/leap/pointing_target", String,
            self._target_cb, queue_size=1
        )

        # ── Worker thread (owns all MoveIt calls) ─────────────────────────
        t = threading.Thread(target=self._worker_loop, daemon=True)
        t.start()

        rospy.loginfo("[look_at_point] Ready — subscribed to /leap/pointing_target")

    # ── Subscriber callback ───────────────────────────────────────────────

    def _target_cb(self, msg: String):
        """
        Called when detect_point publishes a stable target.  Just stores it;
        the worker thread picks it up when the arm is free.
        """
        try:
            data      = json.loads(msg.data)
            leap_tgt  = np.array(data["target_pos_m"], dtype=float)
            # Convert from Leap world frame to robot world frame
            target    = LEAP_TO_ROBOT @ leap_tgt
        except Exception as e:
            rospy.logwarn_throttle(5.0, "Malformed /leap/pointing_target: %s", e)
            return

        rospy.loginfo(
            "[look_at_point] Target received:  Leap [%.3f, %.3f, %.3f]  "
            "→  Robot [%.3f, %.3f, %.3f] m",
            leap_tgt[0], leap_tgt[1], leap_tgt[2],
            target[0],   target[1],   target[2],
        )
        with self._lock:
            self._pending_target = target   # always overwrite — latest wins

    # ── Worker thread ─────────────────────────────────────────────────────

    def _worker_loop(self):
        """
        Poll for a pending target at 10 Hz.  When the arm is free and a new
        target is waiting, grab it and execute.  MIN_REPLAN_INTERVAL enforces
        a settle pause between consecutive motions.
        """
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            target = None
            with self._lock:
                if not self._busy and self._pending_target is not None:
                    target               = self._pending_target
                    self._pending_target = None
                    self._busy           = True

            if target is not None:
                try:
                    self._point_at(target)
                except Exception as e:
                    rospy.logerr("_point_at raised: %s", e)
                finally:
                    rospy.sleep(MIN_REPLAN_INTERVAL)   # let controller settle
                    with self._lock:
                        self._pending_target = None    # discard anything queued during motion
                        self._busy = False

            rate.sleep()

    # ── Core pointing logic ───────────────────────────────────────────────

    def _point_at(self, target_point: np.ndarray):
        """
        Orient the arm so EEF_POINT_AXIS (+X of link06) points at target_point.
        """
        rospy.loginfo(
            "[look_at_point] Processing target [%.3f, %.3f, %.3f] m",
            target_point[0], target_point[1], target_point[2],
        )
        eef_pose = self.group.get_current_pose().pose
        eef_pos  = np.array([eef_pose.position.x,
                              eef_pose.position.y,
                              eef_pose.position.z])

        cur_pointing = current_pointing_direction(self.group)
        offset       = target_point - eef_pos
        dist         = np.linalg.norm(offset)

        if dist < 1e-4:
            rospy.logwarn("Target coincides with EEF origin — ignoring.")
            return

        target_dir = offset / dist

        # Reject targets below the minimum elevation angle
        elevation_deg = math.degrees(math.asin(np.clip(target_dir[2], -1.0, 1.0)))
        if elevation_deg < MIN_ELEVATION_DEG:
            rospy.logwarn(
                f"Target elevation {elevation_deg:.1f}° is below minimum "
                f"{MIN_ELEVATION_DEG}° — ignoring (pointing underground)."
            )
            return

        angle_err  = angle_between_deg(cur_pointing, target_dir)

        rospy.loginfo(
            f"Target {np.round(target_point, 3)} m  |  "
            f"EEF {np.round(eef_pos, 3)} m  |  "
            f"angular error {angle_err:.1f}°"
        )

        if angle_err < ALREADY_POINTING_THRESHOLD_DEG:
            rospy.loginfo_throttle(2.0, "Already pointing — skipping motion.")
            return

        R    = rotation_matrix_from_vectors(EEF_POINT_AXIS, target_dir)
        quat = R_to_quat(R)
        rospy.loginfo(f"  Goal quat (xyzw): {np.round(quat, 4)}")

        # Scale velocity based on angular distance — big moves go slower
        # to give the Gazebo controller time to track without timeout.
        if angle_err > 60:
            scale = 0.15
        elif angle_err > 30:
            scale = 0.25
        else:
            scale = MAX_VELOCITY
        self.group.set_max_velocity_scaling_factor(scale)
        self.group.set_max_acceleration_scaling_factor(scale)
        rospy.loginfo(f"  Velocity scale: {scale} (angle_err={angle_err:.0f}\u00b0)")

        # Strategy 1: orientation-only (preferred — no position drift)
        if plan_orientation_only(self.group, quat):
            rospy.loginfo("SUCCESS (orientation-only).")
            return

        # Strategy 2: full pose goal at current position with new orientation
        rospy.logwarn("Orientation-only failed \u2014 trying pose-at-current-position fallback \u2026")
        if plan_joint_fallback(self.group, quat):
            rospy.loginfo("SUCCESS (pose fallback).")
            return

        rospy.logerr(
            "All strategies exhausted for target %s.\n"
            "Checklist:\n"
            "  - EEF_POINT_AXIS=[1,0,0] confirmed from tf. Re-run:\n"
            "      rosrun tf tf_echo link06 gripperStator\n"
            "  - CONTROL_FAILED? Increase `stopped_velocity_tolerance`\n"
            "    and `goal_time` in ros_controllers.yaml.\n"
            "  - Try reducing MAX_VELOCITY/MAX_ACCEL further (e.g. 0.05).",
            np.round(target_point, 3),
        )


def main():
    rospy.init_node("look_at_point", anonymous=False)
    LookAtPointNode()
    rospy.spin()


if __name__ == "__main__":

    main()