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
from geometry_msgs.msg import Pose, Quaternion
from moveit_msgs.msg import Constraints, JointConstraint
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

# After a failed execution, treat the attempt as success if the arm ended up
# within this many degrees of the goal (catches spurious GOAL_TOLERANCE_VIOLATED
# where the arm physically reached the target but residual velocity caused a
# controller rejection — avoids a pointless second move).
CLOSE_ENOUGH_DEG = 12.0

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
#   +Y = left    (from robot's perspective)
#   +Z = up
#
# Physical setup: Leap and arm are mounted so that the user faces the arm,
# meaning Leap's +Z (toward user) maps to robot's +X (forward), and
# Leap's +X (user-right) maps to robot's +Y (robot-left = user-right
# when facing each other).
#
#   robot_x = +leap_z
#   robot_y = +leap_x
#   robot_z =  leap_y   (up = up, unchanged)
LEAP_TO_ROBOT = np.array([
    [ 0,  0,  1],
    [ 1,  0,  0],
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


def pointing_dir_from_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    """Direction the EEF would point if at the given goal orientation."""
    R = quaternion_matrix([quat_xyzw[0], quat_xyzw[1],
                           quat_xyzw[2], quat_xyzw[3]])[:3, :3]
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
    Also clears any path constraints left over from a previous failed attempt.
    """
    group.clear_path_constraints()
    group.set_start_state_to_current_state()


def _make_joint6_constraint(group) -> Constraints:
    """
    Return a path constraint that keeps the wrist roll joint (last active joint)
    near its current value during OMPL planning.

    When solving orientation-only goals, OMPL is free to vary joint6 (wrist roll)
    arbitrarily because it has no effect on EEF pointing direction.  On every
    new target it can choose a wildly different joint6 value, causing large
    unnecessary wrist rotations that overheat the motor.

    A JointConstraint bounds how far OMPL will move joint6 away from its current
    position.  ±0.5 rad (~28°) is generous enough to never block valid pointing
    solutions while eliminating the gratuitous full-rotation moves.
    """
    joint_names  = group.get_active_joints()
    joint_values = group.get_current_joint_values()
    j6_name = joint_names[-1]           # last active joint = wrist roll
    j6_val  = float(joint_values[-1])

    jc = JointConstraint()
    jc.joint_name      = j6_name
    jc.position        = j6_val
    jc.tolerance_above = 0.5   # ±0.5 rad (~28°) wrist drift allowed
    jc.tolerance_below = 0.5
    jc.weight          = 1.0

    c = Constraints()
    c.joint_constraints.append(jc)
    return c


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
    A JointConstraint on the wrist roll joint keeps joint6 near its current
    value so OMPL does not choose arbitrary roll solutions that overheat the motor.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        rospy.loginfo(f"Orientation planning — attempt {attempt}/{MAX_RETRIES}")

        # Sync start state to live robot state before every plan
        # (_prepare_group also clears any leftover path constraints)
        _prepare_group(group)

        group.set_planning_time(PLANNING_TIME)
        group.set_orientation_target(quat_xyzw.tolist(), EEF_LINK)
        # Keep joint6 near current value — prevents OMPL using large wrist rolls
        group.set_path_constraints(_make_joint6_constraint(group))

        plan_ok, plan, t, err = group.plan()
        group.clear_pose_targets()

        if not plan_ok:
            rospy.logwarn(f"  Planning failed (error {err.val}, {t:.2f}s).")
            continue

        rospy.loginfo(f"  Plan found in {t:.2f}s — executing.")
        if _execute_and_check(group, plan):
            return True

        # Controller rejected, but check whether the arm actually got there.
        goal_dir = pointing_dir_from_quat(quat_xyzw)
        cur_dir  = current_pointing_direction(group)
        angular_err = angle_between_deg(cur_dir, goal_dir)
        if angular_err < CLOSE_ENOUGH_DEG:
            rospy.loginfo(
                f"  Controller rejected but arm is within {angular_err:.1f}°"
                f" of goal (<{CLOSE_ENOUGH_DEG}°) — accepting result."
            )
            return True

        rospy.logwarn(
            f"  Execution failed on attempt {attempt} — arm is"
            f" {angular_err:.1f}° from goal. Re-reading state and retrying …"
        )
        rospy.sleep(0.3)

    return False


def plan_joint_fallback(group, quat_xyzw: np.ndarray) -> bool:
    """
    Fallback: full Pose goal at the current EEF position with the desired
    orientation, giving the planner more flexibility than orientation-only.
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
        # Keep joint6 near current value even in the fallback path
        group.set_path_constraints(_make_joint6_constraint(group))

        plan_ok, plan, t, err = group.plan()
        group.clear_pose_targets()

        if not plan_ok:
            rospy.logwarn(f"    Planning failed (error {err.val}, {t:.2f}s).")
            continue

        rospy.loginfo(f"    Plan found in {t:.2f}s — executing.")
        if _execute_and_check(group, plan):
            return True

        rospy.logwarn(f"    Execution failed on attempt {attempt}.")
        rospy.sleep(0.3)

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
        self._lock                 = threading.Lock()
        self._pending_target       = None   # latest np.ndarray from subscriber
        self._busy                 = False  # True while a motion is executing
        self._consecutive_failures = 0      # full-failure count for thermal backoff

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
        # After this many consecutive full failures (all strategies exhausted),
        # pause to let overheated motors cool before trying again.
        THERMAL_BACKOFF_FAILURES = 3
        THERMAL_BACKOFF_SEC      = 20.0

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
                    success = self._point_at(target)
                    if success:
                        self._consecutive_failures = 0
                    else:
                        self._consecutive_failures += 1
                        if self._consecutive_failures >= THERMAL_BACKOFF_FAILURES:
                            rospy.logwarn(
                                "[look_at_point] %d consecutive full failures — "
                                "pausing %.0fs for motor cooldown.",
                                self._consecutive_failures, THERMAL_BACKOFF_SEC,
                            )
                            rospy.sleep(THERMAL_BACKOFF_SEC)
                            self._consecutive_failures = 0
                except Exception as e:
                    rospy.logerr("_point_at raised: %s", e)
                finally:
                    rospy.sleep(MIN_REPLAN_INTERVAL)   # let controller settle
                    with self._lock:
                        self._busy = False

            rate.sleep()

    # ── Core pointing logic ───────────────────────────────────────────────

    def _point_at(self, target_point: np.ndarray):
        """
        Orient the arm so EEF_POINT_AXIS (+X of link06) points at target_point.

        The pointing direction is computed from the arm base (world origin),
        NOT from the current EEF position.  This is critical: orientation-only
        goals let MoveIt move the EEF to any position, so basing the direction
        on eef_pos would give a different quaternion goal every time — even for
        the same target — because eef_pos drifts after each move.

        Since targets are ~5 m away and the arm is ~0.5 m long, the direction
        from the base is effectively the same as from the EEF, but stable.
        """
        rospy.loginfo(
            "[look_at_point] Processing target [%.3f, %.3f, %.3f] m",
            target_point[0], target_point[1], target_point[2],
        )

        # Read current EEF orientation once — used for both the angular-error
        # check and the goal quaternion.  Getting R_current here avoids a
        # second get_current_pose() call later.
        current_pose_q = self.group.get_current_pose().pose.orientation
        R_current  = quat_to_R(current_pose_q)
        cur_pointing = unit_vector(R_current @ EEF_POINT_AXIS)

        dist = np.linalg.norm(target_point)

        if dist < 1e-4:
            rospy.logwarn("Target at origin — ignoring.")
            return True

        target_dir = target_point / dist

        # Reject targets below the minimum elevation angle
        elevation_deg = math.degrees(math.asin(np.clip(target_dir[2], -1.0, 1.0)))
        if elevation_deg < MIN_ELEVATION_DEG:
            rospy.logwarn(
                f"Target elevation {elevation_deg:.1f}° is below minimum "
                f"{MIN_ELEVATION_DEG}° — ignoring (pointing underground)."
            )
            return True

        angle_err = angle_between_deg(cur_pointing, target_dir)

        rospy.loginfo(
            f"Target dir {np.round(target_dir, 3)}  |  "
            f"angular error {angle_err:.1f}°"
        )

        if angle_err < ALREADY_POINTING_THRESHOLD_DEG:
            rospy.loginfo("Already pointing (error %.1f°) — skipping.", angle_err)
            return True

        # Build the goal orientation by rotating the CURRENT EEF orientation
        # by the minimum rotation that aligns cur_pointing → target_dir.
        #
        # The old approach (rotation_matrix_from_vectors(EEF_POINT_AXIS, target_dir))
        # produced a canonical absolute orientation that baked in a specific roll
        # angle, which could be 90°+ away from joint6's current position.  On real
        # hardware that drove joint6 to overheat on every target change.
        #
        # This approach preserves the current roll — joint6 only moves the minimum
        # amount needed to redirect the arm, keeping wrist motor load low.
        R_delta = rotation_matrix_from_vectors(cur_pointing, target_dir)
        R_goal  = R_delta @ R_current
        quat    = R_to_quat(R_goal)
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
            return True

        # Strategy 2: full pose goal at current position with new orientation
        rospy.logwarn("Orientation-only failed \u2014 trying pose-at-current-position fallback \u2026")
        if plan_joint_fallback(self.group, quat):
            rospy.loginfo("SUCCESS (pose fallback).")
            return True

        rospy.logerr("All strategies exhausted for target %s.", np.round(target_point, 3))
        return False


def main():
    rospy.init_node("look_at_point", anonymous=False)
    LookAtPointNode()
    rospy.spin()


if __name__ == "__main__":

    main()