#!/usr/bin/env python3
"""
Subscribes to /leap/pointing_target published by detect_point.py and
continuously drives the Z1 arm to point at whatever 3D coordinate the
Leap Motion detects.

Topic consumed:
    /leap/pointing_target  (std_msgs/String, JSON)
    {"target_pos_m": [x, y, z], "frame": "arm_base"}
"""

import sys
import math
import threading
import json
from collections import deque
import numpy as np

import rospy
import moveit_commander
from geometry_msgs.msg import Quaternion
from std_msgs.msg import String, Bool
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
# Disabled — the ground plane and frame in the MoveIt scene handle this now.
# MIN_ELEVATION_DEG = -30.0

PLANNING_TIME = 10.0
MAX_VELOCITY = 0.6    # arm velocity
MAX_ACCEL = 0.3

# Hard timeout for group.execute().  On real hardware the trajectory controller
# can accept a goal but never send a terminal result (stuck ACTIVE state), which
# blocks group.execute(wait=True) forever.  This timeout cancels the goal and
# recovers the worker thread so subsequent gestures keep working.
EXECUTE_TIMEOUT_SEC = 30.0

MAX_RETRIES = 2

# Settle time after a motion before accepting the next target.
MIN_REPLAN_INTERVAL = 1.0

# Stored pointing targets are represented as 3D coordinates in robot world
# frame on a sphere centered at the arm base.
STORED_TARGET_DISTANCE_M = 5.0

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


# Parse commands like 'assign,WEATHER_REPORT' or 'check,WEATHER_REPORT'.
def parse_command_message(raw: str):
    parts = raw.split(",", 1)
    if len(parts) != 2:
        raise ValueError("expected '<command>,<keyword>'")

    command = parts[0].strip().lower()
    keyword = parts[1].strip()
    if not command or not keyword:
        raise ValueError("command and keyword must be non-empty")
    if command not in {"assign", "check"}:
        raise ValueError(f"unsupported command '{command}'")
    return command, keyword


def normalize_keyword(keyword: str) -> str:
    return keyword.strip().upper()


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


def _execute_and_check(group, plan) -> bool:
    """
    Execute a plan and return True only if the controller confirms success.
    Run execute() in a daemon thread. If it hasn't finished within
    EXECUTE_TIMEOUT_SEC, call group.stop() to cancel the goal
    """
    result_holder = [None]

    def _run():
        result_holder[0] = group.execute(plan, wait=True)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=EXECUTE_TIMEOUT_SEC)

    if t.is_alive():
        rospy.logerr(
            "group.execute() timed out after %.0fs — ",
            EXECUTE_TIMEOUT_SEC,
        )
        try:
             # cancels the action goal
            group.stop()
        except Exception:
            pass
        # wait for _run() to actually return
        t.join(timeout=5.0)
        group.clear_pose_targets()
        group.clear_path_constraints()
        # Let the controller settle and the joint-state monitor update before
        # the next planning attempt reads start state.
        rospy.sleep(1.0)
        return False

    group.stop()
    group.clear_pose_targets()
    group.clear_path_constraints()
    result = result_holder[0]
    if not result:
        rospy.logerr("group.execute() returned False — controller rejected trajectory.")
    return bool(result)


def _plan_and_execute(group, quat_xyzw: np.ndarray) -> bool:
    """
    Plan and execute an orientation-only goal.

    set_orientation_target is the correct API for pointing tasks: position is
    unconstrained, only the EEF orientation must be achieved.  TRAC-IK samples
    random EEF positions from the reachable workspace and solves for joint
    configs that satisfy the orientation — this works regardless of where the
    arm currently is.

    set_pose_target (position + orientation) fails when the CURRENT EEF position
    has no IK solution for the desired orientation (e.g. near-singular home
    config), causing 'unable to sample valid states for goal tree'.  That is
    why position-anchored goals were unreliable.

    No path constraints are applied.  The goal quaternion is computed as
    R_delta @ R_current (minimum rotation from current pointing direction to
    target direction), which naturally preserves joint6 (wrist roll).  TRAC-IK
    seeded from the current joint state finds the nearest-configuration IK
    solution, so joint6 only moves the minimum necessary amount.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        _prepare_group(group)

        rospy.loginfo(f"Planning attempt {attempt}/{MAX_RETRIES}")

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

        # Controller rejected — check if the arm actually reached the goal.
        # Spurious GOAL_TOLERANCE_VIOLATED can fire even when the arm is on target.
        goal_dir = pointing_dir_from_quat(quat_xyzw)
        cur_dir  = current_pointing_direction(group)
        angular_err = angle_between_deg(cur_dir, goal_dir)
        if angular_err < CLOSE_ENOUGH_DEG:
            rospy.loginfo(
                f"  Controller rejected but arm is within {angular_err:.1f}° "
                f"of goal (<{CLOSE_ENOUGH_DEG}°) — accepting."
            )
            return True

        rospy.logwarn(
            f"  Execution failed on attempt {attempt} — "
            f"{angular_err:.1f}° from goal. Retrying …"
        )
        rospy.sleep(0.3)

    return False


# ── ROS Node ───────────────────────────────────────────────────────────────

class LookAtPointNode:
    """
    Persistent ROS node that listens to /leap/pointing_target (published by
    detect_point.py) and continuously orients the arm to point at that target.

    It also listens to /ventriloquism_commands. 'assign,<keyword>' stores the
    robot's current pointing target under that keyword, and 'check,<keyword>'
    replays the stored target.

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
        self._lock = threading.Lock()
        self._pending_target = None   # latest np.ndarray from subscriber
        self._pending_home   = False  # True when /leap/home has been received
        self._pending_commands = deque()
        self._busy = False  # True while a motion is executing
        self._consecutive_failures = 0      # full-failure count for thermal backoff
        self._current_target = None  # last successful pointing target in robot frame
        self._saved_targets = {}     # normalized keyword -> {label: str, target: np.ndarray}

        # ── ROS subscribers ────────────────────────────────────────
        rospy.Subscriber(
            "/leap/pointing_target", String,
            self._target_cb, queue_size=1
        )
        rospy.Subscriber(
            "/leap/home", Bool,
            self._home_cb, queue_size=1
        )
        rospy.Subscriber(
            "/ventriloquism_commands", String,
            self._command_cb, queue_size=10
        )

        # ── Worker thread (owns all MoveIt calls) ─────────────────────────
        t = threading.Thread(target=self._worker_loop, daemon=True)
        t.start()

        rospy.loginfo(
            "[look_at_point] Ready — subscribed to /leap/pointing_target, "
            "/leap/home, and /ventriloquism_commands"
        )

    # ── Subscriber callbacks ────────────────────────────────────

    def _home_cb(self, msg: Bool):
        """Preempt any pending pointing target and request a home motion."""
        if not msg.data:
            return
        rospy.loginfo("[look_at_point] /leap/home received — queuing home motion.")
        with self._lock:
            self._pending_home   = True
            self._pending_target = None   # discard any queued pointing target

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

    def _command_cb(self, msg: String):
        """Queue assign/check commands for the worker thread to handle."""
        try:
            command, keyword = parse_command_message(msg.data)
        except ValueError as exc:
            rospy.logwarn_throttle(
                5.0,
                "Malformed /ventriloquism_commands '%s': %s",
                msg.data,
                exc,
            )
            return

        rospy.loginfo(
            "[look_at_point] Command received on /ventriloquism_commands: %s,%s",
            command,
            keyword,
        )
        with self._lock:
            self._pending_commands.append((command, keyword))

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
            target      = None
            do_home     = False
            command     = None
            with self._lock:
                if not self._busy:
                    if self._pending_commands:
                        command            = self._pending_commands.popleft()
                        self._busy         = True
                    elif self._pending_home:
                        do_home            = True
                        self._pending_home = False
                        self._busy         = True
                    elif self._pending_target is not None:
                        target               = self._pending_target
                        self._pending_target = None
                        self._busy           = True

            if command is not None:
                try:
                    success = self._handle_command(*command)
                    if success:
                        self._consecutive_failures = 0
                except Exception as e:
                    rospy.logerr("_handle_command raised: %s", e)
                finally:
                    rospy.sleep(MIN_REPLAN_INTERVAL)
                    with self._lock:
                        self._busy = False

            elif do_home:
                try:
                    success = self._do_home()
                    if success:
                        self._consecutive_failures = 0
                    else:
                        self._consecutive_failures += 1
                except Exception as e:
                    rospy.logerr("_do_home raised: %s", e)
                finally:
                    rospy.sleep(MIN_REPLAN_INTERVAL)
                    with self._lock:
                        self._busy = False

            elif target is not None:
                do_backoff = False
                try:
                    success = self._point_at(target)
                    if success:
                        self._consecutive_failures = 0
                    else:
                        self._consecutive_failures += 1
                        if self._consecutive_failures >= THERMAL_BACKOFF_FAILURES:
                            do_backoff = True
                            self._consecutive_failures = 0
                except Exception as e:
                    rospy.logerr("_point_at raised: %s", e)
                finally:
                    rospy.sleep(MIN_REPLAN_INTERVAL)   # let controller settle
                    with self._lock:
                        self._busy = False
                # Thermal backoff runs AFTER _busy is cleared so gestures can
                # queue normally during the cooldown period.
                if do_backoff:
                    rospy.logwarn(
                        "[look_at_point] %d consecutive full failures — "
                        "pausing %.0fs for motor cooldown.",
                        THERMAL_BACKOFF_FAILURES, THERMAL_BACKOFF_SEC,
                    )
                    rospy.sleep(THERMAL_BACKOFF_SEC)

            rate.sleep()

    def _current_target_from_pose(self) -> np.ndarray:
        """Infer current far-field pointing target from the live EEF pose."""
        direction = current_pointing_direction(self.group)
        return unit_vector(direction) * STORED_TARGET_DISTANCE_M

    def _handle_command(self, command: str, keyword: str) -> bool:
        normalized = normalize_keyword(keyword)

        if command == "assign":
            if self._current_target is not None:
                target = self._current_target.copy()
            else:
                target = self._current_target_from_pose()

            self._saved_targets[normalized] = {
                "label": keyword,
                "target": target.copy(),
            }
            rospy.loginfo(
                "[look_at_point] Assigned keyword '%s' to target [%.3f, %.3f, %.3f] m",
                keyword,
                target[0], target[1], target[2],
            )
            return True

        saved = self._saved_targets.get(normalized)
        if saved is None:
            rospy.logwarn("[look_at_point] No saved target for keyword '%s'", keyword)
            return False

        target = saved["target"].copy()
        rospy.loginfo(
            "[look_at_point] Recalling keyword '%s' -> [%.3f, %.3f, %.3f] m",
            saved["label"],
            target[0], target[1], target[2],
        )
        return self._point_at(target)

    # ── Core pointing logic ───────────────────────────────────────────────
    def _do_home(self) -> bool:
        """
        Move the arm to the named 'home' position defined in the SRDF
        (all joints near zero).  Called from the worker thread.
        """
        rospy.loginfo("[look_at_point] Homing — moving to 'home' named target.")
        for attempt in range(1, MAX_RETRIES + 1):
            _prepare_group(self.group)
            self.group.set_planning_time(PLANNING_TIME)
            self.group.set_named_target("home")
            plan_ok, plan, t, err = self.group.plan()
            self.group.clear_pose_targets()
            if not plan_ok:
                rospy.logwarn(
                    "Homing: planning failed (err %s, %.2fs), attempt %d/%d.",
                    err.val, t, attempt, MAX_RETRIES,
                )
                continue
            rospy.loginfo("Homing: plan found in %.2fs — executing.", t)
            if _execute_and_check(self.group, plan):
                self._current_target = None
                rospy.loginfo("Homing SUCCESS.")
                return True
            rospy.logwarn("Homing: execution failed on attempt %d.", attempt)
            rospy.sleep(0.3)
        rospy.logerr("Homing: all attempts exhausted.")
        return False
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

        # Elevation pre-filter disabled — scene collision objects (ground plane,
        # frame) now prevent the planner from accepting underground trajectories.
        # elevation_deg = math.degrees(math.asin(np.clip(target_dir[2], -1.0, 1.0)))
        # if elevation_deg < MIN_ELEVATION_DEG:
        #     rospy.logwarn(
        #         f"Target elevation {elevation_deg:.1f}° is below minimum "
        #         f"{MIN_ELEVATION_DEG}° — ignoring (pointing underground)."
        #     )
        #     return True

        angle_err = angle_between_deg(cur_pointing, target_dir)

        rospy.loginfo(
            f"Target dir {np.round(target_dir, 3)}  |  "
            f"angular error {angle_err:.1f}°"
        )

        if angle_err < ALREADY_POINTING_THRESHOLD_DEG:
            self._current_target = target_point.copy()
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

        self.group.set_max_velocity_scaling_factor(MAX_VELOCITY)
        self.group.set_max_acceleration_scaling_factor(MAX_ACCEL)

        if _plan_and_execute(self.group, quat):
            self._current_target = target_point.copy()
            rospy.loginfo("SUCCESS.")
            return True

        rospy.logerr("Planning failed for target %s.", np.round(target_point, 3))
        return False


def main():
    rospy.init_node("look_at_point", anonymous=False)
    LookAtPointNode()
    rospy.spin()


if __name__ == "__main__":

    main()