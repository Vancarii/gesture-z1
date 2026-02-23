#!/usr/bin/env python3
"""
look_at_point.py  (v4)
-----------------------
Fixes:
  1. EEF_POINT_AXIS corrected to +X (confirmed from tf link06→gripperStator).
  2. set_start_state_to_current_state() called before every plan to prevent
     CONTROL_FAILED caused by stale start-state vs actual controller position.
  3. execute() return value is properly checked.
  4. Velocity/acceleration reduced to give the controller more headroom.
  5. Retry loop on CONTROL_FAILED — re-reads joint state and replans.
"""

import sys
import math
import numpy as np

import rospy
import moveit_commander
from moveit_msgs.msg import OrientationConstraint, Constraints
from geometry_msgs.msg import Pose, Point, Quaternion
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

# Skip motion if already within this angular error
ALREADY_POINTING_THRESHOLD_DEG = 3.0

# Orientation constraint tolerance
ORIENTATION_TOLERANCE_DEG = 5.0

PLANNING_TIME   = 10.0
MAX_VELOCITY    = 0.15   # conservative — reduces CONTROL_FAILED risk
MAX_ACCEL       = 0.15

MAX_RETRIES     = 3      # re-read state and replan on CONTROL_FAILED

# Default demo target (metres, world frame)
TARGET_POINT    = np.array([0.5, 0.3, 0.8])

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
    Plan with an OrientationConstraint (position unconstrained).
    Retries up to MAX_RETRIES times on CONTROL_FAILED by re-reading
    the current joint state before each attempt.
    """
    tol = math.radians(ORIENTATION_TOLERANCE_DEG)
    q   = Quaternion(x=float(quat_xyzw[0]), y=float(quat_xyzw[1]),
                     z=float(quat_xyzw[2]), w=float(quat_xyzw[3]))

    for attempt in range(1, MAX_RETRIES + 1):
        rospy.loginfo(f"Orientation-only planning — attempt {attempt}/{MAX_RETRIES}")

        # Key fix: sync start state to live robot state before each plan
        _prepare_group(group)

        oc = OrientationConstraint()
        oc.header.frame_id           = BASE_FRAME
        oc.link_name                 = EEF_LINK
        oc.orientation               = q
        oc.absolute_x_axis_tolerance = tol
        oc.absolute_y_axis_tolerance = tol
        oc.absolute_z_axis_tolerance = tol
        oc.weight                    = 1.0

        constraints = Constraints()
        constraints.orientation_constraints.append(oc)
        group.set_path_constraints(constraints)
        group.set_planning_time(PLANNING_TIME)
        group.set_orientation_target(quat_xyzw.tolist(), EEF_LINK)

        plan_ok, plan, t, err = group.plan()
        group.clear_pose_targets()
        group.clear_path_constraints()

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
        rospy.sleep(0.5)   # brief pause to let the controller settle

    return False


def plan_pose_fallback(group, eef_pos: np.ndarray,
                       target: np.ndarray,
                       quat_xyzw: np.ndarray) -> bool:
    """
    Fallback: full pose goals stepped along the ray toward the target.
    eef_pos is the SNAPSHOT taken at script start — never re-read mid-loop
    so successive attempts don't drift.
    """
    direction = unit_vector(target - eef_pos)
    distances = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    q = Quaternion(x=float(quat_xyzw[0]), y=float(quat_xyzw[1]),
                   z=float(quat_xyzw[2]), w=float(quat_xyzw[3]))

    for d in distances:
        candidate = eef_pos + direction * d
        rospy.loginfo(f"  Pose fallback: ray +{d:.2f}m → {np.round(candidate, 3)}")

        for attempt in range(1, MAX_RETRIES + 1):
            _prepare_group(group)   # sync start state every attempt

            pose = Pose()
            pose.position    = Point(*candidate)
            pose.orientation = q
            group.set_pose_target(pose)

            plan_ok, plan, t, err = group.plan()
            group.clear_pose_targets()

            if not plan_ok:
                rospy.logwarn(
                    f"    Attempt {attempt}: planning failed "
                    f"(error {err.val}, {t:.2f}s)."
                )
                break   # no point retrying the same pose if planner fails

            rospy.loginfo(f"    Attempt {attempt}: plan found in {t:.2f}s — executing.")
            if _execute_and_check(group, plan):
                return True

            rospy.logwarn(f"    Attempt {attempt}: execution failed. Retrying …")
            rospy.sleep(0.5)

    return False


# ── Main ───────────────────────────────────────────────────────────────────

def point_arm_at(target_point: np.ndarray) -> bool:
    moveit_commander.roscpp_initialize(sys.argv)
    moveit_commander.RobotCommander()
    moveit_commander.PlanningSceneInterface()
    group = moveit_commander.MoveGroupCommander(PLANNING_GROUP)

    group.set_max_velocity_scaling_factor(MAX_VELOCITY)
    group.set_max_acceleration_scaling_factor(MAX_ACCEL)
    group.set_pose_reference_frame(BASE_FRAME)
    group.set_end_effector_link(EEF_LINK)

    # Snapshot EEF state once at startup
    eef_pose = group.get_current_pose().pose
    eef_pos  = np.array([eef_pose.position.x,
                          eef_pose.position.y,
                          eef_pose.position.z])

    cur_pointing = current_pointing_direction(group)
    rospy.loginfo(
        f"\n"
        f"  EEF position (world)   : {np.round(eef_pos, 4)}\n"
        f"  Current pointing dir   : {np.round(cur_pointing, 4)}\n"
        f"  Target point   (world) : {np.round(target_point, 4)}"
    )

    offset = target_point - eef_pos
    dist   = np.linalg.norm(offset)
    if dist < 1e-4:
        rospy.logerr("Target is at the EEF origin.")
        return False

    target_dir  = offset / dist
    angle_err   = angle_between_deg(cur_pointing, target_dir)
    rospy.loginfo(
        f"  Desired pointing dir   : {np.round(target_dir, 4)}\n"
        f"  Angular error          : {angle_err:.2f}°  "
        f"(threshold {ALREADY_POINTING_THRESHOLD_DEG}°)"
    )

    if angle_err < ALREADY_POINTING_THRESHOLD_DEG:
        rospy.loginfo("Already pointing at target within tolerance — no motion needed.")
        return True

    R    = rotation_matrix_from_vectors(EEF_POINT_AXIS, target_dir)
    quat = R_to_quat(R)
    rospy.loginfo(f"  Goal quat (xyzw)       : {np.round(quat, 4)}")

    # Strategy 1: orientation-only (preferred — no position drift)
    if plan_orientation_only(group, quat):
        rospy.loginfo("SUCCESS (orientation-only).")
        return True

    # Strategy 2: pose goals along ray (fallback)
    rospy.logwarn("Orientation-only failed. Trying ray-sampled pose fallback …")
    if plan_pose_fallback(group, eef_pos, target_point, quat):
        rospy.loginfo("SUCCESS (pose fallback).")
        return True

    rospy.logerr(
        "All strategies exhausted.\n\n"
        "Checklist:\n"
        "  - EEF_POINT_AXIS is now [1,0,0] (confirmed from tf). "
        "If the arm still points wrong, re-run:\n"
        "      rosrun tf tf_echo link06 gripperStator\n"
        "    and check if the axis changed after moving.\n\n"
        "  - CONTROL_FAILED persists? Check controller tolerances in:\n"
        "      $(rospack find z1_description)/config/ros_controllers.yaml\n"
        "    Increase `stopped_velocity_tolerance` and "
        "`goal_time` parameters.\n\n"
        "  - Add collision geometry to link01 and link05 in the URDF;\n"
        "    missing collision shapes make self-collision detection unreliable.\n\n"
        "  - Try reducing MAX_VELOCITY/MAX_ACCEL further (e.g. 0.05)."
    )
    return False


def main():
    rospy.init_node("z1_point_to_target", anonymous=False)

    if len(sys.argv) == 4:
        try:
            target = np.array([float(sys.argv[1]),
                               float(sys.argv[2]),
                               float(sys.argv[3])])
        except ValueError:
            rospy.logerr("Arguments must be three floats: x y z")
            sys.exit(1)
    else:
        rospy.logwarn(
            f"No target given — using demo target {TARGET_POINT.tolist()}. "
            "Pass x y z as CLI args to override."
        )
        target = TARGET_POINT

    ok = point_arm_at(target)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()