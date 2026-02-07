#!/usr/bin/env python3
"""
ROS node: gesture_moveit_controller

Subscribes to /leap/gesture (JSON action commands published by
gesture_detector_ml) and moves the Z1 arm via MoveIt.

The detector publishes:
  { gesture, confidence, cmdid, velocity, targetpos, grippercmd }

This node uses the *velocity* vector as a directional hint, applies a
coordinate transform from "Brain space" to the robot base-link frame,
and executes a small Cartesian step using MoveIt.

Coordinate mapping (Brain -> Z1 robot base_link):
  robot_x (forward)  =  brain_y   (swipe towards = forward)
  robot_y (left)     = -brain_x   (point left   = left)
  robot_z (up)       =  brain_z   (point up     = up)
"""

import rospy
import sys
import json
import numpy as np
from std_msgs.msg import String

import moveit_commander
from geometry_msgs.msg import Pose
import actionlib
from moveit_msgs.msg import MoveGroupAction


class GestureMoveItController:

    # Default Brain-to-robot coordinate transform
    # (rows = robot axes, columns = brain axes)
    BRAIN_TO_ROBOT = np.array([
        [ 0,  1,  0],   # robot_x = brain_y
        [-1,  0,  0],   # robot_y = -brain_x
        [ 0,  0,  1],   # robot_z = brain_z
    ], dtype=float)

    def __init__(self):
        rospy.init_node("gesture_moveit_controller")

        # ---------- tuneable parameters ----------
        self.step_scale          = rospy.get_param("~step_scale", 0.04)
        self.velocity_scale      = rospy.get_param("~velocity_scale", 0.3)
        self.acceleration_scale  = rospy.get_param("~acceleration_scale", 0.3)
        self.min_confidence      = rospy.get_param("~min_confidence", 0.85)
        self.planning_time       = rospy.get_param("~planning_time", 1.0)
        self.cartesian_step_res  = rospy.get_param("~cartesian_step_res", 0.005)
        self.min_cartesian_frac  = rospy.get_param("~min_cartesian_fraction", 0.3)
        self.workspace_bounds    = rospy.get_param(
            "~workspace_bounds",
            {"x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [0.05, 0.6]},
        )

        # ---------- MoveIt setup ----------
        moveit_commander.roscpp_initialize(sys.argv)

        rospy.loginfo("Waiting for move_group action server ...")
        client = actionlib.SimpleActionClient("move_group", MoveGroupAction)
        if not client.wait_for_server(timeout=rospy.Duration(30.0)):
            rospy.logfatal("move_group server not available!")
            sys.exit(1)
        rospy.loginfo("move_group server connected")

        self.group = moveit_commander.MoveGroupCommander("manipulator")
        self.group.set_max_velocity_scaling_factor(self.velocity_scale)
        self.group.set_max_acceleration_scaling_factor(self.acceleration_scale)
        self.group.set_planning_time(self.planning_time)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()

        # ---------- state ----------
        self.busy = False
        self.last_gesture = None

        # ---------- subscriber ----------
        rospy.Subscriber(
            "/leap/gesture", String, self.gesture_cb, queue_size=5
        )

        rospy.loginfo(
            "GestureMoveItController ready (step=%.3f m)", self.step_scale
        )

    # ----------------------------------------------------------------
    # Callback
    # ----------------------------------------------------------------
    def gesture_cb(self, msg):
        if self.busy:
            return

        # --- parse JSON message (with fallback to legacy format) ---
        try:
            data = json.loads(msg.data)
        except (json.JSONDecodeError, ValueError):
            try:
                parts = msg.data.split(":")
                data = {
                    "gesture": parts[0],
                    "confidence": float(parts[1]),
                    "velocity": [0, 0, 0],
                    "grippercmd": 0.0,
                }
            except Exception:
                rospy.logwarn("Bad gesture message: %s", msg.data)
                return

        gesture    = data.get("gesture", "")
        confidence = data.get("confidence", 0.0)
        velocity   = np.array(data.get("velocity", [0, 0, 0]), dtype=float)
        grippercmd = data.get("grippercmd", 0.0)

        # Skip low confidence / idle gestures
        if confidence < self.min_confidence:
            return
        if gesture in ("background", "pause", "NAN"):
            return
        if np.linalg.norm(velocity) < 0.01:
            return

        rospy.loginfo(
            "Gesture: %s  conf=%.2f  vel=%s", gesture, confidence, velocity
        )

        self.busy = True
        try:
            self.execute_motion(velocity, grippercmd)
        except Exception as e:
            rospy.logwarn("Motion execution error: %s", e)
        finally:
            self.busy = False
            self.last_gesture = gesture

    # ----------------------------------------------------------------
    # Motion execution
    # ----------------------------------------------------------------
    def execute_motion(self, velocity, grippercmd):
        current_pose = self.group.get_current_pose().pose

        # Normalise velocity to unit direction, then scale
        vel_norm = np.linalg.norm(velocity)
        if vel_norm < 0.01:
            return
        direction = velocity / vel_norm

        # Transform Brain coords -> robot base_link coords
        robot_dir = self.BRAIN_TO_ROBOT @ direction
        step = robot_dir * self.step_scale

        # Build target pose (keep current orientation)
        target_pose = Pose()
        target_pose.position.x = current_pose.position.x + step[0]
        target_pose.position.y = current_pose.position.y + step[1]
        target_pose.position.z = current_pose.position.z + step[2]
        target_pose.orientation = current_pose.orientation

        # Enforce workspace bounds
        wb = self.workspace_bounds
        target_pose.position.x = np.clip(
            target_pose.position.x, wb["x"][0], wb["x"][1]
        )
        target_pose.position.y = np.clip(
            target_pose.position.y, wb["y"][0], wb["y"][1]
        )
        target_pose.position.z = np.clip(
            target_pose.position.z, wb["z"][0], wb["z"][1]
        )

        # Plan via Cartesian path (fast for small incremental steps)
        # avoid_collisions=False because link02-gripperStator self-collision
        # is a known false positive in many Z1 configs.
        waypoints = [target_pose]
        (plan, fraction) = self.group.compute_cartesian_path(
            waypoints, self.cartesian_step_res, False
        )

        if fraction < self.min_cartesian_frac:
            # Even partial Cartesian failed, fall back to RRT
            rospy.logwarn(
                "Cartesian path %.0f%% - falling back to pose target",
                fraction * 100,
            )
            self.group.set_pose_target(target_pose)
            plan_result = self.group.plan()
            if isinstance(plan_result, tuple):
                success = plan_result[0]
                plan = plan_result[1]
            else:
                success = True
                plan = plan_result
            if not success or len(plan.joint_trajectory.points) == 0:
                rospy.logwarn("Planning failed")
                return
        elif fraction < 1.0:
            rospy.loginfo(
                "Cartesian path %.0f%% - executing partial", fraction * 100
            )

        self.group.execute(plan, wait=True)
        self.group.stop()
        self.group.clear_pose_targets()


# ----------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------
def main():
    controller = GestureMoveItController()
    rospy.spin()
    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
