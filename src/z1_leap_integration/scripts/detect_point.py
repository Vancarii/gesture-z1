#!/usr/bin/env -S /bin/bash -c 'conda run -n gesture-env python "$0" "$@"'
# -*- coding: utf-8 -*-
"""
ROS node for Leap hand pointing detection.
"""

import sys
import os
import ctypes
import math

import time
import json
import numpy as np
from collections import deque

# Path setup
_SCRIPT_DIR   = os.path.dirname(os.path.realpath(__file__))
_WS_SRC       = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
Z1_LEAPC_ROOT = os.path.join(_WS_SRC, "Z1-LeapC")
Z1_LEAPC_SRC  = os.path.join(Z1_LEAPC_ROOT, "src")
sys.path.insert(0, Z1_LEAPC_SRC)

# Ensure rospy is importable under conda python.
for _p in [
    "/usr/lib/python3/dist-packages",
    "/opt/ros/noetic/lib/python3/dist-packages",
    "/home/tangentlab/z1_ws/devel/lib/python3/dist-packages",
]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)


import rospy
from std_msgs.msg import String, Bool

import leap
import cv2

# TrackingMode import (supports older leap layouts)
try:
    TrackingMode = leap.TrackingMode
except AttributeError:
    try:
        from leap.enums import TrackingMode as _TM
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "TrackingMode enum unavailable from the leap module. "
            "Ensure leapc-python-api is installed correctly."
        ) from exc
    TrackingMode = _TM
    setattr(leap, "TrackingMode", TrackingMode)


def _pointing_label(direction):
    """
    Return a compact direction label from a unit vector.
    Leap desktop axes: +X = right, +Y = up, +Z = toward user (i.e. -Z = away/forward).
    Only axes whose component exceeds 0.25 are included.
    """
    d = np.array(direction)
    pos_names = ['R', 'Up', 'ToUser']
    neg_names = ['L', 'Dn', 'Fwd']
    parts = []
    for i in range(3):
        if abs(d[i]) > 0.25:
            parts.append(pos_names[i] if d[i] > 0 else neg_names[i])
    return '-'.join(parts) if parts else 'Pointing'


def _angle_between_deg(a, b) -> float:
    """Angle in degrees between two vectors (handles near-zero norms)."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return math.degrees(math.acos(np.clip(np.dot(a / na, b / nb), -1.0, 1.0)))


# Physical setup (mm)
ARM_DISTANCE_FROM_LEAP_MM: float = 1000.0
ARM_VERTICAL_OFFSET_MM:   float = 500.0

LEAP_TO_ARM_OFFSET_MM: np.ndarray = np.array([
    0.0,
    -ARM_VERTICAL_OFFSET_MM,
    -ARM_DISTANCE_FROM_LEAP_MM,
])

OBJECT_DISTANCE_M:  float = 5.0                        # 5 metres
OBJECT_DISTANCE_MM: float = OBJECT_DISTANCE_M * 1000.0  # 5000 mm

# Stability / debounce
STABILITY_WINDOW_SEC = 1.0    # seconds the direction must be stable
STABILITY_MAX_DEG    = 12.0    # max angular spread across the whole window
MOVEMENT_RESET_DEG   = 20.0   # frame-to-frame jump that clears the window
RELOCK_ESCAPE_DEG    = 15.0   # hand must move this far from the locked dir
                               # before a new lock is allowed

# Leap sensor tilt correction
LEAP_TILT_DEG: float = 30.0
_ct = np.cos(np.radians(LEAP_TILT_DEG))
_st = np.sin(np.radians(LEAP_TILT_DEG))
LEAP_TILT_R: np.ndarray = np.array([
    [1.0,  0.0,  0.0],
    [0.0,  _ct, -_st],
    [0.0,  _st,  _ct],
], dtype=float)
del _ct, _st


class Canvas:

    def __init__(self):
        self.name = "Leap Gesture Detector Visualization"
        self.screen_size = [500, 700]
        self.hands_colour = (255, 255, 255)
        self.font_colour  = (0, 255, 44)
        self.hands_format = "Skeleton"
        self.output_image = np.zeros(
            (self.screen_size[0], self.screen_size[1], 3), np.uint8)
        self.predict = "NAN"
        self.tracking_mode = None
        self.current_direction = "No hands detected"
        self.palm_direction_text = ""
        self.target_text = ""
        self.stability_text = ""

    def get_joint_position(self, bone):
        if bone is None:
            return None
        SCALE = 1.0
        return (
            int(bone.x * SCALE + (self.screen_size[1] / 2)),
            int((self.screen_size[0] / 2) - (bone.z * SCALE)),
        )

    def render_hands(self, event):
        self.output_image[:, :] = 0

        cv2.putText(
            self.output_image, f"Direction: {self.current_direction}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.font_colour, 2,
        )

        cv2.putText(
            self.output_image, self.palm_direction_text,
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.font_colour, 1,
        )

        cv2.putText(
            self.output_image, self.target_text,
            (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1,
        )

        if self.stability_text:
            colour = (0, 255, 0) if self.stability_text.startswith("LOCKED") else (0, 165, 255)
            cv2.putText(
                self.output_image, self.stability_text,
                (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2,
            )

        _TRACKING_NAMES = {
            TrackingMode.Desktop: "Desktop",
            TrackingMode.HMD: "HMD",
            TrackingMode.ScreenTop: "ScreenTop",
        }
        cv2.putText(
            self.output_image, 
            f"Tracking Mode: {_TRACKING_NAMES.get(self.tracking_mode, 'Unknown')}",
            (10, self.screen_size[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.font_colour, 1,
        )

        if len(event.hands) == 0:
            return

        for i in range(len(event.hands)):
            hand = event.hands[i]

            palm_pos = self.get_joint_position(hand.palm.position)
            if palm_pos:
                cv2.circle(self.output_image, palm_pos, 8, self.hands_colour, -1)
                cv2.putText(
                    self.output_image, "PALM",
                    (palm_pos[0] + 10, palm_pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.hands_colour, 1,
                )

                palm_dir = hand.palm.direction
                arrow_end_x = int(palm_pos[0] + palm_dir.x * 50)
                arrow_end_y = int(palm_pos[1] - palm_dir.z * 50)
                cv2.arrowedLine(
                    self.output_image, palm_pos,
                    (arrow_end_x, arrow_end_y), (0, 255, 0), 3,
                )

            wrist = self.get_joint_position(hand.arm.next_joint)
            elbow = self.get_joint_position(hand.arm.prev_joint)
            if wrist:
                cv2.circle(self.output_image, wrist, 3, self.hands_colour, -1)
            if elbow:
                cv2.circle(self.output_image, elbow, 3, self.hands_colour, -1)
            if wrist and elbow:
                cv2.line(self.output_image, wrist, elbow, self.hands_colour, 2)

            for index_digit in range(5):
                digit = hand.digits[index_digit]
                for index_bone in range(4):
                    bone = digit.bones[index_bone]

                    bone_start = self.get_joint_position(bone.prev_joint)
                    bone_end   = self.get_joint_position(bone.next_joint)

                    if bone_start:
                        cv2.circle(self.output_image, bone_start, 3, self.hands_colour, -1)
                    if bone_end:
                        cv2.circle(self.output_image, bone_end, 3, self.hands_colour, -1)
                    if bone_start and bone_end:
                        cv2.line(self.output_image, bone_start, bone_end, self.hands_colour, 2)

                    if index_digit == 1 and index_bone == 3:
                        if bone_end:
                            cv2.circle(self.output_image, bone_end, 6, (0, 255, 0), -1)
                            cv2.putText(
                                self.output_image, "INDEX",
                                (bone_end[0] + 10, bone_end[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
                            )

                    if index_bone == 0 and bone_start and wrist:
                        cv2.line(self.output_image, bone_start, wrist, self.hands_colour, 2)


class GestureDetectorROS(leap.Listener):
    """
        Leap Motion listener for pointing direction and target publishing.
    """

    def __init__(self, show_viz=True):
        super().__init__()

        self.pub        = rospy.Publisher("/leap/gesture",          String, queue_size=10)
        self.target_pub = rospy.Publisher("/leap/pointing_target", String, queue_size=10)
        self.home_pub   = rospy.Publisher("/leap/home",            Bool,   queue_size=1)

        self.current_point_tip = None   # [x, y, z] mm in Leap world space
        self.current_point_dir = None   # unit vector [dx, dy, dz]
        self.current_target_m  = None   # [x, y, z] m  in arm base frame

        self.show_viz = show_viz
        if self.show_viz:
            self.canvas = Canvas()

        self._dir_history    = deque()   # (timestamp_sec, unit_dir_np)
        self._stable_since   = None      # time.time() when current stable run began
        self._stable_locked  = False     # True once published for this stable period
        self._last_locked_dir = None     # direction at the time of last publish

        self._two_hand_start   = None   # time.time() when 2-hand window began
        self._home_triggered   = False  # True once published for this 2-hand window
        self.TWO_HAND_HOME_SEC = 1.0

        rospy.loginfo("GestureDetectorROS initialised")

    def _check_stability(self, dir_now, target_m) -> str:
        """
        Publish a target once direction is stable over the configured window.
        """
        now = time.time()
        dir_now = np.array(dir_now, dtype=float)

        if self._stable_locked and self._last_locked_dir is not None:
            escape_angle = _angle_between_deg(dir_now, self._last_locked_dir)
            if escape_angle < RELOCK_ESCAPE_DEG:
                return "LOCKED (holding)"
            else:
                self._stable_locked   = False
                self._last_locked_dir = None
                self._dir_history.clear()
                self._stable_since    = None

        if self._dir_history:
            jump = _angle_between_deg(dir_now, self._dir_history[-1][1])
            if jump > MOVEMENT_RESET_DEG:
                self._dir_history.clear()
                self._stable_since  = None
                self._stable_locked = False

        if not self._dir_history:
            self._stable_since = now

        self._dir_history.append((now, dir_now))

        while self._dir_history and (now - self._dir_history[0][0]) > STABILITY_WINDOW_SEC:
            self._dir_history.popleft()

        run_dur = (now - self._stable_since) if self._stable_since is not None else 0.0

        if run_dur < STABILITY_WINDOW_SEC:
            self._stable_locked = False
            return f"Stabilizing: {run_dur:.1f}/{STABILITY_WINDOW_SEC:.0f}s"

        if not self._dir_history:
            return "Stabilizing: bad data"
        dirs = np.array([d for _, d in self._dir_history])
        mean_dir = dirs.mean(axis=0)
        norm_val = np.linalg.norm(mean_dir)
        if norm_val < 1e-9:
            return "Stabilizing: bad data"
        mean_dir /= norm_val
        max_spread = max(_angle_between_deg(d, mean_dir) for d in dirs)

        if max_spread > STABILITY_MAX_DEG:
            self._stable_locked = False
            return f"Moving (spread {max_spread:.1f}\u00b0)"

        if not self._stable_locked:
            self._stable_locked   = True
            self._last_locked_dir = dir_now.copy()
            self.target_pub.publish(json.dumps({
                "target_pos_m": list(target_m),
                "frame":        "arm_base",
            }))
            rospy.loginfo(
                "Stable %.1fs, spread %.1f\u00b0 \u2192 publishing target [%s] m",
                run_dur, max_spread,
                ", ".join(f"{v:.3f}" for v in target_m),
            )
            return "LOCKED \u2713"

        return "LOCKED (holding)"

    def _compute_target_point(self, tip_mm, direction):
        """
        Intersect the pointing ray with the fixed-radius target sphere.

        Parameters
        ----------
        tip_mm    : array-like (3,)  fingertip position in Leap frame, mm
        direction : array-like (3,)  unit pointing direction vector

        Returns
        -------
        list[float]  target [x, y, z] in arm base frame, **metres**
        """
        p = np.array(tip_mm, dtype=float) - LEAP_TO_ARM_OFFSET_MM
        d = np.array(direction, dtype=float)

        pdotd        = float(np.dot(p, d))
        discriminant = pdotd ** 2 - (float(np.dot(p, p)) - OBJECT_DISTANCE_MM ** 2)

        if discriminant >= 0.0:
            t = -pdotd + np.sqrt(discriminant)
            target_arm_mm = p + t * d
        else:
            target_arm_mm = d * OBJECT_DISTANCE_MM

        return (target_arm_mm / 1000.0).tolist()

    def _index_tip_and_direction(self, hand):
        """
        Estimate index-finger pointing from whole-finger geometry.

        Primary estimate is metacarpal base -> fingertip, which is much less
        noisy than only using the distal bone. If that vector degenerates,
        fall back to a weighted sum of all index-finger bone vectors.

        Returns
        -------
        tip_np : np.ndarray shape (3,)
            Fingertip position in Leap device frame (mm).
        dir_np : np.ndarray shape (3,)
            Unit pointing direction in Leap device frame.
        """
        idx = hand.digits[1]  # index finger

        base_joint = idx.bones[0].prev_joint
        tip_joint = idx.bones[3].next_joint
        base = np.array([base_joint.x, base_joint.y, base_joint.z], dtype=float)
        tip = np.array([tip_joint.x, tip_joint.y, tip_joint.z], dtype=float)
        whole_vec = tip - base
        whole_norm = np.linalg.norm(whole_vec)
        if whole_norm > 1e-6:
            return tip, (whole_vec / whole_norm)

        # Fallback: weighted blend of index finger segments.
        weights = np.array([1.0, 1.2, 1.4, 1.8], dtype=float)
        acc = np.zeros(3, dtype=float)
        for i, b in enumerate(idx.bones):
            s = np.array([b.prev_joint.x, b.prev_joint.y, b.prev_joint.z], dtype=float)
            e = np.array([b.next_joint.x, b.next_joint.y, b.next_joint.z], dtype=float)
            v = e - s
            n = np.linalg.norm(v)
            if n > 1e-6:
                acc += weights[i] * (v / n)

        acc_norm = np.linalg.norm(acc)
        if acc_norm <= 1e-6:
            raise ValueError("index-finger direction degenerate")

        return tip, (acc / acc_norm)

    def on_tracking_event(self, event):
        if self.show_viz:
            self.canvas.render_hands(event)

        num_hands = len(event.hands)

        if num_hands == 2:
            now = time.time()
            if self._two_hand_start is None:
                self._two_hand_start = now
                self._home_triggered = False
            elif not self._home_triggered:
                held_for = now - self._two_hand_start
                if held_for >= self.TWO_HAND_HOME_SEC:
                    self.home_pub.publish(Bool(data=True))
                    self._home_triggered = True
                    rospy.loginfo(
                        "Two hands held for %.1fs — publishing /leap/home", held_for
                    )
                    if self.show_viz:
                        self.canvas.stability_text = "HOMING ✓"
                else:
                    if self.show_viz:
                        self.canvas.stability_text = (
                            f"Homing: {held_for:.1f}/{self.TWO_HAND_HOME_SEC:.0f}s"
                        )
            else:
                if self.show_viz:
                    self.canvas.stability_text = "HOMING (holding)"
        else:
            self._two_hand_start = None
            self._home_triggered = False

        if num_hands == 1:
            hand = event.hands[0]

            try:
                tip, dir_dev = self._index_tip_and_direction(hand)

                tip_world = LEAP_TILT_R @ tip
                dir_world = LEAP_TILT_R @ dir_dev
                self.current_point_tip = tip_world.tolist()
                self.current_point_dir = dir_world.tolist()
                self.current_target_m = self._compute_target_point(
                    self.current_point_tip, self.current_point_dir
                )

                stability_status = self._check_stability(
                    self.current_point_dir, self.current_target_m
                )

                if self.show_viz:
                    self.canvas.current_direction = _pointing_label(self.current_point_dir)
                    self.canvas.palm_direction_text = (
                        f"Tip : [{self.current_point_tip[0]:+6.1f}, "
                        f"{self.current_point_tip[1]:+6.1f}, "
                        f"{self.current_point_tip[2]:+6.1f}] mm  "
                        f"Dir: [{self.current_point_dir[0]:+.2f}, "
                        f"{self.current_point_dir[1]:+.2f}, "
                        f"{self.current_point_dir[2]:+.2f}]"
                    )
                    tgt = self.current_target_m
                    dist = float(np.linalg.norm(tgt))
                    self.canvas.target_text = (
                        f"Target: [{tgt[0]:+.3f}, {tgt[1]:+.3f}, {tgt[2]:+.3f}] m"
                        f"  |d|={dist:.3f}m"
                    )
                    self.canvas.stability_text = stability_status

                self.pub.publish(json.dumps({
                    "tip_pos":      self.current_point_tip,
                    "direction":    self.current_point_dir,
                    "target_pos_m": self.current_target_m,
                }))
            except Exception as e:
                rospy.logwarn_throttle(5.0, "Pointing detect error: %s", e)

        if num_hands != 1:
            self.current_point_tip = None
            self.current_point_dir = None
            self.current_target_m  = None
            self._dir_history.clear()
            self._stable_since = None
            if self.show_viz and num_hands == 0:
                self.canvas.current_direction = "No hands detected"
                self.canvas.target_text = ""
                self.canvas.stability_text = ""

def main():
    rospy.init_node("detect_point")
    show_viz = rospy.get_param("~show_visualization", True)

    detector = GestureDetectorROS(show_viz=show_viz)

    connection = leap.Connection()
    connection.add_listener(detector)

    rospy.loginfo("Opening Leap Motion connection ...")
    with connection.open():
        connection.set_tracking_mode(TrackingMode.Desktop)
        if show_viz:
            detector.canvas.tracking_mode = TrackingMode.Desktop
        rospy.loginfo("Leap Motion connected - detecting gestures")

        if show_viz:
            rate = rospy.Rate(60)
            while not rospy.is_shutdown():
                cv2.imshow(detector.canvas.name, detector.canvas.output_image)
                if cv2.waitKey(1) == ord('x'):
                    break
                rate.sleep()
            cv2.destroyAllWindows()
        else:
            rospy.spin()

    rospy.loginfo("Gesture detector shut down")


if __name__ == "__main__":
    main()
