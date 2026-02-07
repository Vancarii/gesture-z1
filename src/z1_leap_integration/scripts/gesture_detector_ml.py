#!/home/tangentlab/anaconda3/envs/gesture-env/bin/python
"""
ROS node: gesture_detector_ml

Detects hand gestures via Leap Motion + CNN classifier, maps them to
robot actions, and publishes action commands as JSON on the /leap/gesture
topic.

This is the ROS-integrated version of Z1-LeapC/src/pointing.py.
It reuses the same CNN model and scaler.  The gesture-to-velocity
mapping is a lightweight table that matches the converged PPO policy
from policy.py, so stable_baselines3/gymnasium are NOT needed.

Requires the 'gesture-env' conda environment (Leap SDK, PyTorch, etc.).
ROS (rospy) is picked up via PYTHONPATH set by roslaunch.
"""

import sys
import os
import ctypes

# ---------------------------------------------------------------------------
# GLIBCXX fix: conda's scipy/sklearn/matplotlib need GLIBCXX_3.4.29 but
# the system libstdc++ only goes to 3.4.28.  Preload conda's own copy.
# ---------------------------------------------------------------------------
_CONDA_LIBSTDCPP = os.path.join(
    sys.prefix, "lib", "libstdc++.so.6"
)
if os.path.isfile(_CONDA_LIBSTDCPP):
    ctypes.CDLL(_CONDA_LIBSTDCPP, mode=ctypes.RTLD_GLOBAL)

import time
import json
import numpy as np
from collections import deque, Counter

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
Z1_LEAPC_ROOT = "/home/tangentlab/z1_ws/src/Z1-LeapC"
Z1_LEAPC_SRC  = os.path.join(Z1_LEAPC_ROOT, "src")
sys.path.insert(0, Z1_LEAPC_SRC)

# Guarantee rospy is importable even when running under conda python.
# rospkg lives in /usr/lib/python3/dist-packages (system), rospy in /opt/ros/...
for _p in [
    "/usr/lib/python3/dist-packages",
    "/opt/ros/noetic/lib/python3/dist-packages",
    "/home/tangentlab/z1_ws/devel/lib/python3/dist-packages",
]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import rospy
from std_msgs.msg import String

import leap
import torch
import joblib
import cv2

from architecture.preprocessing import extract_features
from architecture.model_cnn import Net


# ===================================================================
# Lightweight gesture -> action map  (replaces Brain + PPO)
#
# The PPO in policy.py is trained against a reward function whose
# optimal policy is a deterministic velocity direction per gesture.
# We hard-code that converged mapping here so we don't need
# stable_baselines3 / gymnasium / matplotlib at runtime.
# ===================================================================
_GESTURE_ALIASES = {
    "point forward": "swipe towards",
    "point back":    "swipe back",
    "move hand up":  "point up",
    "move hand down": "point down",
    "pause":         "background",
}

# velocity magnitude used by the PPO-converged policy
_V = 0.3

_ACTION_TABLE = {
    #  gesture (canonical)   cmdid    velocity (brain-space)       gripper
    "swipe back":            {"cmdid": 0, "velocity": np.array([0.0, -_V, 0.0]), "gripper": 0.0},
    "swipe towards":         {"cmdid": 1, "velocity": np.array([0.0,  _V, 0.0]), "gripper": 0.0},
    "point up":              {"cmdid": 2, "velocity": np.array([0.0, 0.0,  _V]), "gripper": 0.0},
    "point down":            {"cmdid": 3, "velocity": np.array([0.0, 0.0, -_V]), "gripper": 0.0},
    "pull back":             {"cmdid": 4, "velocity": np.array([0.0, 0.0, 0.0]), "gripper": 1.0},
    "background":            {"cmdid": 5, "velocity": np.array([0.0, 0.0, 0.0]), "gripper": 0.0},
    "point left":            {"cmdid": 6, "velocity": np.array([-0.2, 0.0, 0.0]), "gripper": 0.0},
    "point right":           {"cmdid": 7, "velocity": np.array([ 0.2, 0.0, 0.0]), "gripper": 0.0},
}


class GestureActionMap:
    """Drop-in replacement for Brain.action() — no PPO needed."""

    def __init__(self):
        self._pos = np.array([0.5, 0.0, 0.5])   # virtual position tracker

    def action(self, gesture):
        canonical = _GESTURE_ALIASES.get(gesture, gesture)
        spec = _ACTION_TABLE.get(canonical)
        if spec is None:
            return None
        vel = spec["velocity"]
        targetpos = self._pos + vel * 0.5
        self._pos = targetpos.copy()
        return {
            "cmdid":      spec["cmdid"],
            "velocity":   vel,
            "targetpos":  targetpos,
            "grippercmd": spec["gripper"],
        }


_brain = GestureActionMap()

# ---------------------------------------------------------------------------
# TrackingMode import (handles different leapc-python-api layouts)
# ---------------------------------------------------------------------------
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


# ===================================================================
# Helper: load CNN model + scaler + class labels
# ===================================================================
def load_models(model_path=None):
    if model_path is None:
        model_path = os.path.join(Z1_LEAPC_SRC, "architecture", "models")

    classes = np.load(os.path.join(model_path, "classes.npy"))
    model   = Net(len(classes))
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(
        torch.load(
            os.path.join(model_path, "gesture_model_cnn.pth"),
            map_location=device,
        )
    )
    model.to(device)
    model.eval()

    # Load the TRAINED scaler (not a fresh one!)
    scaler = joblib.load(os.path.join(model_path, "scalerCNN.pkl"))

    rospy.loginfo("Models loaded: %d classes, device=%s", len(classes), device)
    return model, scaler, classes, device


# ===================================================================
# Temporal smoothing (majority-vote over sliding window)
# ===================================================================
class TemporalSmoothening:
    def __init__(self, buffersize=15):
        self.buffer = deque(maxlen=buffersize)
        self.lastpredict = "NAN"

    def smooth(self, prediction):
        self.buffer.append(prediction)
        likely = Counter(self.buffer).most_common(1)[0]
        if likely[1] > len(self.buffer) // 2:
            self.lastpredict = likely[0]
        return self.lastpredict


# ===================================================================
# CV2 hand visualisation (mirrors pointing.py Canvas)
# ===================================================================
class Canvas:
    """CV2 hand skeleton overlay — matches pointing.py rendering."""

    def __init__(self):
        self.name = "Gesture Detector (ROS)"
        self.screen_size = [500, 700]
        self.hands_colour = (255, 255, 255)
        self.font_colour  = (0, 255, 44)
        self.hands_format = "Skeleton"
        self.output_image = np.zeros(
            (self.screen_size[0], self.screen_size[1], 3), np.uint8
        )
        self.predict = "NAN"
        self.tracking_mode = None
        self.current_direction = "No hands detected"
        self.palm_direction_text = ""

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

        # --- Prediction / gesture label ---
        cv2.putText(
            self.output_image, f"Predict: {self.predict}",
            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.font_colour, 3,
        )

        # --- Direction text ---
        cv2.putText(
            self.output_image, f"Direction: {self.current_direction}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.font_colour, 2,
        )

        # --- Palm direction detail ---
        cv2.putText(
            self.output_image, self.palm_direction_text,
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.font_colour, 1,
        )

        # --- Tracking mode ---
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

            # --- Palm ---
            palm_pos = self.get_joint_position(hand.palm.position)
            if palm_pos:
                cv2.circle(self.output_image, palm_pos, 8, self.hands_colour, -1)
                cv2.putText(
                    self.output_image, "PALM",
                    (palm_pos[0] + 10, palm_pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.hands_colour, 1,
                )

                # Palm direction arrow
                palm_dir = hand.palm.direction
                arrow_end_x = int(palm_pos[0] + palm_dir.x * 50)
                arrow_end_y = int(palm_pos[1] - palm_dir.z * 50)
                cv2.arrowedLine(
                    self.output_image, palm_pos,
                    (arrow_end_x, arrow_end_y), (0, 255, 0), 3,
                )

            # --- Wrist / Elbow ---
            wrist = self.get_joint_position(hand.arm.next_joint)
            elbow = self.get_joint_position(hand.arm.prev_joint)
            if wrist:
                cv2.circle(self.output_image, wrist, 3, self.hands_colour, -1)
            if elbow:
                cv2.circle(self.output_image, elbow, 3, self.hands_colour, -1)
            if wrist and elbow:
                cv2.line(self.output_image, wrist, elbow, self.hands_colour, 2)

            # --- Fingers ---
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

                    # Highlight index fingertip
                    if index_digit == 1 and index_bone == 3:
                        if bone_end:
                            cv2.circle(self.output_image, bone_end, 6, (0, 255, 0), -1)
                            cv2.putText(
                                self.output_image, "INDEX",
                                (bone_end[0] + 10, bone_end[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
                            )

                    # Connect metacarpal base to wrist
                    if index_bone == 0 and bone_start and wrist:
                        cv2.line(self.output_image, bone_start, wrist, self.hands_colour, 2)


# ===================================================================
# Main ROS gesture detector (Leap listener)
# ===================================================================
class GestureDetectorROS(leap.Listener):
    """
    Leap Motion listener that:
      1. Extracts 19-dim features per frame
      2. Classifies gesture with CNN after 90 frames
      3. Runs Brain policy to get velocity / target-pos
      4. Publishes JSON action message to /leap/gesture
      5. (Optionally) renders CV2 hand visualisation
    """

    def __init__(self, model, scaler, classes, device, brain,
                 show_viz=True):
        super().__init__()
        self.model   = model
        self.scaler  = scaler
        self.classes = classes
        self.device  = device
        self.brain   = brain

        # ROS publisher
        self.pub = rospy.Publisher("/leap/gesture", String, queue_size=10)

        # Feature buffer (90 frames ~ 1.5 s at 60 Hz)
        self.prevhand      = None
        self.prevhand2     = None
        self.featurebuffer = deque(maxlen=90)
        self.smoothening   = TemporalSmoothening(buffersize=15)

        # Timing & stability
        self.eval_interval = 0.1
        self.last_eval     = 0.0
        self.threshold     = 0.85
        self.stableframes  = 3
        self.publish_hz    = 0.10
        self.delta         = 0.01

        self.history           = deque(maxlen=self.stableframes)
        self.last_publish_time = 0
        self.last_gesture      = None
        self.last_velocity     = None
        self.last_targetpos    = None

        # Visualisation
        self.show_viz = show_viz
        if self.show_viz:
            self.canvas = Canvas()

        rospy.loginfo("GestureDetectorROS initialised")

    # ---- Leap callback -----------------------------------------------
    def on_tracking_event(self, event):
        if self.show_viz:
            self.canvas.render_hands(event)

        if len(event.hands) == 1:
            hand = event.hands[0]
            feat = extract_features(hand, self.prevhand, self.prevhand2)
            self.featurebuffer.append(feat)
            self.prevhand2 = self.prevhand
            self.prevhand  = hand
            self._predict()
        else:
            self.prevhand  = None
            self.prevhand2 = None
            self.featurebuffer.clear()
            if self.show_viz:
                self.canvas.predict = self.smoothening.smooth("NAN")
                self.canvas.current_direction = "No hands detected"

    # ---- CNN inference -----------------------------------------------
    def _predict(self):
        now = time.perf_counter()
        if len(self.featurebuffer) < 90:
            return
        if now - self.last_eval < self.eval_interval:
            return
        self.last_eval = now

        try:
            sequence = np.array(self.featurebuffer)
            scaled   = self.scaler.transform(sequence)
            tensor   = (torch.tensor(scaled, dtype=torch.float32)
                             .unsqueeze(0).to(self.device))

            with torch.no_grad():
                output = self.model(tensor)
                probs  = torch.softmax(output, dim=1)
                conf_val, pred_idx = torch.max(probs, 1)
                confidence = conf_val.item()
                prediction = self.classes[pred_idx.item()]

            if confidence < self.threshold:
                if self.show_viz:
                    self.canvas.predict = self.smoothening.smooth("NAN")
                return

            # Temporal stability check
            self.history.append((prediction, confidence))
            stable = (
                len(self.history) == self.stableframes
                and all(
                    g == prediction and c >= self.threshold
                    for g, c in self.history
                )
            )

            if self.show_viz:
                self.canvas.predict = self.smoothening.smooth(
                    prediction if stable else "NAN"
                )
                self.canvas.current_direction = prediction if stable else ""

            if stable:
                self._publish_action(prediction, confidence)

        except Exception as e:
            rospy.logwarn("Prediction error: %s", e)

    # ---- Brain -> ROS publish ----------------------------------------
    def _publish_action(self, gesture, confidence):
        action = self.brain.action(gesture)
        if action is None:
            return

        curr = time.time()
        if curr - self.last_publish_time < self.publish_hz:
            return

        # Avoid flooding identical commands
        motion = False
        if self.last_velocity is None or self.last_targetpos is None:
            motion = True
        else:
            dv = np.linalg.norm(action["velocity"] - self.last_velocity)
            tp = np.linalg.norm(action["targetpos"] - self.last_targetpos)
            if dv > self.delta or tp > self.delta:
                motion = True
        if gesture != self.last_gesture:
            motion = True
        if not motion:
            return

        msg = {
            "gesture":    str(gesture),
            "confidence": float(confidence),
            "cmdid":      int(action["cmdid"]),
            "velocity":   [float(v) for v in action["velocity"]],
            "targetpos":  [float(v) for v in action["targetpos"]],
            "grippercmd": float(action["grippercmd"]),
        }
        self.pub.publish(json.dumps(msg))
        rospy.loginfo("Published: %s  conf=%.2f  vel=%s",
                      gesture, confidence, msg["velocity"])

        self.last_publish_time = curr
        self.last_gesture   = gesture
        self.last_velocity  = action["velocity"].copy()
        self.last_targetpos = action["targetpos"].copy()


# ===================================================================
# Entry point
# ===================================================================
def main():
    rospy.init_node("gesture_detector_ml")
    show_viz = rospy.get_param("~show_visualization", True)

    model, scaler, classes, device = load_models()

    detector = GestureDetectorROS(
        model, scaler, classes, device, _brain,
        show_viz=show_viz,
    )

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
