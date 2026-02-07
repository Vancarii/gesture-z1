import leap
import time
import math
import numpy as np
import cv2
from architecture.preprocessing import extract_features, twohand_extract_feature
import joblib 
from collections import deque, Counter
import os
from architecture.model_cnn import Net
import torch
import torch.nn as nn
from middleware import server
from policy import Brain
import threading
import asyncio
import sys

#Local or remote mode
brain = Brain(train=False)
MODE = str(sys.argv[1])
if MODE not in {"local", "remote"}:
    print("Invalid mode")
    exit(1)


def start_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server.loop = loop
    loop.create_task(server.start())
    loop.run_forever()


threading.Thread(target=start_server, daemon=True).start()


def emit_remote(message: dict) -> None:
    loop = getattr(server, "loop", None)
    if loop is None or not loop.is_running():
        print("[Bridge]: Data server loop not ready; dropping message")
        return

    try:
        asyncio.run_coroutine_threadsafe(server.broadcast(message), loop)
    except Exception as exc:
        print(f"[Bridge]: Failed to broadcast message: {exc}")

#Loads CNN model
def load_models(model_path = "architecture/models"):
    try:
        classes = np.load(os.path.join(model_path, "classes.npy"))
        model = Net(len(classes))
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model.load_state_dict(torch.load(os.path.join(model_path, "gesture_model_cnn.pth")))
        else:
            device = torch.device("cpu")
            model.load_state_dict(torch.load(os.path.join(model_path, "gesture_model_cnn.pth"), map_location=device))
            print("WARNING: Running on CPU (slower)")
        
        model.to(device)
        model.eval()
        print("Models loaded successfully")

        scaler = joblib.load(os.path.join(model_path, "scalerCNN.pkl"))
        return model, scaler, classes, device

    except Exception as e:
        print(f"error loading models: {e}")
        exit(1)

#Temporal smoothening for prediction
class TemporalSmoothening:
    def __init__(self, buffersize = 15):
        self.buffer = deque(maxlen = buffersize)
        self.lastpredict = 'NAN'
    
    def smooth(self, prediction):
        self.buffer.append(prediction)
        likely = Counter(self.buffer).most_common(1)[0]
        if likely[1] > len(self.buffer) // 2:
            self.lastpredict = likely[0]
        return self.lastpredict


#Loads model
model, scaler, classes, device = load_models()

#Useless classes, was used in older version
def normalize(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

def pointing(pos, dir):
    p = np.array(pos)
    d = normalize(np.array(dir))
    return lambda t: p + t * d

def intersection(p, d, point, normal):
    p = np.array(p)
    d = normalize(np.array(d))
    n = np.array(normal)
    d_dot_n = np.dot(d, n)
    if d_dot_n == 0:
        return None
    t = np.dot(n, point - p) / d_dot_n
    return p + t * d

#Importing TrackingMode
try:
    TrackingMode = leap.TrackingMode
except AttributeError:
    try:
        from leap.enums import TrackingMode as _TrackingMode
    except (ImportError, AttributeError) as exc:  
        raise ImportError(
            "TrackingMode enum is unavailable from the loaded leap module. "
            "Install the repo's leapc-python-api package (pip install -e leapc-python-api) "
            "and ensure the correct LEAPSDK path is configured."
        ) from exc
    else:
        TrackingMode = _TrackingMode
        setattr(leap, "TrackingMode", TrackingMode)

_TRACKING_MODES = {
    TrackingMode.Desktop: "Desktop",
    TrackingMode.HMD: "HMD",
    TrackingMode.ScreenTop: "ScreenTop",
}

class DirectionVector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

#UI to display hand and prediction
class Canvas:
    def __init__(self):
        self.name = "Gesture Detector"
        self.screen_size = [500, 700]
        self.hands_colour = (255, 255, 255)
        self.font_colour = (0, 255, 44)
        self.hands_format = "Skeleton"
        self.output_image = np.zeros((self.screen_size[0], self.screen_size[1], 3), np.uint8)
        self.tracking_mode = None
        self.current_direction = "No hands detected"
        self.palm_direction_text = ""
        self.predict = "NAN"

    def set_tracking_mode(self, tracking_mode):
        self.tracking_mode = tracking_mode

    def get_joint_position(self, bone):
        if bone:
            SCALE = 1.0 
            return int(bone.x * SCALE + (self.screen_size[1] / 2)), int((self.screen_size[0] / 2) - (bone.z * SCALE))
        else:
            return None

    def render_hands(self, event):
        self.output_image[:, :] = 0

        cv2.putText(
            self.output_image,
            f"Predict: {self.predict}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            self.font_colour,
            3,
        )

        cv2.putText(
            self.output_image,
            f"Direction: {self.current_direction}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.font_colour,
            2,
        )

        cv2.putText(
            self.output_image,
            self.palm_direction_text,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.font_colour,
            1,
        )

        cv2.putText(
            self.output_image,
            f"Tracking Mode: {_TRACKING_MODES.get(self.tracking_mode, 'Unknown')}",
            (10, self.screen_size[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.font_colour,
            1,
        )

        if len(event.hands) == 0:
            return

        for i in range(0, len(event.hands)):
            hand = event.hands[i]
            
            palm_pos = self.get_joint_position(hand.palm.position)
            if palm_pos:
                cv2.circle(self.output_image, palm_pos, 8, self.hands_colour, -1)
                cv2.putText(self.output_image, "PALM", 
                           (palm_pos[0] + 10, palm_pos[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.hands_colour, 1)
                
                palm_dir = hand.palm.direction
                arrow_end_x = int(palm_pos[0] + palm_dir.x * 50)
                arrow_end_y = int(palm_pos[1] - palm_dir.z * 50) 
                cv2.arrowedLine(self.output_image, palm_pos, (arrow_end_x, arrow_end_y), (0, 255, 0), 3)
            
            wrist = self.get_joint_position(hand.arm.next_joint)
            elbow = self.get_joint_position(hand.arm.prev_joint)
            if wrist and elbow:
                cv2.circle(self.output_image, wrist, 3, self.hands_colour, -1)
                cv2.circle(self.output_image, elbow, 3, self.hands_colour, -1)
                if wrist and elbow:
                    cv2.line(self.output_image, wrist, elbow, self.hands_colour, 2)

            for index_digit in range(0, 5):
                digit = hand.digits[index_digit]
                for index_bone in range(0, 4):
                    bone = digit.bones[index_bone]
                    
                    bone_start = self.get_joint_position(bone.prev_joint)
                    bone_end = self.get_joint_position(bone.next_joint)

                    if bone_start:
                        cv2.circle(self.output_image, bone_start, 3, self.hands_colour, -1)

                    if bone_end:
                        cv2.circle(self.output_image, bone_end, 3, self.hands_colour, -1)

                    if bone_start and bone_end:
                        cv2.line(self.output_image, bone_start, bone_end, self.hands_colour, 2)

                    if index_digit == 1 and index_bone == 3:  
                        if bone_end:
                            cv2.circle(self.output_image, bone_end, 6, (0, 255, 0), -1)
                            cv2.putText(self.output_image, "INDEX", 
                                       (bone_end[0] + 10, bone_end[1]), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                    if index_bone == 0 and bone_start and wrist:
                        cv2.line(self.output_image, bone_start, wrist, self.hands_colour, 2)

#Old outdated direction detector
class DirectionDetector(leap.Listener):
    def __init__(self, canvas, min=50, max=2000):
        self.canvas = canvas
        self.prev_dir = None
        self.prev_hand = None
        self.dt = None
        self.min = min
        self.max = max
        self.alpha = 0.5
        self.sequence = []
        self.curr_label = None
    
    def vector_length(self, v):
        v = np.array(v, dtype=float)
        n = np.linalg.norm(v)
        return (v/n) if n > 1e-8 else None
    
    def adapt_alpha(self, angle):
        a_upper, a_lower = 0.9, 0.1
        roundoff = 30
        t = min(angle / roundoff, 1)
        return a_upper * (1-t) + a_lower * t
        
    def calc_angle(self, x, y):
        dot = np.clip(np.dot(x, y), -1, 1)
        return math.degrees(math.acos(dot))
    
    def save_gesture(self, label):
        np.save(f"data/{label}_{int(time.time())}.npy", np.array(self.sequence))
        self.sequence.clear()

    
    def on_tracking_event(self, event):

        self.canvas.render_hands(event) 

        if len(event.hands) == 0:
            self.canvas.current_direction = "No hands detected"
            return
        
        hand = event.hands[0] 

        index = hand.digits[1]

        palm_pos = np.array([
            hand.palm.position.x,
            hand.palm.position.y,
            hand.palm.position.z
        ])

        tip = np.array([
            index.bones[3].next_joint.x,
            index.bones[3].next_joint.y,
            index.bones[3].next_joint.z
        ])
        
        feature = extract_features(hand, self.prev_hand, self.prev_hand2)

        if len(self.sequence) > 0:
            feature = self.alpha * feature + (1-self.alpha) * self.sequence[-1]
        
        self.sequence.append(feature)
        self.prev_hand = hand

        # Limits buffer to 2 seconds
        if(len(self.sequence) > 60):
            self.sequence.pop(0)
        
        if len(self.sequence) % 30 == 0:
            print(f"Feature snapshot: {self.sequence[-1]}")
        

        base = np.array([
            index.bones[0].prev_joint.x,
            index.bones[0].prev_joint.y,
            index.bones[0].prev_joint.z
        ])

        virtual_plane = -100.0
        point_dir = normalize(tip - base)

        if self.prev_dir is not None:
            smooth_dir = normalize(self.alpha * self.prev_dir + (1 - self.alpha) * point_dir)
        else:
            smooth_dir = point_dir
        self.prev_dir = smooth_dir
        plane_point = np.array([0.0, 0.0, virtual_plane])
        plane_normal = np.array([0.0, 0.0, 1.0])
        
        target_intersection = intersection(palm_pos, point_dir, plane_point, plane_normal)

        SCALE = 1.0 
        screen_w = self.canvas.screen_size[1]
        screen_h = self.canvas.screen_size[0]
        
        palm2d = (int(palm_pos[0] * SCALE + screen_w / 2), 
                  int(screen_h / 2 - palm_pos[2] * SCALE))
        
        arrow_length = 150 
        endpointer = (
            int(palm2d[0] + point_dir[0] * arrow_length),
            int(palm2d[1] - point_dir[2] * arrow_length)
        )
        
        cv2.arrowedLine(self.canvas.output_image, palm2d, endpointer, (255, 0, 255), 3)

        if target_intersection is not None:
            inter2d = (int(target_intersection[0] * SCALE + screen_w / 2),
                       int(screen_h / 2 - target_intersection[2] * SCALE))

            cv2.circle(self.canvas.output_image, inter2d, 8, (0,0,255), -1)
            cv2.putText(self.canvas.output_image, "Target", (inter2d[0] + 5, inter2d[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        self.canvas.current_direction = self.calculate_direction(DirectionVector(point_dir[0], point_dir[1], point_dir[2]))
        self.canvas.palm_direction_text = f"Index Direction (Norm): ({point_dir[0]:.2f}, {point_dir[1]:.2f}, {point_dir[2]:.2f})"

    
    def calculate_direction(self, palm_direction):

        x = palm_direction.x
        y = palm_direction.y
        z = palm_direction.z
        
        threshold = 0.3
        
        abs_x = abs(x)
        abs_y = abs(y)
        abs_z = abs(z)
        
        if abs_x < threshold and abs_y < threshold and abs_z < threshold:
            return "Center"
        
        max_component = max(abs_x, abs_y, abs_z)
        
        if max_component == abs_x:
            return "Right" if x > 0 else "Left"
        elif max_component == abs_y:
            return "Up" if y > 0 else "Down"
        else: 
            return "Forward" if z < 0 else "Backward"

#Relevant class to work on
class GestureDetector(leap.Listener):
    def __init__(self, canvas):
        super().__init__()
        self.canvas = canvas
        self.prevhand = None
        self.prevhand2 = None
        self.featurebuffer = deque(maxlen = 90)
        self.smoothening = TemporalSmoothening(buffersize = 15)
        self.eval_interval = 0.1
        self.last_eval = 0.0
        self.threshold = 0.85

        #Stable frames, hertz, delta for motion detection
        self.stableframes = 3
        self.hertz = 0.10
        self.delta = 0.01

        #History for stable frames
        self.history = deque(maxlen = self.stableframes)
        self.last_time = 0
        self.last_gesture = None
        self.last_velocity = None
        self.last_targetpos = None
    
    #Tracks hands live and does feature extraction
    def on_tracking_event(self, event):
        self.canvas.render_hands(event)

        # Feature extraction
        if len(event.hands) == 1:
            hand = event.hands[0]
            feature = extract_features(hand, self.prevhand, self.prevhand2)
            self.featurebuffer.append(feature)
            self.prevhand2 = self.prevhand
            self.prevhand = hand
            self.predict()

        #Update for clapping in future
        elif len(event.hands) == 2:
            self.prevhand2 = None
            self.prevhand = None
            self.featurebuffer.clear()
            self.canvas.predict = self.smoothening.smooth("NAN")

        else:
            self.prevhand = None
            self.prevhand2 = None
            self.featurebuffer.clear()
            self.canvas.predict = self.smoothening.smooth("NAN")
    
    #Predicts gesture
    def predict(self):

        #If history is not initialized, initialize it
        if not hasattr(self, "history"):
            self.history = deque(maxlen = self.stableframes)
            self.last_time = 0
            self.last_gesture = None
            self.last_velocity = None
            self.last_targetpos = None

        currtime = time.perf_counter()
        if (len(self.featurebuffer) == 90 and currtime - self.last_eval >= self.eval_interval):
            self.last_eval = currtime

            try:
                sequence = np.array(self.featurebuffer)
                scaled_sequence = scaler.transform(sequence)
                tensor = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(tensor)
                    pr = torch.nn.functional.softmax(output, dim=1)
                    confidenceobj, predictionindex = torch.max(pr, 1)
                    confidence = confidenceobj.item()
                    prediction = classes[predictionindex.item()]
                
                finalprediction = 'NAN'
                if confidence > self.threshold:
                    finalprediction = prediction

                    self.history.append((finalprediction, confidence))
                    
                    if len(self.history) == self.stableframes:
                        stable = all(
                            label == finalprediction and conf >= self.threshold
                            for label, conf in self.history
                        )
                    else:
                        stable = False
                    
                    if stable:
                        action = brain.action(finalprediction)
                        if action is not None:
                            message = {
                                "gesture": str(finalprediction),
                                "confidence": float(confidence),
                                "cmdid": int(action["cmdid"]),
                                "velocity": action["velocity"].astype(float).tolist(),
                                "targetpos": action["targetpos"].astype(float).tolist(),
                                "grippercmd": int(action["grippercmd"]),
                            }
                            
                            curr = time.time()
                            if curr - self.last_time >= self.hertz:
                                motion = False

                                if self.last_velocity is None or self.last_targetpos is None:
                                    motion = True
                                else:
                                    dv = np.linalg.norm(action["velocity"] - self.last_velocity)
                                    tp = np.linalg.norm(action["targetpos"] - self.last_targetpos)
                                    if dv > self.delta or tp > self.delta:
                                        motion = True

                                if finalprediction != self.last_gesture:
                                    motion = True

                                if motion:
                                    if MODE == "remote":
                                        emit_remote(message)
                                    else:
                                        print(f"[LOCAL] Predicted gesture: {finalprediction} with confidence: {confidence}")
                                        print(f"[Local] action: {message}")
                                    self.last_time = curr
                                    self.last_gesture = finalprediction
                                    self.last_velocity = action["velocity"].copy()
                                    self.last_targetpos = action["targetpos"].copy()

                self.canvas.predict = self.smoothening.smooth(finalprediction)
            
            except Exception as e:
                print(f"Error predicting gesture: {e}")
    

def run_pointing():
    canvas = Canvas()
    detector = GestureDetector(canvas)
    connection = leap.Connection()
    connection.add_listener(detector)
    running = True

    with connection.open():
        connection.set_tracking_mode(TrackingMode.Desktop)
        canvas.set_tracking_mode(TrackingMode.Desktop)

        while running:
            key = cv2.waitKey(1)
            cv2.imshow(canvas.name, canvas.output_image)

            if key == ord('x'):
                break

    cv2.destroyAllWindows()

#Dynamic gestures (4 or 5, detect differentiate)

if __name__ == "__main__":
    run_pointing()