import leap
import os
import cv2
import time
import numpy as np
from architecture.preprocessing import extract_features
from architecture.preprocessing import twohand_extract_feature
from sklearn.preprocessing import StandardScaler

GESTURE_SET = [
    "swipe back", "pull back", "background"
]

GESTURE_LEFT = [
    "point up", "point down", "point forward", "point back",
    "point left", "point right",
    "move hand up", "move hand down",
    "pause", "clap",
    "swipe towards", 
]

twohand_gestures = ["clap"]

class addlistener(leap.Listener):
    def __init__(self, gesture_dir, label, sample_num):
        super().__init__()
        self.label = label
        self.twohand = label in twohand_gestures
        self.buffer = []
        self.isrecording = False
        self.frames = 0
        self.gesture_dir = gesture_dir
        self.sample_num = sample_num
        self.currhand = []
        self.prevhand = None
        self.prevhand2 = None
    
    def startrecording(self):
        self.buffer = []
        self.frames = 0
        self.prevhand = None
        self.prevhand2 = None
        self.isrecording = True
        print(f"Starting recording for '{self.label}'")

    def on_tracking_event(self, event):
        self.currhand = event.hands
        if not self.isrecording:
            return
        
        if self.frames >= 90:
            self.stoprecording()
            return
        
        handobj = None
        
        if self.twohand:
            if len(event.hands) != 2:
                print("Two hand gesture requires two hands")
                return
            
            hand1, hand2 = event.hands[0], event.hands[1]
            feature = twohand_extract_feature(hand1, hand2)
        
        else:
            if len(event.hands) != 1:
                print("One hand gesture requires one hand")
                self.prevhand = None
                self.prevhand2 = None
                return
                
            hand = event.hands[0]
            handobj = hand
            feature = extract_features(hand, self.prevhand, self.prevhand2)
        
        if not self.twohand and handobj is not None:
            self.prevhand2 = self.prevhand
            self.prevhand = handobj
        
        if np.any(feature): #Non-zero feature
            self.buffer.append(feature)
            self.frames += 1
        else:
            print("Skipping frame with zero feature")
    
    def stoprecording(self):
        self.isrecording = False
        
        if len(self.buffer) < 90:
            print(f"Buffer too small to save, <90 frames")
            return False
        
        self.buffer = self.buffer[:90] #Limit to 90 frames
        
        filename = f"{self.label.replace(' ', '_')}_{self.sample_num:03d}.npy"
        np.save(os.path.join(self.gesture_dir, filename), np.array(self.buffer))
        print(f"Saved gesture '{self.label}' ({len(self.buffer)} frames => {filename})")
        return True

    def get_hand(self):
        return self.currhand
    
def visualize_hand(canvas, hands):
    scale = 1.0
    center_x = canvas.shape[1] // 2
    center_y = canvas.shape[0] // 2 + 250

    hand_color = [(255, 0, 0), (0, 0, 255)]

    for i, hand in enumerate(hands):
        color = hand_color[i % len(hand_color)]
        palm_pos = hand.palm.position

        cx_palm = int(palm_pos.x * scale + center_x)
        cy_palm = int(-palm_pos.y * scale + center_y)

        cv2.circle(canvas, (cx_palm, cy_palm), 8, color, -1)

        for digit in hand.digits:
            for boneindex in range(4):
                bone = digit.bones[boneindex]
                nextjoint = bone.next_joint
                prevjoint = bone.prev_joint

                cx_prev = int(prevjoint.x * scale + center_x)
                cy_prev = int(-prevjoint.y * scale + center_y)

                cx_next = int(nextjoint.x * scale + center_x)
                cy_next = int(-nextjoint.y * scale + center_y)

                if boneindex == 3:
                    cv2.circle(canvas, (cx_next, cy_next), 6, color, -1)
                
                cv2.line(canvas, (cx_prev, cy_prev), (cx_next, cy_next), color, 2)
    return canvas

def collect_samples(data_dir = "data"):
    os.makedirs(data_dir, exist_ok=True)

    cv2.namedWindow("Preprocessing")
    connection = leap.Connection()

    with connection.open():
        connection.set_tracking_mode(leap.TrackingMode.Desktop)

        for gesture in GESTURE_SET:
            os.makedirs(os.path.join(data_dir, gesture.replace(" ", "_")), exist_ok=True)
            sample = len([f for f in os.listdir(os.path.join(data_dir, gesture.replace(" ", "_"))) if f.endswith(".npy")])
            print(f"Preparing to record gesture: '{gesture}' ({sample} existing samples)")

            while sample < 50:
                listener = addlistener(os.path.join(data_dir, gesture.replace(" ", "_")), gesture, sample)
                connection.add_listener(listener)

                while True:
                    img = np.zeros((1000, 1400, 3), dtype = np.uint8)
                    cv2.putText(img, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
                    cv2.putText(img, f"Sample: {sample+1}/50", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
                    
                    visualizer_2d = img[:, 450:]
                    hands = listener.get_hand()
                    if hands:
                        visualize_hand(visualizer_2d, hands)
                    
                    cv2.imshow("Preprocessing", img)
                    key = cv2.waitKey(10)

                    if key == ord('s'):
                        break
                
                for i in range(3, 0, -1):
                    print(f"Ready? {i}")
                    
                    img = np.zeros((1000, 1400, 3), dtype = np.uint8)
                    cv2.putText(img, f"Starting in {i}...", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)
                    
                    visualizer_2d = img[:, 450:]
                    hands = listener.get_hand()
                    if hands:
                        visualize_hand(visualizer_2d, hands)
                    
                    cv2.imshow("Preprocessing", img)
                    cv2.waitKey(1000)

                listener.startrecording()

                while listener.isrecording:
                    img = np.zeros((1000, 1400, 3), dtype = np.uint8)
                    cv2.putText(img, f"Recording gesture: {gesture}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255))
                    progress = listener.frames / 90.0
                    bar_width = int(430 * progress)
                    cv2.rectangle(img, (10, 80), (10 + bar_width, 120), (0, 255, 0), -1)
                    cv2.rectangle(img, (10, 80), (440, 120), (255, 255, 255), 2)
                    cv2.putText(img, f"{listener.frames}/90 frames", (150, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    visualizer_2d = img[:, 450:]
                    hands = listener.get_hand()
                    if hands:
                        visualize_hand(visualizer_2d, hands)
                    
                    cv2.imshow("Preprocessing", img)
                    cv2.waitKey(10)

                
                if len(os.listdir(os.path.join(data_dir, gesture.replace(" ", "_")))) > sample:
                    while True:
                        img = np.zeros((1000, 1400, 3), dtype = np.uint8)
                        cv2.putText(img, f"Sample {sample+1} saved, press 's' to save next sample or 'r' to record again", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255))
                        cv2.imshow("Preprocessing", img)
                        key = cv2.waitKey(0)

                        if key == ord('s'):
                            sample += 1
                            break
                        elif key == ord('r'):
                            filename = f"{gesture.replace(' ', '_')}_{sample:03d}.npy"
                            os.remove(os.path.join(data_dir, gesture.replace(" ", "_"), filename))
                            break
                else:
                    print(f"Sample {sample+1} not saved, buffer too small")
                
                connection.remove_listener(listener)
   
    cv2.destroyAllWindows()
    return True


if __name__ == "__main__":
    collect_samples()