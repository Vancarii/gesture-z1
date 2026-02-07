import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

def visualize_gesture(label, data_dir="data"):
    labeldir = os.path.join(data_dir, label)
    paths = sorted(glob(os.path.join(labeldir, f"*.npy")))

    plt.figure(figsize=(10, 4))
    for path in paths:
        data = np.load(path)

        if label == "clap":
            traject = data[:, 0]
            plt.ylabel("palm distance")
        else:
            traject = data[:, 4]
            plt.ylabel("index tip y position")
        
        plt.plot(traject, label=os.path.basename(path), alpha=0.5)
    
    plt.title(f"Gesture trajectories (label={label})")
    plt.xlabel("Frame index")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"visualizations/{label}.png")
    plt.close()


if __name__ == "__main__":
    os.makedirs("visualizations", exist_ok=True)
    gesture = ['clap', 'move_hand_down', 'move_hand_up', 'pause', 'point_back',
    'point_down', 'point_forward', 'point_left', 'point_right', 'point_up', 
    'pull_back', 'swipe_back', 'swipe_towards']
    for g in gesture:
        visualize_gesture(g)