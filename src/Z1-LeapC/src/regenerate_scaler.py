#!/usr/bin/env python
"""Regenerate the scaler from training data to fix numpy compatibility issues."""

import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

DATA_DIR = "architecture/data"
MODELS_DIR = "architecture/models"

GESTURES = [
    "point up", "point down", "point forward", "point back",
    "point left", "point right",
    "move hand up", "move hand down",
    "pause", 
    "swipe towards", "swipe back", "pull back",
    "background"
]

def regenerate_scaler():
    X = []
    
    for label in GESTURES:
        labeldir = label.replace(" ", "_")
        path = os.path.join(DATA_DIR, labeldir)
        
        if not os.path.isdir(path):
            print(f"Warning: Directory not found: {path}")
            continue
        
        files = os.listdir(path)
        count = 0
        for file in files:
            if file.endswith(".npy"):
                data = np.load(os.path.join(path, file))
                
                if data.shape == (90, 19) and not np.isnan(data).any():
                    X.append(data)
                    count += 1
        
        print(f"Loaded {count} samples for {label}")
    
    if len(X) == 0:
        print("ERROR: No data found to regenerate scaler!")
        return
    
    X = np.array(X)
    print(f"Total samples: {len(X)}")
    
    # Recreate scaler
    scaler = StandardScaler()
    X_flat = X.reshape(-1, 19)
    scaler.fit(X_flat)
    
    # Save new scaler
    scaler_path = os.path.join(MODELS_DIR, "scalerCNN.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler regenerated and saved to {scaler_path}")

if __name__ == "__main__":
    regenerate_scaler()
