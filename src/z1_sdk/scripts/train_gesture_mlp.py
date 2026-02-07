#!/usr/bin/python3
"""
Train MLP for Gesture-to-Action Mapping

Uses data collected by collector.py to train a neural network
that maps human gestures to robot actions.

Usage:
    python3 train_gesture_mlp.py --data gesture_data.csv [--epochs 100]
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not found. Install with: pip install torch")


# ==================== Model Definition ====================

class GestureActionMLP(nn.Module):
    """
    Multi-layer Perceptron for gesture-to-action mapping.
    
    Input features:
        - gesture_id (one-hot encoded, 16 classes)
        - gesture_confidence (1)
        - gesture_velocity (3)
        - ee_position (3)
        - joint_positions (6)
        - prev_action_id (one-hot encoded, 15 classes)
        
    Output:
        - action_id (15 classes)
    """
    
    def __init__(self, 
                 num_gestures: int = 16,
                 num_actions: int = 15,
                 hidden_sizes: list = [128, 64, 32]):
        super().__init__()
        
        self.num_gestures = num_gestures
        self.num_actions = num_actions
        
        # Input size: gesture one-hot + confidence + velocity + ee_pos + joints + prev_action one-hot
        input_size = num_gestures + 1 + 3 + 3 + 6 + num_actions
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_actions))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# ==================== Dataset ====================

class GestureActionDataset(Dataset):
    """Dataset for gesture-action pairs"""
    
    def __init__(self, csv_path: str, num_gestures: int = 16, num_actions: int = 15):
        self.df = pd.read_csv(csv_path)
        self.num_gestures = num_gestures
        self.num_actions = num_actions
        
        print(f"Loaded {len(self.df)} samples from {csv_path}")
        print(f"Gesture distribution:\n{self.df['gesture'].value_counts()}")
        print(f"Action distribution:\n{self.df['action'].value_counts()}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # One-hot encode gesture
        gesture_onehot = np.zeros(self.num_gestures, dtype=np.float32)
        gesture_id = int(row['gesture_id'])
        if 0 <= gesture_id < self.num_gestures:
            gesture_onehot[gesture_id] = 1.0
        
        # One-hot encode previous action
        prev_action_onehot = np.zeros(self.num_actions, dtype=np.float32)
        prev_action_id = int(row['prev_action_id'])
        if 0 <= prev_action_id < self.num_actions:
            prev_action_onehot[prev_action_id] = 1.0
        
        # Build feature vector
        features = np.concatenate([
            gesture_onehot,
            [row['gesture_confidence']],
            [row['gesture_vel_x'], row['gesture_vel_y'], row['gesture_vel_z']],
            [row['ee_x'], row['ee_y'], row['ee_z']],
            [row['joint_0'], row['joint_1'], row['joint_2'], 
             row['joint_3'], row['joint_4'], row['joint_5']],
            prev_action_onehot,
        ]).astype(np.float32)
        
        # Target
        target = int(row['action_id'])
        
        return torch.tensor(features), torch.tensor(target)


# ==================== Training ====================

def train_model(model, train_loader, val_loader, epochs, device, lr=0.001):
    """Train the MLP model"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_acc = 0
    best_model_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.1f}% | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.1f}%")
    
    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return history, best_val_acc


def export_for_inference(model, output_path: str):
    """Export model for inference"""
    # Save PyTorch model
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_gestures': model.num_gestures,
        'num_actions': model.num_actions,
    }, output_path)
    print(f"Saved model to {output_path}")
    
    # Also export as ONNX for cross-platform inference
    try:
        dummy_input = torch.randn(1, model.num_gestures + 1 + 3 + 3 + 6 + model.num_actions)
        onnx_path = output_path.replace('.pt', '.onnx')
        torch.onnx.export(model, dummy_input, onnx_path,
                         input_names=['features'],
                         output_names=['action_logits'],
                         dynamic_axes={'features': {0: 'batch_size'},
                                      'action_logits': {0: 'batch_size'}})
        print(f"Saved ONNX model to {onnx_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='Train Gesture-Action MLP')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to CSV data file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--output', type=str, default=None,
                        help='Output model path')
    args = parser.parse_args()
    
    if not HAS_TORCH:
        print("PyTorch required for training!")
        sys.exit(1)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    dataset = GestureActionDataset(args.data)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create model
    model = GestureActionMLP(
        num_gestures=dataset.num_gestures,
        num_actions=dataset.num_actions,
        hidden_sizes=[128, 64, 32]
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\nTraining...")
    history, best_acc = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs,
        device=device,
        lr=args.lr
    )
    
    print(f"\nBest validation accuracy: {best_acc:.1f}%")
    
    # Save model
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"gesture_mlp_{timestamp}.pt"
    
    export_for_inference(model, args.output)
    
    # Save training history
    history_path = args.output.replace('.pt', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()

