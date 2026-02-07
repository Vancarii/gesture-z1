import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

GESTURES = [
    "point up", "point down", "point forward", "point back",
    "point left", "point right",
    "move hand up", "move hand down",
    "pause", 
    "swipe towards", "swipe back", "pull back",
    "background"
]

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=19, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5),
            nn.AdaptiveMaxPool1d(output_size=20)
        )

        #90 frames -> conv -> 86 -> /2 -> 43 -> conv3 -> 41 -> /2 -> 20
        flatten_size = 128 * 20 
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = x.transpose(1, 2) # x -> (batch, 19, 90)
        x=self.cnn(x)
        x=self.fc(x)
        return x


def load_data():
    X, Y = [], []

    for label in GESTURES:
        labeldir = label.replace(" ", "_")
        path = os.path.join(DATA_DIR, labeldir)

        if not os.path.isdir(path):
            continue

        files = os.listdir(path)
        count = 0
        for file in files:
            if file.endswith(".npy"):
                data = np.load(os.path.join(path, file))

                if data.shape == (90, 19) and not np.isnan(data).any():
                    X.append(data)
                    Y.append(label)
                    count +=1
        
        print(f"Loaded {count} samples for {label}")
    return np.array(X), np.array(Y)

def CNN_Model():
    X, Y = load_data()

    if len(X) == 0 or len(Y) == 0:
        print("No data to train model")
        return
    
    encoder = LabelEncoder()
    Y_encoded = encoder.fit_transform(Y)
    np.save(os.path.join(MODELS_DIR, "classes.npy"), encoder.classes_)
    print(f"Classes: {encoder.classes_}")

    scaler = StandardScaler()
    X_flat = X.reshape(-1, 19)
    X_scaledflat = scaler.fit_transform(X_flat)
    X_scaled = X_scaledflat.reshape(X.shape)

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scalerCNN.pkl"))

    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_encoded, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    Y_test = torch.tensor(Y_test, dtype=torch.long)

    BATCH_SIZE = 16
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(encoder.classes_)
    model = Net(num_classes)
    if torch.cuda.is_available():
        model.to("cuda")
    else:
        print("CUDA not available, using CPU")
        return None

    fnlosses = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #Training loop
    EPOCHS = 30
    for epoch in range(EPOCHS):
        model.train()
        runningloss = 0.0

        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to("cuda"), Yb.to("cuda")
            optimizer.zero_grad()
            predictions = model(Xb)
            loss = fnlosses(predictions, Yb)
            loss.backward()
            optimizer.step()
            runningloss += loss.item()
        
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for Xb, Yb in test_loader:
                Xb, Yb = Xb.to("cuda"), Yb.to("cuda")
                predictions = model(Xb)
                i, predicted = torch.max(predictions, 1)
                correct += (predicted == Yb).sum().item()
                total += Yb.size(0)
        
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {runningloss:.4f}, Accuracy: {accuracy:.4f}")
    
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "gesture_model_cnn.pth"))


if __name__ == "__main__":
    CNN_Model()

