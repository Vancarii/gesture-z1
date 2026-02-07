import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import torch
import torch.nn as nn
import torch.optim as optim


GESTURE_MAP = {
    "swipe back": 0,
    "swipe towards": 1,
    "point up": 2,
    "point down": 3,
    "pull back": 4,
    "background": 5
}

GESTURE_ALIASES = {
    "point forward": "swipe towards",
    "point back": "swipe back",
    "move hand up": "point up",
    "move hand down": "point down",
    "pause": "background",
}

CUSTOM_ACTIONS = {
    "point left": {"cmdid": 6, "velocity": np.array([-0.2, 0.0, 0.0]), "gripper": 0.0},
    "point right": {"cmdid": 7, "velocity": np.array([0.2, 0.0, 0.0]), "gripper": 0.0},
}

class GestureMLP(nn.Module):
    def __init__(self, numgestures, hiddensize, numactions):
        super(GestureMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(numgestures, hiddensize),
            nn.ReLU(),
            nn.Linear(hiddensize, hiddensize),
            nn.ReLU(),
            nn.Linear(hiddensize, numactions),
        )
    
    def forward(self, x):
        return self.network(x)

class RoboticAgent(gym.Env):
    def __init__(self):
        super(RoboticAgent, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.currpos = np.array([0.5, 0.0, 0.5])
        self.gripperpos = 0.0
        self.currcmd = 5

        self.steps = 0
        self.maxsteps = 100

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.currpos = np.array([0.5, 0.0, 0.5])
        self.gripperpos = 0.0
        self.currcmd = 5
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([
            self.currpos, [self.gripperpos], [self.currcmd]
        ]).astype(np.float32)

    def setcmd(self, cmdid):
        self.currcmd = cmdid

    def step(self, action):
        self.steps+=1

        dt = 0.1
        self.currpos = self.currpos + action[:3] * dt
        self.gripperpos = np.clip(self.gripperpos + action[3]*dt, 0, 1)
        reward = 0.0

        targets = {
            0: np.array([0.0, -1, 0.0]),
            1: np.array([0.0, 1, 0.0]),
            2: np.array([0.0, 0.0, 1]),
            3: np.array([0.0, 0.0, -1]),
            5: np.array([0.0, 0.0, 0.0]),
        }

        if self.currcmd in targets:
            vector = targets[self.currcmd]
            velocity = action[:3]
            align = np.dot(vector, velocity)
            reward += align

            if self.currcmd == 5:
                reward -= np.linalg.norm(velocity) * 0.5
        
        if self.currcmd == 4:
            if abs(action[3]) > 0.5: reward += 1.0
        
        if np.any(np.abs(self.currpos) > 1.0):
            reward -= 1.0
        
        return self._get_obs(), reward, self.steps >= self.maxsteps, False, {}

class Brain:
    def __init__(self, train=True):
        self.gestures = list(GESTURE_MAP.keys())
        self.env = RoboticAgent()

        model_path = "src/architecture/models/policy_ppo"
        model_file = f"{model_path}.zip"

        if train:
            print("Training policy...")
            self.model = PPO("MlpPolicy", self.env, verbose=1)
            self.model.learn(total_timesteps=10000)
            self.model.save(model_path)
            print("Policy trained and saved")
        else:
            if os.path.exists(model_file):
                print("Loading trained policy...")
                self.model = PPO.load(model_path, env=self.env)
                print("Policy loaded")
            else:
                print("No saved policy found, training a new one...")
                self.model = PPO("MlpPolicy", self.env, verbose=1)
                self.model.learn(total_timesteps=10000)
                self.model.save(model_path)
                print("Policy trained and saved")

    def action(self, gesture):
        if gesture in CUSTOM_ACTIONS:
            return self._custom_action(gesture, CUSTOM_ACTIONS[gesture])

        canonical = GESTURE_ALIASES.get(gesture, gesture)

        if canonical not in GESTURE_MAP:
            print(f"Invalid gesture: {gesture}")
            return None
        
        cmdid = GESTURE_MAP[canonical]
        self.env.setcmd(cmdid)
        obs = self.env._get_obs()
        action, _ = self.model.predict(obs, deterministic=True)
        currpos = obs[:3]
        targetpos = currpos + (action[:3] * 0.5)
        grippercmd = action[3]

        return {
            "velocity": action[:3],
            "targetpos": targetpos,
            "grippercmd": grippercmd,
            "cmdid": cmdid
        }

    def _custom_action(self, gesture, spec):
        obs = self.env._get_obs()
        currpos = obs[:3]
        velocity = spec["velocity"]
        targetpos = currpos + (velocity * 0.5)
        grippercmd = spec.get("gripper", 0.0)
        cmdid = spec["cmdid"]

        return {
            "velocity": velocity,
            "targetpos": targetpos,
            "grippercmd": grippercmd,
            "cmdid": cmdid,
        }

def simulation():
    brain = Brain(train=False)

    gestures = ["swipe back","swipe towards", "point up", "point down", "pull back", "background"]

    for gesture in gestures:
        curraction = brain.action(gesture)
        if curraction is not None:
            print(f"Gesture: {gesture}, Action: {curraction}")
            print(f"Velocity: {curraction['velocity']}, Target Pos: {curraction['targetpos']}, Gripper Cmd: {curraction['grippercmd']}, Cmd ID: {curraction['cmdid']}")

if __name__ == "__main__":
    simulation()
    