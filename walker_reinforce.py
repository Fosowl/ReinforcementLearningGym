
# solution to walker problem using policy gradient

import os
import gym
import random
import math
from collections import namedtuple, deque
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pprint import pprint
import signal
import time

from helper import *

# fix mac problem with matplotlib window
matplotlib.use('Tkagg')

SHOW_EVERY = 50
RESET_EPSILON_EVERY = 500
SAVE_RETURN_EVERY = 10
SAVE_MODEL_EVERY = 100
KEEP_ACTION = 5

# observation name to index
obs_names = {
    "hullAngleSpeed": 0,
    "angularVelocity": 1,
    "horizontalSpeed": 2,
    "verticalSpeed": 3,
    "positionOfJoints1": 4,
    "positionOfJoints2": 5,
    "positionOfJoints3": 6,
    "positionOfJoints4": 7,
    "jointsAngularSpeed1": 8,
    "jointsAngularSpeed2": 9,
    "jointsAngularSpeed3": 10,
    "jointsAngularSpeed4": 11,
    "legsContactWithGround1": 12,
    "legsContactWithGround2": 13,
    "lidar1": 14,
    "lidar2": 15,
    "lidar3": 16,
    "lidar4": 17,
    "lidar5": 18,
    "lidar6": 19,
    "lidar7": 20,
    "lidar8": 21,
    "lidar9": 22,
    "lidar10": 23
}

# 1 is pink, 2 is darker leg
motors = {
    "hip1": 0,
    "knee1": 3,
    "hip2": 2,
    "knee2": 1
}

class PolicyAgent:
    class NeuralNetwork(nn.Module):
        def __init__(self, n_states, n_actions):
            super().__init__()
            self.l1 = nn.Linear(n_states, 64)
            self.l2 = nn.Linear(64, 128)
            self.l3 = nn.Linear(128, 128)
            self.l4 = nn.Linear(128, 128)
            self.l5 = nn.Linear(128, 128)
            self.l6 = nn.Linear(128, 128)
            self.l7 = nn.Linear(128, 64)
            self.l8 = nn.Linear(64, n_actions)

        def forward(self, x):
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            x = F.relu(self.l3(x))
            x = F.relu(self.l4(x))
            x = F.relu(self.l5(x))
            x = F.relu(self.l6(x))
            x = F.relu(self.l7(x))
            x = torch.sigmoid(self.l8(x))
            return x
    
    def __init__(self, n_actions, n_states, scanner=False, pretrained=False):
        # initialize all variables
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = 0.9 if pretrained == False else 0.0
        self.epsilon_decay = .998
        self.learning_rate = 0.001
        self.gamma = .96
        self.rewards = 0
        self.pretrained = pretrained
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'noise', 'done'))
        self.batch_size = 32
        self.memory = deque([], maxlen=16000)
        self.model = self.NeuralNetwork(n_states, n_actions)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, amsgrad=True)
        self.train_loss = 0
        if scanner:
            self.scanner = brainScan(self.model)
        if pretrained:
            try:
                print("loading pretrained neural network...")
                self.model.load_state_dict(torch.load('checkpoint_actor.pth'))
            except Exception as e:
                print("could not load pretrained model!")

    def memorizeTransition(self, state, action, next_state, reward, noise, done):
        # append transition to batch
        self.rewards += reward
        transition = self.transition(torch.tensor([state], dtype=torch.float32),
                                     torch.tensor([[action]], dtype=torch.float32),
                                     torch.tensor([next_state]),
                                     torch.tensor([reward], dtype=torch.float32),
                                     torch.tensor([noise]),
                                     done)
        self.memory.append(transition)
    
    def save(self):
        torch.save(self.model.state_dict(), 'checkpoint_actor.pth')

    def learnBatch(self):
        # update parameters from batch and optimize
        sample_count = 0
        # discount return (reward sum)
        G_t = torch.tensor([0], dtype=torch.float32)
        # policy
        policy_loss = []
        for transition in reversed(self.memory):
            actions_choice = self.model(transition.state) + transition.noise
            r = transition.reward
            G_t = r + self.gamma * G_t
            log_action_probs = torch.log(torch.clamp(actions_choice, min=1e-5, max=1-1e-5))
            tmp = -log_action_probs * G_t
            policy_loss.append(tmp)
            sample_count += 1
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.resetBatch()
        return G_t.item()

    def resetBatch(self):
        # reset current memory batch to prepare for next one
        self.memory = deque([], maxlen=1600)

    def act(self, state, use_noise=True):
        with torch.no_grad():
            actions = self.model(torch.tensor([state], dtype=torch.float32))
        actions_np = actions.numpy()
        # Add exploration noise (adjust the scale as needed)
        noise = np.random.normal(scale=self.epsilon, size=actions_np.shape)
        if use_noise == False:
            noise = np.zeros(actions_np.shape)
        actions_np += noise
        motors = actions_np.squeeze().clip(-1, 1)
        return motors, noise

def reward_good_posture(state):
    # Define target torso angle for an upright posture (adjust as needed)
    target_torso_angle = 0.0
    torso_angle = state[obs_names["hullAngleSpeed"]]
    angle_deviation = abs(torso_angle - target_torso_angle) * (180/math.pi)
    punish = (math.exp(angle_deviation) / (180/math.pi) * 0.001) 
    reward = 1 - punish 
    if reward > 1:
        reward = 0.2
    if reward < -1:
        reward = -0.2
    return reward

def have_opposite_signs(a, b):
    return (a < 0 and b > 0) or (a > 0 and b < 0)

def reward_coordinated_walk(state):
    reward = 0.0
    penalty = 0.2
    join1 = state[obs_names["jointsAngularSpeed1"]]
    join2 = state[obs_names["jointsAngularSpeed2"]]
    join3 = state[obs_names["jointsAngularSpeed3"]]
    join4 = state[obs_names["jointsAngularSpeed4"]]
    if have_opposite_signs(join1, join3):
        reward += penalty
    else:
        reward -= penalty
    if have_opposite_signs(join2, join4):
        reward += penalty
    else:
        reward -= penalty
    if join1 < 0 and join3 < 0:
        reward -= penalty * 2
    if join2 < 0 and join4 < 0:
        reward -= penalty * 2
    return reward

def demonstrate(env, agent):
    state = env.reset()
    state = state[0]
    done = False
    score = 0
    steps = 0
    action, _ = agent.act(state)
    print("simulating...")
    speeds = []
    while steps < 500:
        action, _ = agent.act(state, False)
        next_state, reward, done, info, _ = env.step(action)
        env.render()
        print("base reward : ", reward)
        print("coordination : ", reward_coordinated_walk(state))
        print("posture reward : ", reward_good_posture(state))
        if done:
            print("dead")
            break
        steps += 1
        speeds.append(state[1])
        if sum(speeds) / len(speeds) <= 0.0 and steps > 50:
            break
        score += reward
        state = next_state
    print(f"Finished after {steps} steps, got score {score}")

abort_training = False

def signal_handler(sig, frame):
    global abort_training
    abort_training = True
    print('interupting...')

def training(episodes, env, agent, plotting, scan, env_demo = None):
    durations = []
    returns = []
    signal.signal(signal.SIGINT, signal_handler)
    for episode in range(1, episodes+1):
        if abort_training:
            break
        state = env.reset()
        state = state[0]
        done = False
        score = 0
        steps = 0
        action = agent.act(state)
        if episode % SHOW_EVERY == 0 and env_demo != None:
            print(f"showing agent performance at episode : {episode}")
            demonstrate(env_demo, agent)
        speeds = []
        while steps < math.log(episode)*100+1:
            if steps % KEEP_ACTION == 0:
                action, noise = agent.act(state)
            next_state, reward, done, info, _ = env.step(action)
            speeds.append(state[1])
            if sum(speeds) / len(speeds) <= 0.0 and steps > 50:
                break
            reward += reward_good_posture(state)
            reward += reward_coordinated_walk(state)
            agent.memorizeTransition(state, action, next_state, reward, noise, done)
            if done:
                break
            steps += 1
            score += reward
            state = next_state
        G = agent.learnBatch()
        agent.epsilon *= agent.epsilon_decay
        returns.append(score)
        print(f"Episode {episode}, Score : {round(score, 2)}, Steps : {steps}, epsilon : {agent.epsilon}")
        if episode % SAVE_RETURN_EVERY == 0:
            returns.append(G)
        if episode > SAVE_MODEL_EVERY and episode % SAVE_MODEL_EVERY == 0:
            agent.save()
        if plotting is not None:
            plotting.plot_values(returns)
        if scan is not None:
            scan.show()
        durations.append(steps)
    if plotting is not None:
        plotting.save()
    print(f"Episode {episode}, score : {score}, Return : {G}")

def main():
    training_mode = True
    plotting = None
    env = gym.make("BipedalWalker-v3", hardcore=False)
    env_demo = gym.make("BipedalWalker-v3", hardcore=False, render_mode="human")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    agent = PolicyAgent(n_actions, n_states, False, pretrained=(training_mode == False))
    scan = brainScan(agent.model)
    episodes = 10000
    if training_mode:
        training(episodes, env, agent, plotting, scan, env_demo)
    else:
        demonstrate(env_demo, agent)
    env.close()

if __name__ == "__main__":
    main()
