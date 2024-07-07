
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

SHOW_EVERY = 40
RESET_EPSILON_EVERY = 15000
SAVE_RETURN_EVERY = 10
SAVE_MODEL_EVERY = 100
KEEP_ACTION = 0

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
    "jointsSpeedHipL": 8,
    "jointsSpeedKneeL": 9,
    "jointsSpeedHipR": 10,
    "jointsSpeedKneeR": 11,
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
            self.l2 = nn.Linear(64, 64)
            self.l3 = nn.Linear(64, 64)
            self.l4 = nn.Linear(64, n_actions)
            self._initialize_weights()

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x):
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            x = F.relu(self.l3(x))
            x = F.tanh(self.l4(x))
            return x
    
    def __init__(self, n_actions, n_states, scanner=False, pretrained=False):
        # initialize all variables
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = 0.8 if pretrained == False else 0.0
        self.epsilon_decay = .9996
        self.learning_rate = 1e-4
        self.gamma = .99
        self.rewards = 0
        self.pretrained = pretrained
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'noise', 'done'))
        self.batch_size = 64
        self.memory = deque([], maxlen=1000000)
        self.model = self.NeuralNetwork(n_states, n_actions)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, amsgrad=True)
        self.train_loss = 0
        self.past_grads = []
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

    def detect_exploding_gradients(self, threshold=1000):
        exploding_gradients = False
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    gradient_norm = torch.norm(param.grad)
                    if gradient_norm > threshold:
                        print(f"Exploding gradient detected in parameter: {name}")
                        print(f"Gradient norm: {gradient_norm}")
                        exploding_gradients = True
                        exit()
        return exploding_gradients

    def detect_vanishing_gradients(self, threshold=1e-5):
        count_vanished = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    gradient_norm = torch.norm(param.grad)
                    if gradient_norm < threshold:
                        count_vanished += 1
        percent_vanished = count_vanished / len(list(self.model.parameters()))
        if percent_vanished > 0.5:
            print(f"Vanishing gradients detected in more than 50% of parameters")
            exit()
        return False

    
    def get_avg_gradients(self):
        avg_gradients = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                avg_gradients.append(round(param.grad.mean().item(), 2))
        return avg_gradients

    def learnBatch(self):
        if len(self.memory) < self.batch_size:
            return 0
        # update parameters from batch and optimize
        sample_count = 0
        # discount return (reward sum)
        G_t = torch.tensor([0], dtype=torch.float32)
        # policy
        policy_loss = []
        memories = list(self.memory)
        mem_chunks = [memories[i:i+self.batch_size] for i in range(0, len(memories), self.batch_size)]
        print(f"Learning from {len(memories)} memory transition...")
        for transition in mem_chunks[random.randint(0, len(mem_chunks)-1)]:
            actions_choice = self.model(transition.state) + transition.noise
            r = transition.reward
            G_t = r + self.gamma * G_t
            log_action_probs = torch.log(torch.clamp(actions_choice, min=1e-3, max=1-1e-3))
            prob = -log_action_probs * G_t
            policy_loss.append(prob)
            sample_count += 1
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        gradients_diff = sum(self.get_avg_gradients()) - sum(self.past_grads)
        #print(f"Backprop gradients diff: {round(gradients_diff, 2)}")
        self.past_grads = self.get_avg_gradients()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
        self.optimizer.step()
        self.detect_exploding_gradients()
        self.detect_vanishing_gradients()
        return G_t.item()

    def resetShortTermMemory(self):
        # reset current memory batch to prepare for next one
        self.memory = deque([], maxlen=1000000)

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
    # Define target torso angle for an upright posture
    reward = -abs(state[obs_names['hullAngleSpeed']]) * 0.01
    return reward

def have_opposite_signs(a, b):
    return (a < 0 and b > 0) or (a > 0 and b < 0)

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
        action, _ = agent.act(state, use_noise=False)
        next_state, reward, done, info, _ = env.step(action)
        print(f"Reward : {round(reward, 2)}, Action: {action}")
        env.render()
        if done:
            print("dead")
            break
        steps += 1
        speeds.append(state[1])
        score += reward
        state = next_state
    print(f"Finished after {steps} steps, got score {score}")
    time.sleep(1)

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
            print(f"Showing agent performance at episode : {episode}")
            demonstrate(env_demo, agent)
        speeds = []
        while steps < 600:
            action, noise = agent.act(state, use_noise=True)
            next_state, reward, done, info, _ = env.step(action)
            speeds.append(state[1])
            agent.memorizeTransition(state, action, next_state, reward, noise, done)
            if done:
                break
            steps += 1
            score += reward
            state = next_state
        G = agent.learnBatch()
        agent.epsilon *= agent.epsilon_decay
        agent.resetShortTermMemory()
        returns.append(score)
        print(f"Learning memory... {episode}/{episodes}, avg G: {G} Epsilon : {round(agent.epsilon, 2)}")
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
    plotting = plotter()
    env = gym.make("BipedalWalker-v3", hardcore=False)
    env_demo = gym.make("BipedalWalker-v3", hardcore=False, render_mode="human")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    agent = PolicyAgent(n_actions, n_states, scanner=False, pretrained=(training_mode == False))
    scan = None
    scan = brainScan(agent.model) 
    episodes = 3000
    if training_mode:
        training(episodes, env, agent, plotting, scan, env_demo)
    else:
        demonstrate(env_demo, agent)
    env.close()

if __name__ == "__main__":
    main()
