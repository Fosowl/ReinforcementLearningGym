
# solution to walker problem using policy gradient

import os
import gym
import random
import math
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pprint import pprint
import time

from helper import *

class PolicyAgent:
    class NeuralNetwork(nn.Module):
        def __init__(self, n_states, n_actions):
            super().__init__()
            self.l1 = nn.Linear(n_states, 64)
            self.l2 = nn.Linear(64, 64)
            self.l3 = nn.Linear(64, 64)
            self.l4 = nn.Linear(64, n_actions)

        def forward(self, x):
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            x = F.relu(self.l3(x))
            x = torch.sigmoid(self.l4(x))
            return x
    
    def __init__(self, n_actions, n_states, scanner=False, pretrained=True):
        # initialize all variables
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = 1.0
        self.epsilon_decay = .999
        self.learning_rate = 0.001
        self.gamma = .996
        self.rewards = 0
        self.pretrained = pretrained
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'noise', 'done'))
        self.batch_size = 100
        self.memory = deque([], maxlen=self.batch_size)
        self.model = self.NeuralNetwork(n_states, n_actions)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, amsgrad=True)
        self.train_loss = 0
        if scanner:
            self.scanner = brainScan(self.model)
        if pretrained:
            try:
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
        G_t = 0
        # policy
        policy_loss = []
        lst = None
        for transition in reversed(self.memory):
            if self.pretrained == False:
                actions_choice = self.model(transition.state) + transition.noise
            else:
                actions_choice = self.model(transition.state)
            r = transition.reward + (actions_choice.abs().sum() / 8)
            G_t = r + self.gamma * G_t
            log_action_probs = torch.log(torch.clamp(actions_choice, min=1e-8, max=1-1e-8))
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
        self.epsilon *= self.epsilon_decay
        self.memory = deque([], maxlen=self.batch_size)

    def act(self, state):
        with torch.no_grad():
            actions = self.model(torch.tensor([state], dtype=torch.float32))
        actions_np = actions.numpy()
        # Add exploration noise (adjust the scale as needed)
        noise = np.random.normal(scale=self.epsilon, size=actions_np.shape)
        if self.pretrained == False:
            actions_np += noise * 2
        motors = actions_np.squeeze() * 2
        return motors, noise

def reward_good_posture(state):
    # Define target torso angle for an upright posture (adjust as needed)
    target_torso_angle = 0.0
    torso_angle = state[2]
    angle_deviation = abs(torso_angle - target_torso_angle) * (180/math.pi)
    reward = 1 - (math.exp(angle_deviation) / (180/math.pi) * 0.1)
    if reward < -1:
        reward = -1
    return reward

def demonstrate(env, agent):
    state = env.reset()
    state = state[0]
    done = False
    score = 0
    steps = 0
    action = agent.act(state)
    while steps < 500:
        action, _ = agent.act(state)
        next_state, reward, done, info, _ = env.step(action)
        print("posture : ", reward_good_posture(state))
        env.render()
        if done:
            print("dead")
            break
        steps += 1
        score += reward
        state = next_state
    print(f"Finished after {steps} steps, got score {score}")

def training(episodes, env, agent, plotter, env_demo = None):
    durations = []
    returns = []
    for episode in range(1, episodes+1):
        state = env.reset()
        state = state[0]
        done = False
        score = 0
        steps = 0
        if episode % 10 == 0:
            action = agent.act(state)
        if episode % 25 == 0 and env_demo != None:
            print(f"showing agent performance at episode : {episode}")
            demonstrate(env_demo, agent)
        while steps < math.log(episode)*100+1:
            action, noise = agent.act(state)
            next_state, reward, done, info, _ = env.step(action)
            reward += reward_good_posture(state)
            if done:
                print("dead")
                break
            agent.memorizeTransition(state, action, next_state, reward, noise, done)
            steps += 1
            score += reward
            state = next_state
        G = agent.learnBatch()
        print(f"Episode {episode}, Return : {G}, Steps : {steps}")
        if episode > 100:
            agent.save()
        returns.append(G)
        plotter.plot_values(returns)
        durations.append(steps)
        #agent.scanner.show()

    print(f"Episode {episode}, score : {score}, Return : {G}")


def main():
    training_mode = True
    env = gym.make("BipedalWalker-v3", hardcore=False)
    env_demo = gym.make("BipedalWalker-v3", hardcore=False, render_mode="human")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    agent = PolicyAgent(n_actions, n_states, False, pretrained=(not training_mode))
    plotting = plotter()
    episodes = 1000
    if training_mode:
        training(episodes, env, agent, plotting, env_demo)
    else:
        demonstrate(env_demo, agent)
    env.close()


if __name__ == "__main__":
    main()
