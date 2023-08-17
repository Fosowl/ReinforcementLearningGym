# solution to gym lunar lander problem using DQN

import gym
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pprint import pprint

env = gym.make('LunarLander-v2', render_mode="human")

print("-- ENVIRONMENTS ---")
print(f"Observation space : {env.observation_space.shape[0]}")
print(f"action space : {env.action_space.n}")

n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

class DQNagent:
    class NeuralNetwork(nn.Module):
        def __init__(self, n_states, n_actions):
            super().__init__()
            self.l1 = nn.Linear(n_states, 128)
            self.l2 = nn.Linear(128, 128)
            self.l3 = nn.Linear(128, n_actions)
        
        def forward(self, x):
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            return self.l3(x)

    def __init__(self, n_states, n_actions):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = 1.0
        self.gamma = .99
        self.batch_size = 64
        self.tau = 0.005 # update rate
        self.learning_rate = 0.001
        self.epsilon_decay = .996
        self.update_rate = 0.005
        self.memory = deque([], maxlen=1000000)
        # policy neural net
        self.policy_dqn = self.NeuralNetwork(n_states, n_actions)
        # adjustement neural net
        self.target_dqn = self.NeuralNetwork(n_states, n_actions)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        self.optimizer = optim.AdamW(self.policy_dqn.parameters(), lr=self.learning_rate, amsgrad=True)
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

    def memorize(self, state, action, next_state, reward, done):
        trans = self.transition(torch.tensor([state], dtype=torch.float32),
                                torch.tensor([[action]], dtype=torch.int64),
                                torch.tensor([next_state]),
                                torch.tensor([reward], dtype=torch.float32),
                                done)
        self.memory.append(trans)

    def replay(self):
        # check enought experience
        if len(self.memory) < self.batch_size:
            return
        transitions = random.sample(self.memory, self.batch_size)
        # converts batch-array of Transitions to inverse
        batch = self.transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_dqn(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_dqn(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_dqn.parameters(), 100)
        self.optimizer.step()
        self.epsilon *= self.epsilon_decay
    
    def sync_dqn(self):
        target_net_state_dict = self.target_dqn.state_dict()
        policy_net_state_dict = self.policy_dqn.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1-self.tau)
        self.target_dqn.load_state_dict(target_net_state_dict)
        
    def act(self, state):
        # random
        if np.random.rand() <= self.epsilon:
            return random.randrange(n_actions)
        # greedy
        with torch.no_grad():
            choices = self.policy_dqn(torch.tensor([state], dtype=torch.float32))
        return np.argmax(choices[0].numpy())

episodes = 500
agent = DQNagent(n_states, n_actions)

for episode in range(1, episodes+1):
    state = env.reset()
    state = state[0]
    done = False
    score = 0
    while True:
        action = agent.act(state)
        next_state, reward, done, info, _ = env.step(action)
        if done:
            break
        agent.memorize(state, action, next_state, reward, done)
        agent.replay()
        agent.sync_dqn()
        score += reward
        state = next_state
        env.render()
    print(f"Episode {episode}, score : {score}")

env.close()