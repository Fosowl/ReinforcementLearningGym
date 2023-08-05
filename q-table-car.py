
# First RL problem I solved

import gym
import random
import numpy as np

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 25

env = gym.make('MountainCar-v0', render_mode='human')
states = env.observation_space.shape[0]
actions = env.action_space.n

DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

print("--- world info : ---")
print(f"observation high {env.observation_space.high}")
print(f"observation low {env.observation_space.low}")
print(f"Actions count {env.action_space.n} ")

q_table_dim = DISCRETE_OS_SIZE + [env.action_space.n]
print(f"Q-table size {q_table_dim}\n---")

q_table = np.random.uniform(low=-2, high=0, size=(q_table_dim))

def get_discrete_state(state):
    discrete_state = (np.array(state, dtype=np.float32) - env.observation_space.low) / discrete_os_win_size
    discrete_state = (int(discrete_state[0]), int(discrete_state[1]))
    return discrete_state

for episode in range(1, EPISODES+1):
    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    done = False
    score = 0
    print(episode)
    display = False
    if episode % SHOW_EVERY == 0:
        display = True
        print("Got score : {} at episode {}".format(score, episode))
    while not done:
        if display == True:
            env.render()
        display = False

        if np.random.random() > epsilon:
            # greedy
            action = np.argmax(q_table[discrete_state])
        else:
            # random
            action = np.random.randint(0, env.action_space.n)
        state, reward, done, info, _ = env.step(action)
        discrete_state = get_discrete_state(state)
        score += reward
        # update q table
        if not done:
            #update Q-table
            max_next_q = np.max(q_table[discrete_state, action])
            current_q = q_table[discrete_state, action]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_next_q)
            q_table[discrete_state, action] = new_q
        elif state[0] >= env.goal_position:
            q_table[discrete_state, action] = 0
        else:
            state, info = env.reset()
            print("Done")
        # decaying every episode
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
