#!/usr/bin/env python
# coding: utf-8

# In[4]:


# %load cartpole_qtable.py

import gym
import json
import math
import numpy as np

from plot import draw

#====== code segment starts ======

# Training episode setting
N_EPISODES = 400
EPISODE_LENGTH = 250

#====== code segment ends ======

def save_qtable(table, buckets):
    data = {
        "qtable": table.tolist(), 
        "buckets" : buckets
    }

    with open('cartpole_qtable.json', 'w', encoding = 'utf-8') as f:
        json.dump(data, f, ensure_ascii = False, indent = 4)


def choose_action(state, q_table, action_space, epsilon):

    #====== code segment starts ======

    if np.random.random_sample() < epsilon: # random action if random sample is less than epsilon
        return action_space.sample() 
    else:                               # greedy action based on Q table, return argmax of the given state from q table
        return np.argmax(q_table[state])

    #====== code segment ends ======

def get_state(observation, n_buckets, state_bounds):
    state = [0] * len(observation)
    for i, s in enumerate(observation):
        l, u = state_bounds[i][0], state_bounds[i][1] # lower- and upper-bounds for each feature in observation
        if s <= l:
            state[i] = 0
        elif s >= u:
            state[i] = n_buckets[i] - 1
        else:
            state[i] = int(((s - l) / (u - l)) * n_buckets[i])

    return tuple(state)

# Environment setting
env = gym.make('CartPole-v0')

rewards_record = []


# Preparing Q table

#====== code segment starts ======

## buckets for continuous state values to be assigned to
## Observation space: [position of cart, velocity of cart, angle of pole, rotation rate of pole]
## Setting bucket size to 1 = ignoring the particular observation state 
## Take example: if we set the bucket of the angle of pole as 6, then it means we divide the angles into 6 
## partition, it means that degree 0 ~ degree 30 corresponds to same interger value(i.e 0), and dgree 30. 
## ~ dgree 60 corresponds to another identical value(i.e 1) and so on.
## You can try different kind of partition combination to help you train your model

n_buckets = (1, 1, 6, 4)

#====== code segment ends ======


## discrete actions
n_actions = env.action_space.n

## state bounds
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = [-0.5, 0.5]
state_bounds[3] = [-math.radians(50), math.radians(50)]


#====== code segment starts ======

## init Q table for each state-action pair
q_table = np.zeros(n_buckets + (n_actions,))

#====== code segment ends ======

# Learning related constants; factors determined by trial-and-error
get_epsilon = lambda i: max(0.01, min(1, 1.0 - math.log10((i + 1) / 25))) # epsilon-greedy, factor to explore randomly; discounted over time
get_lr = lambda i: max(0.01, min(0.5, 1.0 - math.log10((i + 1) / 25))) # learning rate; discounted over time
gamma = 0.99 # reward discount factor

# Q-learning
rewards_record = []     

for i_episode in range(N_EPISODES):
    epsilon = get_epsilon(i_episode)
    lr = get_lr(i_episode)

    #====== code segment starts ======

    observation = env.reset()                               # reset environment to initial state for each episode(DO NOT NEED TO MODIFY)
    rewards = 0                                             # initialize rewards for each episode
    state = get_state(observation, n_buckets, state_bounds)                        # turn observation into discrete state

    #====== code segment ends ======

    for t in range(EPISODE_LENGTH):
        env.render()

        #====== code segment starts ======

        # Agent takes action
        action = choose_action(state, q_table, env.action_space, epsilon)      # choose an action based on q_table 
        observation, reward, done, info = env.step(action) # do the action, get the reward
        rewards += reward                                  # accumulate reward
        next_state = get_state(observation, n_buckets, state_bounds)           # turn observation into discrete state

        # Agent learns via Q-learning
        q_next_max = np.amax(q_table[next_state])                   # find maximum next state action value from q table 
        q_table[state + (action,)] += lr * (reward + gamma * q_next_max - q_table[state + (action,)])   # update current state-action pair table value by q-learning update equation

        # Transition to next state
        state = next_state

        #====== code segment ends ======

        if done:
            print('Episode {} finished after {} timesteps, total rewards {}'.format(i_episode, t + 1, rewards))
            break

    rewards_record.append(rewards)

draw(rewards_record, "Q table action reward")

save_qtable(q_table, n_buckets + (n_actions,))

env.close() # need to close, or errors will be reported


# In[ ]:




