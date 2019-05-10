# Initialize packages
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent


# Load environment and print its specifics
env = gym.make('CartPole-v1')
env.seed(0)
print('State shape: ', env.observation_space)
print('Number of actions: ', env.action_space)

# Define the agent
agent = Agent(state_size=4, action_size=2, seed=0)

# Define the learning function; solution attempt via ordinary Deep Q-learning
def dqn(n_episodes=20000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.95):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
        
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    
    # main loop over the episodes
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        
        # internal loop over the steps the agent can take within an episode
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            #print("State etc.:", next_state, reward, done, _)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        
        # Printing results and observing and quitting when the environment is solved (>=195.00); saving the weights at the end
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=195.0:
            print('\n With the given parameters the environment is solved in {:d} episodes. \tThe precise average score is {:.2f} there.'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

scores = dqn()

# Define moving average functions for plotting.
def movingaverage(values,window):
    weights = np.repeat(1.0,window)/window
    smas = np.convolve(values,weights,'valid')
    return smas

def ExpMovingAverage(values,window):
    weights = np.exp(np.linspace(-1.,0.,window))
    weights /= weights.sum()
    
    emas = np.convolve(values,weights)[:len(values)]
    emas[:window] = emas[window]
    return emas

# We plot the scores over the number of episodes.
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(scores, 'b', lw=1)
plt.plot(movingaverage(scores,5), 'g', lw=2)
plt.plot(ExpMovingAverage(scores,5), 'y', lw=2)

plt.plot([0, len(scores)], [195, 195], 'r', lw=2)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

# Load the weights from file and display the training result of the agent
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

for i in range(3):
    state = env.reset()
    img = plt.imshow(env.render(mode='rgb_array'))
    for j in range(200):
        action = agent.act(state)
        img.set_data(env.render(mode='rgb_array')) 
        plt.axis('off')
        
        state, reward, done, _ = env.step(action)
        if done:
            break 
    
# Close the environment        
env.close()
