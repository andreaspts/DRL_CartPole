'''
Created on 19.04.2019

@author: Andreas
'''

# Initialize the packages
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the fcc network
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Parameters:
        ==========
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Defining the layers eith 2 outputs
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        # Output layer, 2 units - one for each action of q(state_fixed, action)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        state = self.fc1(state)
        state = F.relu(state)
        state = self.fc2(state)
        state = F.relu(state)
        state = self.fc3(state)
        state = F.relu(state)
        state = self.fc4(state)
        
        return state
