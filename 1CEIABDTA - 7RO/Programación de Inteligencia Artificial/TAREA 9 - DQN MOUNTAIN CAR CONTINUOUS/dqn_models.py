# Borrowed from 
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple


# Transition - a named tuple representing a single transition in our environment. 
# It essentially maps (state, action) pairs to their (next_state, reward) result, 
# with the state being the screen difference image as described later on.
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(h * w, 128)  # Adjust input size to match h * w
        self.fc2 = nn.Linear(128, 64)
        self.head = nn.Linear(64, outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(x.size(0), -1)))  # Flatten the input
        x = F.relu(self.fc2(x))
        return self.head(x)