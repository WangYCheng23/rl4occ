'''
Author: WANG CHENG
Date: 2024-04-15 23:30:56
LastEditTime: 2024-04-19 01:26:25
'''

from collections import namedtuple
import random
import numpy as np


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """保存一个transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """从memory中采样一个batch的transition"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
        