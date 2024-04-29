'''
Author: WANG CHENG
Date: 2024-04-15 23:30:56
LastEditTime: 2024-04-29 17:27:13
'''
import numpy as np
from memory_profiler import profile

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.current_idx = 0
        self.is_full = False
        self.buffer = np.empty((capacity,), dtype=[
            ('state', np.ndarray),
            ('action', np.int32),
            ('reward', np.float32),
            ('next_state', np.ndarray),
            ('terminal', bool)
        ])

    @profile
    def add_experience(self, state, action, reward, next_state, terminal):
        self.buffer[self.current_idx] = (state, action, reward, next_state, terminal)
        self.current_idx = (self.current_idx + 1) % self.capacity
        if not self.is_full and self.current_idx == 0:
            self.is_full = True

    def can_sample(self, batch_size):
        return self.capacity >= batch_size
    
    def sample(self, batch_size):
        if not self.is_full:
            batch_size = min(batch_size, self.current_idx)
        idxs = np.random.choice(self.current_idx, size=batch_size, replace=False)
        return self.buffer[idxs]

    def shuffle_buffer(self):
        np.random.shuffle(self.buffer[:self.current_idx])

    def __len__(self):
        return self.current_idx

    def __getitem__(self, idx):
        assert self.is_full or (idx < self.current_idx), "Index out of bounds"
        return {
            'state': self.buffer[idx]['state'],
            'next_state': self.buffer[idx]['next_state'],
            'action': self.buffer[idx]['action'],
            'reward': self.buffer[idx]['reward'],
            'terminal': self.buffer[idx]['terminal']
        }