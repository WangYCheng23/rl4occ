'''
Author: WANG CHENG
Date: 2024-04-15 23:30:56
LastEditTime: 2024-04-20 01:29:42
'''
import threading
import numpy as np


class ReplayBuffer:
    """
    A buffer to store experiences and replay them during training.

    Args:
        capacity (int): The maximum number of experiences that can be stored in the buffer.
        state_dim (int): The dimensionality of the state space.
        action_dim (int): The dimensionality of the action space.
        max_len_state (int): The maximum length of a state vector.
        max_len_next_state (int): The maximum length of a next state vector.
    """

    def __init__(self, capacity):
        self.capacity = capacity

        # Initialize the buffer with empty lists for each component
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.terminals = []

        self.current_idx = 0
        self._lock = threading.Lock()  # random lock for thread safety

    def shuffle_buffer(self):
        """
        Shuffle the buffer randomly.
        """
        with self._lock:
            permutation = np.random.permutation(len(self.states))
            self.states = [self.states[i] for i in permutation]
            self.next_states = [self.next_states[i] for i in permutation]
            self.actions = [self.actions[i] for i in permutation]
            self.rewards = [self.rewards[i] for i in permutation]
            self.terminals = [self.terminals[i] for i in permutation]
    
    def add_experience(self, state, action, reward, next_state, terminal):
        """
        Add a new experience to the buffer.

        Args:
            state (numpy.ndarray): The current state.
            action (numpy.ndarray): The action taken in the current state.
            reward (float): The reward received for taking the action.
            next_state (numpy.ndarray): The next state after taking the action.
            terminal (bool): Whether the episode has terminated.
        """
        with self._lock:  # use a lock to ensure thread safety
            if len(self.states) >= self.capacity:
                self.remove_oldest_experience()

            # Add the new experience to the buffer
            self.states.append(state)
            self.next_states.append(next_state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)

            self.current_idx = (self.current_idx + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            A dictionary containing the sampled experiences.
                Each key-value pair corresponds to one experience.
                The keys are 'state', 'next_state', 'action', 'reward', and 'terminal'.
        """
        idxs = np.random.choice(len(self.states), size=batch_size)
        batch = {
            'states': [self.states[i] for i in idxs],
            'next_states': [self.next_states[i] for i in idxs],
            'actions': [self.actions[i] for i in idxs],
            'rewards': [self.rewards[i] for i in idxs],
            'terminals': [self.terminals[i] for i in idxs]
        }
        return batch

    def remove_oldest_experience(self):
        """
        Remove the oldest experience from the buffer.
        """
        if len(self.states) > 0:
            self.states.pop(0)
            self.next_states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.terminals.pop(0)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        """
        Get an experience from the buffer.

        Args:
            idx (int): The index of the experience to get.

        Returns:
            A dictionary containing the experience.
                Each key-value pair corresponds to one experience.
                The keys are 'state', 'next_state', 'action', 'reward', and 'terminal'.
        """
        return {
            'state': self.states[idx],
            'next_state': self.next_states[idx],
            'action': self.actions[idx],
            'reward': self.rewards[idx],
            'terminal': self.terminals[idx]
        }
