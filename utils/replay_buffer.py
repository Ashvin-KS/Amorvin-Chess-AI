import random
import numpy as np
import torch # Added for potential future tensor handling if needed

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add_experience(self, experience):
        """Adds a list of (board_state, move_probs, outcome) tuples from one game."""
        for exp in experience:
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = exp
            self.position = (self.position + 1) % self.capacity

    def sample_batch(self, batch_size):
        """Samples a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        # Unzip the batch into separate lists
        board_states, move_probs, outcomes = zip(*batch)
        return list(board_states), list(move_probs), list(outcomes)

    def __len__(self):
        return len(self.buffer)
