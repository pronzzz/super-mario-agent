"""
Prioritized Experience Replay Buffer with SumTree for efficient sampling.
"""

import numpy as np
import random
from collections import namedtuple


Transition = namedtuple('Transition', 
    ('frames', 'action_history', 'action', 'reward', 'next_frames', 
     'next_action_history', 'done'))


class SumTree:
    """
    Binary tree data structure for efficient prioritized sampling.
    Each leaf stores a priority, and each node stores the sum of its children.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        """Update parent nodes after priority change"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Find the leaf index for a given sum value"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Return total priority sum"""
        return self.tree[0]

    def add(self, priority, data):
        """Add new data with priority"""
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)

        self.write_idx += 1
        if self.write_idx >= self.capacity:
            self.write_idx = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        """Update priority of a leaf node"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        """Get data and leaf index for a sum value"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer using SumTree.
    
    Samples transitions based on TD error, allowing the agent to learn
    more from surprising/important experiences.
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Args:
            capacity: Maximum number of transitions to store
            alpha: How much prioritization to use (0 = uniform, 1 = full priority)
            beta_start: Initial importance sampling weight
            beta_frames: Number of frames to anneal beta to 1.0
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        # Small constant to prevent zero priorities
        self.epsilon = 0.01
        # Maximum priority for new samples
        self.max_priority = 1.0

    def _get_beta(self):
        """Anneal beta from beta_start to 1.0"""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, *args):
        """
        Store a transition with maximum priority.
        Args: frames, action_history, action, reward, next_frames, next_action_history, done
        """
        transition = Transition(*args)
        # New transitions get max priority
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size):
        """
        Sample a batch of transitions based on priorities.
        
        Returns:
            batch: List of Transition objects
            indices: Tree indices for updating priorities
            weights: Importance sampling weights
        """
        batch = []
        indices = []
        priorities = []
        
        # Divide priority range into batch_size segments
        segment = self.tree.total() / batch_size
        
        beta = self._get_beta()
        self.frame += 1
        
        for i in range(batch_size):
            # Sample uniformly from each segment
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)
        
        # Calculate importance sampling weights
        # w = (N * P(i))^(-beta) / max(w)
        priorities = np.array(priorities)
        sampling_probabilities = priorities / self.tree.total()
        
        weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        weights /= weights.max()  # Normalize
        
        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Tree indices from sampling
            priorities: New priorities (typically TD errors)
        """
        for idx, priority in zip(indices, priorities):
            # Convert priority to alpha-weighted priority
            priority = (abs(priority) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries
