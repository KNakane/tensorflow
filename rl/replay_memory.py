# -*- coding: utf-8 -*-
import random
import numpy as np
import tensorflow as tf
from sum_tree import SumTree
from collections import namedtuple

class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0
        self.Transition = namedtuple('Transition',('states', 'actions', 'dones', 'next_states', 'rewards', 'p_index'))

    def __len__(self):
        return len(self.memory)

    def push(self, state, action, done, next_state, reward, p_index):
        #data = [state, action, done, next_state, reward]
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = self.Transition(state, action, done, next_state, reward, p_index) #data
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size, _=None):
        return (None, random.sample(self.memory, batch_size), 1)

    def update(self, idx, td_error):
        pass


class PrioritizeReplayBuffer(ReplayBuffer):
    # Based on https://github.com/y-kamiya/machine-learning-samples/blob/7b6792ce37cc69051e9053afeddc6d485ad34e79/python3/reinforcement/dqn/agent.py
    EPSILON = 0.0001
    ALPHA = 0.6
    BETA = 0.4
    size = 0

    def __init__(self, capacity):
        super().__init__(capacity=capacity)
        self.td_error_epsilon = 0.0001
        self.tree = SumTree(capacity)

    def __len__(self):
        return self.size
    
    def _getPriority(self, td_error):
        return (td_error + self.EPSILON) ** self.ALPHA

    def push(self, state, action, done, next_state, reward, p_index):
        self.size += 1
        transition = self.Transition(state, action, done, next_state, reward, p_index)
        priority = self.tree.max()
        if priority <= 0:
            priority = 1
        self.tree.add(priority, transition)

    def sample(self, batch_size, episode):
        list = []
        indexes = []
        weights = np.empty(batch_size, dtype='float32')
        total = self.tree.total()
        beta = self.BETA + (1 - self.BETA) * episode #episode / self.config.num_episodes

        for i, rand in enumerate(np.random.uniform(0, total, batch_size)):
            (idx, priority, data) = self.tree.get(rand)
            list.append(data)
            indexes.append(idx)
            weights[i] = (self.capacity * priority / total) ** (-beta)

        return (indexes, list, weights / weights.max())

    def update(self, idx, td_error):
        priority = self._getPriority(td_error)
        self.tree.update(idx, priority)