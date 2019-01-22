# -*- coding: utf-8 -*-
import random
import numpy as np
import tensorflow as tf
from collections import namedtuple

class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0
        self.Transition = namedtuple('Transition',('states', 'actions', 'dones', 'next_states', 'rewards'))

    def __len__(self):
        return len(self.memory)

    def push(self, state, action, done, next_state, reward):
        #data = [state, action, done, next_state, reward]
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = self.Transition(state, action, done, next_state, reward) #data
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        """
        idxes = [random.randint(0, len(self.memory) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
        """


class PrioritizeReplayBuffer(ReplayBuffer):
    def __init__(self, capacity):
        super().__init__(capacity=capacity)
        self.td_error_epsilon = 0.0001

    def push(self, state, action, done, next_state, reward, td_error):
        data = [state, action, done, next_state, reward, td_error]
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = data
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) < self.capacity:
            sum_absolute_td_error = np.sum(np.absolute(self.memory[:][5])) + self.td_error_epsilon * len(self.memory)
            rand_list = np.sort(np.random.uniform(0, sum_absolute_td_error, batch_size))
            indexes = []
            idx = 0
            tmp_sum_absolute_td_error = 0
            for rand_num in rand_list:
                tmp_sum_absolute_td_error += abs(self.memory[idx][5]) + self.td_error_epsilon
                idx += 1
                if idx >= len(self.memory):
                    idx = len(self.memory) -1
                indexes.append(idx)
            return self.memory[np.arange(indexes, dtype=np.int32)]
        else:
            return random.sample(self.memory, batch_size)