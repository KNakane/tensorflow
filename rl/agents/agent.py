# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from writer import Writer

class Agent():
    def __init__(
            self,
            model,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            batch_size=32,
            e_greedy_increment=None,
            optimizer=None
    ):
        self.model = model
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.batch_size = batch_size
        self._optimizer = optimizer
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self._iteration = 0

        # consist of [target_net, evaluate_net]
        self._build_net()
        
    def _build_net(self):
        if hasattr(self.__class__.__name__, 'q_eval') == False:
            raise Exception('please set build net')
        if self._optimizer is None:
            raise Exception('please set Optimizer')

    def choose_action(self, observation):
        raise Exception('please set choose_action')

    def update_q_net(self, replay_data):
        raise Exception('please set update_q_net')