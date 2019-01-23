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
            optimizer=None,
            policy=False # True : On-Policy   False : Off-policy
    ):
        # Eager Mode
        tf.enable_eager_execution()
        self.model = model
        self.on_policy = policy
        self.n_actions = n_actions
        self.actions_list = list(range(n_actions))
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
    
    def inference(self, s):
        raise Exception('please set Optimizer')

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.inference(observation)
            if self.on_policy:
                action = np.random.choice(self.actions_list, size=1, p=actions_value)[0]
            else:
                action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def update_q_net(self, replay_data):
        raise Exception('please set update_q_net')