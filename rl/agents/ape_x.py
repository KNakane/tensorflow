# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utility'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../network'))
import numpy as np
import tensorflow as tf
from agent import Agent
from eager_nn import Dueling_Net
from optimizer import *

class Ape_X(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_net(self):
        # ------------------ build eval_net ------------------
        with tf.variable_scope('eval_net'):
            self.q_eval = Dueling_Net(model=self.model, out_dim=self.n_actions, name='Q_net', opt=self._optimizer, lr=self.lr, trainable=True, is_categorical=self.is_categorical, is_noise=self.is_noise)

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            self.q_next = Dueling_Net(model=self.model, out_dim=self.n_actions, name='target_net', trainable=False, is_categorical=self.is_categorical, is_noise=self.is_noise)

        return

    def inference(self, state):
        if self.is_categorical:
            return tf.argmax(tf.reduce_sum(self.q_eval.inference(state) * self.z_list_broadcasted, axis=2), axis=1)
        else:
            return self.q_eval.inference(state)

class Learner():
    def __init__(self):
        pass

class Actor(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _build_net(self):
        # ------------------ build eval_net ------------------
        with tf.variable_scope('eval_net'):
            self.q_eval = Dueling_Net(model=self.model, out_dim=self.n_actions, name='Q_net', opt=self._optimizer, lr=self.lr, trainable=True, is_categorical=self.is_categorical, is_noise=self.is_noise)

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            self.q_next = Dueling_Net(model=self.model, out_dim=self.n_actions, name='target_net', trainable=False, is_categorical=self.is_categorical, is_noise=self.is_noise)

        return