# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utility'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../network'))
import numpy as np
import tensorflow as tf
from agent import Agent
from eager_cnn import EagerCNN, Dueling_Net
from optimizer import *
from writer import Writer

class DQN(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _build_net(self):
        # ------------------ build eval_net ------------------
        with tf.variable_scope('eval_net'):
            #self.q_eval = EagerCNN(model=self.model, name='Q_net', opt=self._optimizer, lr=self.lr, trainable=True)
            self.q_eval = Dueling_Net(model=self.model, out_dim=self.n_actions, name='Q_net', opt=self._optimizer, lr=self.lr, trainable=True)

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            #self.q_next = EagerCNN(model=self.model, name='target_net', trainable=False)
            self.q_next = Dueling_Net(model=self.model, out_dim=self.n_actions, name='target_net', trainable=False)

    def inference(self, state):
        return self.q_eval.inference(state)

    def update_q_net(self, replay_data):
        bs, ba, done, bs_, br = replay_data
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = ba
        reward = br
        done = done

        # check to replace target parameters
        if self._iteration % self.replace_target_iter == 0:
            self.update_target_net()

        global_step = tf.train.get_or_create_global_step()

        with tf.GradientTape() as tape:
            q_next, q_eval = self.q_next.inference(bs_), self.q_eval.inference(bs)
            q_target = np.array(q_eval).copy()
            q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1) * (1. - done)
            self.td_error = abs(q_target[batch_index, eval_act_index] - np.array(q_eval)[batch_index, eval_act_index])
            self.loss = tf.losses.huber_loss(labels=q_target, predictions=q_eval)
        self.q_eval.optimize(self.loss, global_step, tape)
        
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        self._iteration += 1

    def update_target_net(self):
        for param, target_param in zip(self.q_eval.weights, self.q_next.weights):
            target_param.assign(param)
        return



class DDQN(DQN):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_q_net(self, replay_data):
        bs, ba, done, bs_, br = replay_data
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = ba
        reward = br
        done = done

        # check to replace target parameters
        if self._iteration % self.replace_target_iter == 0:
            self.update_target_net()

        with tf.GradientTape() as tape:
            global_step = tf.train.get_or_create_global_step()
            q_next, q_eval4next, q_eval = np.array(self.q_next.inference(bs_)), self.q_eval.inference(bs_), self.q_eval.inference(bs)
            q_target = np.array(q_eval).copy()
            max_act4next = np.argmax(q_eval4next, axis=1)        # the action that brings the highest value is evaluated by q_eval
            selected_q_next = q_next[batch_index, max_act4next] # Double DQN, select q_next depending on above actions
            q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next * (1. - done)
            self.td_error = abs(q_target[batch_index, eval_act_index] - np.array(q_eval)[batch_index, eval_act_index])
            self.loss = tf.losses.huber_loss(labels=q_target, predictions=q_eval)
        self.q_eval.optimize(self.loss, global_step, tape)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        self._iteration += 1