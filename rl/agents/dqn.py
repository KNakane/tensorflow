# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utility'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../network'))
import numpy as np
import tensorflow as tf
from agent import Agent
from eager_nn import EagerNN, Dueling_Net
from optimizer import *

class DQN(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.is_categorical:
            self.Vmax, self.Vmin = 10, -10
            self.delta_z = (self.Vmax - self.Vmin) / (self.q_eval.N_atoms - 1)
            self.z_list = tf.constant([self.Vmin + i * self.delta_z for i in range(self.q_eval.N_atoms)],dtype=tf.float32)

        
    def _build_net(self):
        # ------------------ build eval_net ------------------
        with tf.variable_scope('eval_net'):
            self.q_eval = eval(self.network)(model=self.model, out_dim=self.n_actions, name='Q_net', opt=self._optimizer, lr=self.lr, trainable=True, is_categorical=self.is_categorical)

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            self.q_next = eval(self.network)(model=self.model, out_dim=self.n_actions, name='target_net', trainable=False, is_categorical=self.is_categorical)

    def inference(self, state):
        return self.q_eval.inference(state)

    def update_q_net(self, replay_data, weights):
        self.bs, ba, done, bs_, br, p_idx = replay_data
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = ba
        reward = br
        done = done

        # check to replace target parameters
        if self._iteration % self.replace_target_iter == 0:
            self.update_target_net()

        global_step = tf.train.get_or_create_global_step()

        with tf.GradientTape() as tape:
            if self.is_categorical:
                q_next, q_eval = self.q_next.inference(bs_), self.q_eval.inference(self.bs)
                next_action = tf.cast(tf.argmax(tf.reduce_sum(q_next * self.z_list, axis=2), axis=1), tf.int32)
                print(next_action.shape, q_next.shape)
                sys.exit()
                next_greedy_probs = tf.reduce_sum(next_action * q_next, axis=1)

                Tz = tf.clip_by_value(tf.reshape(reward,[-1, 1]) + (self.discount ** p_idx * self.z_list), self.Vmin, self.Vmax)
                b = (Tz - self.Vmin) / self.delta_z
                u, l = tf.ceil(b), tf.floor(b)
                eq = tf.cast(u == l, tf.float32)
                l -= eq
                lt0 = tf.cast(l < 0, tf.float32)
                u += lt0
                l += lt0

                ml = next_greedy_probs * (u - b)
                mu = next_greedy_probs * (b - l)

                m = np.zeros((self.batch_size, self.q_eval.N_atoms), dtype=np.float32)
                for i in range(self.q_eval.N_atoms):
                    m[batch_index, l[batch_index, i]] += ml[batch_index, i]
                    m[batch_index, u[batch_index, i]] += mu[batch_index, i]

                probs = tf.reduce_sum(q_eval * self.actions_list, axis=1)
                self.loss = tf.negative(tf.reduce_sum(m * tf.log(probs), axis=1))
                sys.exit()
            else:
                q_next, q_eval = self.q_next.inference(bs_), self.q_eval.inference(self.bs)
                q_target = np.array(q_eval).copy()
                q_target[batch_index, eval_act_index] = reward + self.discount ** p_idx * np.max(q_next, axis=1) * (1. - done)
                self.td_error = abs(q_target[batch_index, eval_act_index] - np.array(q_eval)[batch_index, eval_act_index])
                self.loss = tf.reduce_sum(tf.losses.huber_loss(labels=q_target, predictions=q_eval) * weights, keep_dims=True)
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
        if self.is_categorical:
            self.Vmax, self.Vmin = 10, -10
            self.delta_z = (self.Vmax - self.Vmin) / (self.q_eval.N_atoms - 1)
            self.z_list = tf.constant([self.Vmin + i * self.delta_z for i in range(self.q_eval.N_atoms)],dtype=tf.float32)


    def update_q_net(self, replay_data, weights):
        self.bs, ba, done, bs_, br, p_idx = replay_data
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = ba
        reward = br
        done = done

        # check to replace target parameters
        if self._iteration % self.replace_target_iter == 0:
            self.update_target_net()

        with tf.GradientTape() as tape:
            global_step = tf.train.get_or_create_global_step()
            if self.is_categorical:
                q_next, q_eval4next, q_eval = np.array(self.q_next.inference(bs_)), self.q_eval.inference(bs_), np.array(self.q_eval.inference(self.bs))
                q_target = np.array(q_eval).copy()
                next_action = tf.cast(tf.argmax(tf.reduce_sum(q_eval4next * self.z_list,axis=2), axis=1), tf.int32)
                next_greedy_probs = q_next[batch_index, next_action]
                reward = tf.cast(tf.expand_dims(reward, 1), tf.float32)
                done = tf.cast(tf.expand_dims(done, 1), tf.float32)
                p_idx = tf.cast(tf.expand_dims(p_idx, 1), tf.float32)
                
                Tz = tf.clip_by_value(reward + (self.discount ** p_idx * tf.expand_dims(self.z_list,0) * (1 - done)), self.Vmin, self.Vmax)
                b = (Tz - self.Vmin) / self.delta_z
                u, l = tf.ceil(b), tf.floor(b)
                eq = tf.cast(u == l, tf.float32)
                l -= eq
                lt0 = tf.cast(l < 0, tf.float32)
                u += lt0
                l += lt0

                ml = next_greedy_probs * (u - b)
                mu = next_greedy_probs * (b - l)

                l = np.array(l, dtype=np.int8)
                u = np.array(u, dtype=np.int8)

                m = np.zeros((self.batch_size, self.q_eval.N_atoms), dtype=np.float32)
                for i in range(self.q_eval.N_atoms):
                    m[batch_index, l[batch_index, i]] += ml[:, i]
                    m[batch_index, u[batch_index, i]] += mu[:, i]
                self.loss = - tf.reduce_sum(tf.log(q_eval[batch_index,eval_act_index]) * m, axis=1)
            else:
                q_next, q_eval4next, q_eval = np.array(self.q_next.inference(bs_)), self.q_eval.inference(bs_), self.q_eval.inference(self.bs)
                q_target = np.array(q_eval).copy()
                max_act4next = np.argmax(q_eval4next, axis=1)        # the action that brings the highest value is evaluated by q_eval
                selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
                q_target[batch_index, eval_act_index] = reward + self.discount ** p_idx * selected_q_next * (1. - done)
                self.td_error = abs(q_target[batch_index, eval_act_index] - np.array(q_eval)[batch_index, eval_act_index])
                self.loss = tf.reduce_sum(tf.losses.huber_loss(labels=q_target, predictions=q_eval) * weights, keep_dims=True)
        self.q_eval.optimize(self.loss, global_step, tape)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        self._iteration += 1

class Rainbow(DDQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)