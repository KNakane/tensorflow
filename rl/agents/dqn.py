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
            self.Vmax, self.Vmin = 10.0, -10.0
            self.delta_z = (self.Vmax - self.Vmin) / (self.q_eval.N_atoms - 1)
            self.z_list = tf.constant([self.Vmin + i * self.delta_z for i in range(self.q_eval.N_atoms)],dtype=tf.float32)
            self.z_list_broadcasted = tf.tile(tf.reshape(self.z_list,[1,self.q_eval.N_atoms]), tf.constant([self.n_actions,1]))

        
    def _build_net(self):
        # ------------------ build eval_net ------------------
        with tf.variable_scope('eval_net'):
            self.q_eval = eval(self.network)(model=self.model, out_dim=self.n_actions, name='Q_net', opt=self._optimizer, lr=self.lr, trainable=True, is_categorical=self.is_categorical)

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            self.q_next = eval(self.network)(model=self.model, out_dim=self.n_actions, name='target_net', trainable=False, is_categorical=self.is_categorical)

    def inference(self, state):
        if self.is_categorical:
            return tf.argmax(tf.reduce_sum(self.q_eval.inference(state) * self.z_list_broadcasted, axis=2), axis=1)
        else:
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
                reward = tf.cast(reward, tf.float32)
                done = tf.cast(done, tf.float32)
                q_next = self.q_next.inference(bs_)           #target network Q'(s', a)
                q_eval = self.q_eval.inference(self.bs)       #main network   Q(s, a)
                q_next_value = tf.reduce_sum(q_next * self.z_list_broadcasted, axis=2)
                action_chosen_by_target_Q = tf.cast(tf.argmax(q_next_value, axis=1), tf.int32)
                Q_distributional_chosen_by_action_target = tf.gather_nd(q_next,
                                        tf.concat([tf.reshape(tf.range(self.batch_size), [-1, 1]),
                                                    tf.reshape(action_chosen_by_target_Q,[-1,1])], axis = 1))

                target = tf.tile(tf.reshape(reward,[-1, 1]), tf.constant([1, self.q_eval.N_atoms])) \
                    + tf.cast(tf.reshape((self.discount ** p_idx), [-1, 1]), tf.float32) * tf.multiply(tf.reshape(self.z_list,[1, self.q_eval.N_atoms]),
                      (1.0 - tf.tile(tf.reshape(done ,[-1, 1]), tf.constant([1, self.q_eval.N_atoms]))))

                target = tf.clip_by_value(target, self.Vmin, self.Vmax)
                b = (target - self.Vmin) / self.delta_z
                u, l = tf.ceil(b), tf.floor(b)
                u_id, l_id = tf.cast(u, tf.int32), tf.cast(l, tf.int32)
                u_minus_b, b_minus_l = u - b, b - l

                Q_distributional_chosen_by_action_online = tf.gather_nd(q_eval,
                                                                        list(enumerate(eval_act_index)))


                index_help = tf.tile(tf.reshape(tf.range(self.batch_size),[-1, 1]), tf.constant([1, self.q_eval.N_atoms]))
                index_help = tf.expand_dims(index_help, -1)
                u_id = tf.concat([index_help, tf.expand_dims(u_id, -1)], axis=2)
                l_id = tf.concat([index_help, tf.expand_dims(l_id, -1)], axis=2)
                error = Q_distributional_chosen_by_action_target * u_minus_b * \
                        tf.log(tf.gather_nd(Q_distributional_chosen_by_action_online, l_id)) \
                    + Q_distributional_chosen_by_action_target * b_minus_l * \
                        tf.log(tf.gather_nd(Q_distributional_chosen_by_action_online, u_id))
                self.td_error = tf.negative(tf.reduce_sum(error, axis=1))
                self.loss = tf.reduce_sum(self.td_error * weights)
            else:
                q_next, q_eval = self.q_next.inference(bs_), self.q_eval.inference(self.bs)
                q_target = np.array(q_eval).copy()
                q_target[batch_index, eval_act_index] = reward + self.discount ** p_idx * np.max(q_next, axis=1) * (1. - done)
                self.td_error = abs(q_target[batch_index, eval_act_index] - np.array(q_eval)[batch_index, eval_act_index])
                self.loss = tf.reduce_sum(tf.losses.huber_loss(labels=q_target, predictions=q_eval) * weights)
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

    def update_q_net(self, replay_data, weights):
        self.bs, eval_act_index, done, bs_, reward, p_idx = replay_data
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # check to replace target parameters
        if self._iteration % self.replace_target_iter == 0:
            self.update_target_net()

        global_step = tf.train.get_or_create_global_step()

        with tf.GradientTape() as tape:
            if self.is_categorical:
                reward = tf.cast(reward, tf.float32)
                done = tf.cast(done, tf.float32)
                q_next = np.array(self.q_next.inference(bs_)) #target network Q'(s', a)
                q_eval4next = self.q_eval.inference(bs_)      #main network   Q(s', a)
                q_eval = self.q_eval.inference(self.bs)       #main network   Q(s, a)
                q_ = tf.reduce_sum(tf.multiply(q_eval4next, self.z_list), axis=2) # a = argmax(Q(s',a))
                next_action = tf.cast(tf.argmax(q_, axis=1), tf.int32)
                Q_distributional_chosen_by_action_target = tf.gather_nd(q_next,
                                                                        list(enumerate(next_action)))

                target = tf.tile(tf.reshape(reward,[-1, 1]), tf.constant([1, self.q_eval.N_atoms])) \
                    + tf.cast(tf.reshape((self.discount ** p_idx), [-1, 1]), tf.float32) * tf.multiply(tf.reshape(self.z_list,[1, self.q_eval.N_atoms]),
                      (1.0 - tf.tile(tf.reshape(done ,[-1, 1]), tf.constant([1, self.q_eval.N_atoms]))))

                target = tf.clip_by_value(target, self.Vmin, self.Vmax)
                b = (target - self.Vmin) / self.delta_z
                u, l = tf.ceil(b), tf.floor(b)
                u_id, l_id = tf.cast(u, tf.int32), tf.cast(l, tf.int32)
                u_minus_b, b_minus_l = u - b, b - l

                Q_distributional_chosen_by_action_online = tf.gather_nd(q_eval,
                                                                        list(enumerate(eval_act_index)))


                index_help = tf.tile(tf.reshape(tf.range(self.batch_size),[-1, 1]), tf.constant([1, self.q_eval.N_atoms]))
                index_help = tf.expand_dims(index_help, -1)
                u_id = tf.concat([index_help, tf.expand_dims(u_id, -1)], axis=2)
                l_id = tf.concat([index_help, tf.expand_dims(l_id, -1)], axis=2)
                error = Q_distributional_chosen_by_action_target * u_minus_b * \
                        tf.log(tf.gather_nd(Q_distributional_chosen_by_action_online, l_id)) \
                    + Q_distributional_chosen_by_action_target * b_minus_l * \
                        tf.log(tf.gather_nd(Q_distributional_chosen_by_action_online, u_id))
                self.td_error = tf.negative(tf.reduce_sum(error, axis=1))
                self.loss = tf.reduce_sum(self.td_error * weights)
            else:
                q_next, q_eval4next, q_eval = np.array(self.q_next.inference(bs_)), self.q_eval.inference(bs_), self.q_eval.inference(self.bs)
                q_target = np.array(q_eval).copy()
                max_act4next = np.argmax(q_eval4next, axis=1)        # the action that brings the highest value is evaluated by q_eval
                selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
                q_target[batch_index, eval_act_index] = reward + self.discount ** p_idx * selected_q_next * (1. - done)
                self.td_error = abs(q_target[batch_index, eval_act_index] - np.array(q_eval)[batch_index, eval_act_index])
                self.loss = tf.reduce_sum(tf.losses.huber_loss(labels=q_target, predictions=q_eval) * weights)
        self.q_eval.optimize(self.loss, global_step, tape)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        self._iteration += 1

class Rainbow(DDQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)