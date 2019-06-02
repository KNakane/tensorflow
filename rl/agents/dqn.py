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
            self.q_eval = eval(self.network)(model=self.model, out_dim=self.n_actions, name='Q_net', opt=self._optimizer, lr=self.lr, trainable=self.trainable, is_categorical=self.is_categorical, is_noise=self.is_noise)

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            self.q_next = eval(self.network)(model=self.model, out_dim=self.n_actions, name='target_net', trainable=False, is_categorical=self.is_categorical, is_noise=self.is_noise)

    def inference(self, state):
        if self.is_categorical:
            return tf.argmax(tf.reduce_sum(self.q_eval.inference(state) * self.z_list_broadcasted, axis=2), axis=1)
        else:
            return self.q_eval.inference(state)


    def update_q_net(self, replay_data, weights):
        self.bs, eval_act_index, done, bs_, reward, p_idx = replay_data
        eval_act_index = np.reshape(np.array(eval_act_index, dtype=np.int32),(self.batch_size,1))
        reward = np.array(reward, dtype=np.float32)
        done = np.array(done, dtype=np.float32)

        loss, td_error = self._train_body(self.bs, eval_act_index, done, bs_, reward, p_idx, weights)

        # check to replace target parameters
        if self._iteration % self.replace_target_iter == 0:
            self.update_target_net()

        # decreasing epsilon
        self.epsilon = max(self.epsilon - self.epsilon_decrement, self.epsilon_min)

        self._iteration += 1

        return loss, td_error

    @tf.contrib.eager.defun
    def _train_body(self, bs, eval_act_index, done, bs_, reward, p_idx, weights):

        global_step = tf.train.get_or_create_global_step()

        with tf.device(self.device):
            with tf.GradientTape() as tape:
                if self.is_categorical:
                    # based on https://github.com/cmusjtuliuyuan/RainBow/blob/master/agent.py
                    q_next = self.q_next.inference(bs_)           #target network Q'(s', a)
                    q_eval = self.q_eval.inference(bs)       #main network   Q(s, a)
                    q_next_value = tf.reduce_sum(q_next * self.z_list_broadcasted, axis=2)
                    action_chosen_by_target_Q = tf.cast(tf.argmax(q_next_value, axis=1), tf.int32)
                    Q_distributional_chosen_by_action_target = tf.gather_nd(q_next,
                                            tf.concat([tf.reshape(tf.range(self.batch_size), [-1, 1]),
                                                        tf.reshape(action_chosen_by_target_Q,[-1,1])], axis = 1))

                    target = tf.tile(tf.reshape(reward,[-1, 1]), tf.constant([1, self.q_eval.N_atoms])) \
                        + tf.cast(tf.reshape((self.discount ** tf.cast(p_idx, tf.float32)), [-1, 1]), tf.float32) * tf.multiply(tf.reshape(self.z_list,[1, self.q_eval.N_atoms]),
                        (1.0 - tf.tile(tf.reshape(done ,[-1, 1]), tf.constant([1, self.q_eval.N_atoms]))))

                    target = tf.clip_by_value(target, self.Vmin, self.Vmax)
                    b = (target - self.Vmin) / self.delta_z
                    u, l = tf.ceil(b), tf.floor(b)
                    u_id, l_id = tf.cast(u, tf.int32), tf.cast(l, tf.int32)
                    u_minus_b, b_minus_l = u - b, b - l

                    action_list = tf.concat([tf.expand_dims(tf.range(self.batch_size), axis=1), eval_act_index], axis=1)

                    Q_distributional_chosen_by_action_online = tf.gather_nd(q_eval, action_list)


                    index_help = tf.tile(tf.reshape(tf.range(self.batch_size),[-1, 1]), tf.constant([1, self.q_eval.N_atoms]))
                    index_help = tf.expand_dims(index_help, -1)
                    u_id = tf.concat([index_help, tf.expand_dims(u_id, -1)], axis=2)
                    l_id = tf.concat([index_help, tf.expand_dims(l_id, -1)], axis=2)
                    error = Q_distributional_chosen_by_action_target * u_minus_b * \
                            tf.log(tf.gather_nd(Q_distributional_chosen_by_action_online, l_id)) \
                        + Q_distributional_chosen_by_action_target * b_minus_l * \
                            tf.log(tf.gather_nd(Q_distributional_chosen_by_action_online, u_id))
                    error = tf.reduce_mean(error, axis=1)
                    td_error = tf.abs(error)
                    loss = tf.reduce_mean(tf.negative(error) * weights)
                else:
                    q_next, q_eval = self.q_next.inference(bs_), self.q_eval.inference(bs)
                    q_target = reward + self.discount ** tf.cast(p_idx, tf.float32) * tf.reduce_max(q_next, axis = 1) * (1. - done)
                    q_target = tf.stop_gradient(q_target)
                    action_list = tf.concat([tf.expand_dims(tf.range(self.batch_size), axis=1), eval_act_index], axis=1)
                    q_eval = tf.gather_nd(q_eval, action_list)
                    td_error = tf.abs(q_target - q_eval)
                    loss = tf.reduce_mean(tf.losses.huber_loss(labels=q_target, predictions=q_eval) * weights)
            self.q_eval.optimize(loss, global_step, tape)

        return loss, td_error

    def update_target_net(self):
        for param, target_param in zip(self.q_eval.weights, self.q_next.weights):
            target_param.assign(param)
        return



class DDQN(DQN):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_q_net(self, replay_data, weights):
        self.bs, eval_act_index, done, bs_, reward, p_idx = replay_data
        eval_act_index = np.reshape(np.array(eval_act_index, dtype=np.int32),(self.batch_size,1))
        reward = np.array(reward, dtype=np.float32)
        done = np.array(done, dtype=np.float32)

        loss, td_error = self._train_body(self.bs, eval_act_index, done, bs_, reward, p_idx, weights)

        # check to replace target parameters
        if self._iteration % self.replace_target_iter == 0:
            self.update_target_net()

        # decreasing epsilon
        self.epsilon = max(self.epsilon - self.epsilon_decrement, self.epsilon_min)

        self._iteration += 1

        return loss, td_error

        

    @tf.contrib.eager.defun
    def _train_body(self, bs, eval_act_index, done, bs_, reward, p_idx, weights):
        with tf.device(self.device):
            global_step = tf.train.get_or_create_global_step()
            with tf.GradientTape() as tape:
                if self.is_categorical:
                    q_next = self.q_next.inference(bs_) #target network Q'(s', a)
                    q_eval4next = self.q_eval.inference(bs_)      #main network   Q(s', a)
                    q_eval = self.q_eval.inference(bs)       #main network   Q(s, a)
                    q_ = tf.reduce_sum(tf.multiply(q_eval4next, self.z_list), axis=2) # a = argmax(Q(s',a))
                    next_action = tf.cast(tf.argmax(q_, axis=1), tf.int32)
                    indices = tf.concat(values=[tf.expand_dims(tf.range(self.batch_size), axis=1), tf.expand_dims(next_action, axis=1)], axis=1)
                    Q_distributional_chosen_by_action_target = tf.gather_nd(q_next, indices)

                    target = tf.tile(tf.reshape(reward,[-1, 1]), tf.constant([1, self.q_eval.N_atoms])) \
                        + tf.cast(tf.reshape((self.discount ** tf.cast(p_idx, tf.float32)), [-1, 1]), tf.float32) * tf.multiply(tf.reshape(self.z_list,[1, self.q_eval.N_atoms]),
                        (1.0 - tf.tile(tf.reshape(done ,[-1, 1]), tf.constant([1, self.q_eval.N_atoms]))))

                    target = tf.clip_by_value(target, self.Vmin, self.Vmax)
                    b = (target - self.Vmin) / self.delta_z
                    u, l = tf.ceil(b), tf.floor(b)
                    u_id, l_id = tf.cast(u, tf.int32), tf.cast(l, tf.int32)
                    u_minus_b, b_minus_l = u - b, b - l

                    action_list = tf.concat([tf.expand_dims(tf.range(self.batch_size), axis=1), eval_act_index], axis=1)
                    Q_distributional_chosen_by_action_online = tf.gather_nd(q_eval,action_list)


                    index_help = tf.tile(tf.reshape(tf.range(self.batch_size),[-1, 1]), tf.constant([1, self.q_eval.N_atoms]))
                    index_help = tf.expand_dims(index_help, -1)
                    u_id = tf.concat([index_help, tf.expand_dims(u_id, -1)], axis=2)
                    l_id = tf.concat([index_help, tf.expand_dims(l_id, -1)], axis=2)
                    error = Q_distributional_chosen_by_action_target * u_minus_b * \
                            tf.log(tf.gather_nd(Q_distributional_chosen_by_action_online, l_id)) \
                        + Q_distributional_chosen_by_action_target * b_minus_l * \
                            tf.log(tf.gather_nd(Q_distributional_chosen_by_action_online, u_id))
                    error = tf.reduce_mean(error, axis=1)
                    td_error = tf.abs(error)
                    loss = tf.reduce_mean(tf.negative(error) * weights)
                else:
                    q_next, q_eval = self.q_next.inference(bs_), self.q_eval.inference(bs)
                    q_eval4next = tf.argmax(self.q_eval.inference(bs_), axis=1, output_type=tf.int32)
                    indices = tf.concat([tf.expand_dims(tf.range(self.batch_size), axis=1), tf.expand_dims(q_eval4next, axis=1)], axis=1)
                    q_target = tf.gather_nd(q_next, indices)
                    q_target = reward + self.discount ** tf.cast(p_idx, tf.float32) * q_target * (1. - done)
                    q_target = tf.stop_gradient(q_target)
                    action_list = tf.concat([tf.expand_dims(tf.range(self.batch_size), axis=1), eval_act_index], axis=1)
                    q_eval = tf.gather_nd(q_eval, action_list)
                    td_error = tf.abs(q_target - q_eval)
                    loss = tf.reduce_mean(tf.losses.huber_loss(labels=q_target, predictions=q_eval) * weights)
            self.q_eval.optimize(loss, global_step, tape)
            
        return loss, td_error

class Rainbow(DDQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = self.epsilon_max

    def update_q_net(self, replay_data, weights):
        self.bs, eval_act_index, done, bs_, reward, p_idx = replay_data
        eval_act_index = np.reshape(np.array(eval_act_index, dtype=np.int32),(self.batch_size,1))
        reward = np.array(reward, dtype=np.float32)
        done = np.array(done, dtype=np.float32)

        loss, td_error = self._train_body(self.bs, eval_act_index, done, bs_, reward, p_idx, weights)

        # check to replace target parameters
        if self._iteration % self.replace_target_iter == 0:
            self.update_target_net()

        self._iteration += 1

        return loss, td_error

    
    @tf.contrib.eager.defun
    def _train_body(self, bs, eval_act_index, done, bs_, reward, p_idx, weights):
        with tf.device(self.device):
            global_step = tf.train.get_or_create_global_step()
            with tf.GradientTape() as tape:
                q_next = self.q_next.inference(bs_) #target network Q'(s', a)
                q_eval4next = self.q_eval.inference(bs_)      #main network   Q(s', a)
                q_eval = self.q_eval.inference(bs)       #main network   Q(s, a)
                q_ = tf.reduce_sum(tf.multiply(q_eval4next, self.z_list), axis=2) # a = argmax(Q(s',a))
                next_action = tf.cast(tf.argmax(q_, axis=1), tf.int32)
                indices = tf.concat(values=[tf.expand_dims(tf.range(self.batch_size), axis=1), tf.expand_dims(next_action, axis=1)], axis=1)
                Q_distributional_chosen_by_action_target = tf.gather_nd(q_next, indices)

                target = tf.tile(tf.reshape(reward,[-1, 1]), tf.constant([1, self.q_eval.N_atoms])) \
                    + tf.cast(tf.reshape((self.discount ** tf.cast(p_idx, tf.float32)), [-1, 1]), tf.float32) * tf.multiply(tf.reshape(self.z_list,[1, self.q_eval.N_atoms]),
                    (1.0 - tf.tile(tf.reshape(done ,[-1, 1]), tf.constant([1, self.q_eval.N_atoms]))))

                target = tf.clip_by_value(target, self.Vmin, self.Vmax)
                b = (target - self.Vmin) / self.delta_z
                u, l = tf.ceil(b), tf.floor(b)
                u_id, l_id = tf.cast(u, tf.int32), tf.cast(l, tf.int32)
                u_minus_b, b_minus_l = u - b, b - l

                action_list = tf.concat([tf.expand_dims(tf.range(self.batch_size), axis=1), eval_act_index], axis=1)
                Q_distributional_chosen_by_action_online = tf.gather_nd(q_eval,action_list)


                index_help = tf.tile(tf.reshape(tf.range(self.batch_size),[-1, 1]), tf.constant([1, self.q_eval.N_atoms]))
                index_help = tf.expand_dims(index_help, -1)
                u_id = tf.concat([index_help, tf.expand_dims(u_id, -1)], axis=2)
                l_id = tf.concat([index_help, tf.expand_dims(l_id, -1)], axis=2)
                error = Q_distributional_chosen_by_action_target * u_minus_b * \
                        tf.log(tf.gather_nd(Q_distributional_chosen_by_action_online, l_id)) \
                    + Q_distributional_chosen_by_action_target * b_minus_l * \
                        tf.log(tf.gather_nd(Q_distributional_chosen_by_action_online, u_id))
                error = tf.reduce_mean(error, axis=1)
                td_error = tf.abs(error)
                loss = tf.reduce_mean(tf.negative(error) * weights)

            self.q_eval.optimize(loss, global_step, tape)
            
        return loss, td_error
