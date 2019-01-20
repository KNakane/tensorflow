# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utility'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../network'))
import numpy as np
import tensorflow as tf
from agent import Agent
from cnn import CNN
from optimizer import *
from writer import Writer

class DQN(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _build_net(self):
        # ------------------ build eval_net ------------------
        with tf.variable_scope('eval_net'):
            self.q_eval = CNN(model=self.model, name='Q_net', opt=self._optimizer, lr=self.lr, trainable=True)

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            self.q_next = CNN(model=self.model, name='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net/Q_net')
        assert len(self.e_params) > 0
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net/target_net')
        assert len(self.t_params) > 0

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

        with tf.GradientTape() as tape:
            q_next, q_eval = self.q_next.inference(bs_), self.q_eval.inference(bs)
            q_target = q_eval.copy()
            q_target[batch_index, 0] = reward + self.gamma * np.max(q_next, axis=1) * (1. - done)
            self.loss = tf.losses.huber_loss(labels=q_target, predictions=q_eval)
        
        grads = tape.gradient(self.loss, self.e_params)
        self.q_eval.method.optimizer.apply_gradients(zip(grads, self.e_params))
        
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        self._iteration += 1

    def update_target_net(self):
        with tf.GradientTape() as tape:
            [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        return



class DDQN(DQN):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_q_net(self, replay_data):
        # check to replace target parameters
        if self._iteration % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        bs, ba, done, bs_, br = replay_data

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: bs_,  # fixed params
                self.s: bs_,  # newest params
            })

        q_eval = self.sess.run(
            self.q_eval,
            feed_dict={
                self.s: bs,  # newest params
            })

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = ba
        reward = br
        done = done

        max_act4next = np.argmax(q_eval4next, axis=1)        # the action that brings the highest value is evaluated by q_eval
        selected_q_next = q_next[batch_index, max_act4next] # Double DQN, select q_next depending on above actions

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next * (1. - done)

        self.merged = tf.summary.merge_all()
        
        # train eval network
        merged, _, self.cost = self.sess.run([self.merged, self._train_op, self.loss],
                                     feed_dict={self.s: bs,
                                                self.q_target: q_target})

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        self._iteration += 1
        self.writer.add(merged, self._iteration)