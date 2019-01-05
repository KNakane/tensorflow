# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utility'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../network'))
import numpy as np
import tensorflow as tf
from optimizer import *
from cnn import CNN
from critic_net import Critic_Net
from writer import Writer
from dqn import DQN

class DDPG():
    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()

    def update(self, replay_data):

        bs, ba, done, bs_, br = replay_data

        #target_actor session run
        q_next, q_eval = self.sess.run([self.critic.q_next, self.critic.q_eval],
                feed_dict = {self.s_:bs_,
                            self.a_:actor.q_next})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = ba
        reward = br
        done = done

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1) * (1. - done)

        # train eval network
        merged, _, self.cost = self.sess.run([self.merged, self.critic._train_op, self.loss],
                                     feed_dict={self.s: bs,
                                                self.q_target: q_target})


    def update_critic(self, ):
        self.critic.update_q_net()

    def update_actor(self):
        self.actor.update_q_net()



class Actor(DQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net/Actor')
        assert len(e_params) > 0
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net/target_Actor')
        assert len(t_params) > 0

        with tf.variable_scope('replace_op'):
            self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def _build_net(self):
        with tf.variable_scope('eval_net'):
            self.q_eval_model = CNN(model=self.model, name='Actor', opt=self._optimizer, lr=self.lr, trainable=True)

        with tf.variable_scope('target_net'):
            self.q_next = CNN(model=self.model, name='target_Actor', trainable=False).inference(self.s_)


class Critic(DQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = 0.001
        self.a = tf.placeholder(tf.float32, [None, self.n_actions], name='a')
        self.a_ = tf.placeholder(tf.float32, [None, self.n_actions], name='a_') #target action

        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net/Critic')
        assert len(e_params) > 0
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net/target_Critic')
        assert len(t_params) > 0

        with tf.variable_scope('replace_op'):
            self.replace_target_op = [tf.assign(t, self.tau * e + (1 - self.tau) * t) for t, e in zip(t_params, e_params)]

        with tf.variable_scope('action_gradients'):
            self.action_gradients = tf.gradients(self.q_eval, self.a)

    def _build_net(self):
        with tf.variable_scope('eval_net'):
            self.q_eval_model = Critic_Net(model=self.model, name='Critic', opt=self._optimizer, lr=self.lr, trainable=True)
            self.q_eval = self.q_eval_model.inference(self.s, self.a)

        with tf.variable_scope('target_net'):
            self.q_next = Critic_Net(model=self.model, name='target_Critic', trainable=False).inference(self.s_, self.a)

    def update_q_net(self):
        return