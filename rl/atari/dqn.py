# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../practice/program'))
import numpy as np
import tensorflow as tf
from optimizer import *
from writer import Writer

class DQN():
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
            optimizer='RMSProp',
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
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net/Q_net')
        assert len(e_params) > 0
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net/target_net')
        assert len(t_params) > 0

        with tf.variable_scope('replace_op'):
            self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.writer = Writer(self.sess)

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('qeval_input'):
            try:
                self.s = tf.placeholder(tf.float32, [None] + list(self.n_features), name='s')  # input
            except:
                self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input

        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        
        with tf.variable_scope('eval_net'):
            self.q_eval = self.model(inputs=self.s, name='Q_net', trainable=True)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            self.loss_summary = tf.summary.scalar("loss", self.loss)
        
        with tf.variable_scope('train'):
            opt = eval(self._optimizer)(self.lr)
            self._train_op = opt.optimize(self.loss)

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_input'):
            try:
                self.s_ = tf.placeholder(tf.float32, [None] + list(self.n_features), name='s_')    # input
            except:
                self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            self.q_next = self.model(inputs=self.s_, name='target_net', trainable=False)

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def update_q_net(self, replay_data):
        # check to replace target parameters
        if self._iteration % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        bs, ba, done, bs_, br = replay_data

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: bs_,  # fixed params
                self.s: bs,  # newest params
            })

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = ba
        reward = br
        done = done

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1) * (1. - done)

        self.merged = tf.summary.merge_all()
        
        # train eval network
        merged, _, self.cost = self.sess.run([self.merged, self._train_op, self.loss],
                                     feed_dict={self.s: bs,
                                                self.q_target: q_target})

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        self._iteration += 1
        self.writer.add(merged, self._iteration)


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