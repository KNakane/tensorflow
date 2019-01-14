# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utility'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../network'))
import numpy as np
import tensorflow as tf
from optimizer import *
from dqn import DQN
from cnn import CNN
from critic_net import Critic_Net
from writer import Writer
from OU_noise import OrnsteinUhlenbeckProcess


class DDPG(DQN):
    def __init__(self, *args, **kwargs):
        self.actor_model, self.critic_model = kwargs['model']
        super().__init__(model=self.actor_model,
                         n_actions=kwargs['n_actions'],
                         n_features=kwargs['n_features'],
                         learning_rate=kwargs['learning_rate'],
                         reward_decay=kwargs['reward_decay'] if 'reward_decay' in kwargs else 0.9,
                         e_greedy=kwargs['e_greedy'],
                         replace_target_iter=kwargs['replace_target_iter'],
                         batch_size=kwargs['batch_size'],
                         e_greedy_increment=kwargs['e_greedy_increment'],
                         optimizer=kwargs['optimizer'])

        self.actor = Actor(model=self.actor_model, 
                           n_actions=kwargs['n_actions'],
                           n_features=kwargs['n_features'],
                           learning_rate=kwargs['learning_rate'],
                           reward_decay=kwargs['reward_decay'] if 'reward_decay' in kwargs else 0.9,
                           e_greedy=kwargs['e_greedy'],
                           replace_target_iter=kwargs['replace_target_iter'],
                           batch_size=kwargs['batch_size'],
                           e_greedy_increment=kwargs['e_greedy_increment'],
                           optimizer=kwargs['optimizer'])

        self.critic = Critic(model=self.critic_model, 
                             n_actions=kwargs['n_actions'],
                             n_features=kwargs['n_features'],
                             learning_rate=kwargs['learning_rate'],
                             reward_decay=kwargs['reward_decay'] if 'reward_decay' in kwargs else 0.9,
                             e_greedy=kwargs['e_greedy'],
                             replace_target_iter=kwargs['replace_target_iter'],
                             batch_size=kwargs['batch_size'],
                             e_greedy_increment=kwargs['e_greedy_increment'],
                             optimizer=kwargs['optimizer'])

        self.noise = OrnsteinUhlenbeckProcess(num_actions=self.n_actions)

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.actor.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action + self.noise.generate()

    def update(self, replay_data):
        # Prepare minibatch sampling from replay buffer
        bs, ba, done, bs_, br = replay_data

        #target_actor session run
        action = self.sess.run(self.actor.q_next, feed_dict={self.s_:bs_})
        q_next, q_eval = self.sess.run([self.critic.q_next, self.critic.q_eval],
                                        feed_dict = {self.critic.s_: bs_,
                                                     self.critic.a_: action,
                                                     self.critic.s: bs,
                                                     self.critic.a: ba})

        # put result in q_target[]
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = ba
        reward = br
        done = done

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1) * (1. - done)

        # train critic net
        _ = self.sess.run(self.critic.update_q_net, 
                                feed_dict={self.critic.s: bs,
                                           self.critic.a: ba,
                                           self.critic.q_target: q_target})

        # calculate policy gradient and train actor net
        action_batch_for_gradients = self.sess.run(self.actor.q_eval, feed_dict={self.actor.s:bs})
        q_gradient_batch = self.sess.run(self.critic.action_gradients,
                                                    feed_dict = {self.critic.s: bs,
                                                                 self.critic.a: action_batch_for_gradients})

        self.merged = tf.summary.merge_all()
        _, merged = self.sess.run([self.actor.update_q_net,self.merged], 
                                            feed_dict={self.actor.q_target:q_gradient_batch,
                                                       self.actor.s :bs})

        # check to replace target parameters
        if self._iteration % self.replace_target_iter == 0:
            self.sess.run([self.critic.replace_target_op,self.actor.replace_target_op])

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        self._iteration += 1
        self.writer.add(merged, self._iteration)


class Actor(DQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.policy_gradients()

    def _build_net(self):
        with tf.variable_scope('Actor_qeval_input'):
            try:
                self.s = tf.placeholder(tf.float32, [None] + list(self.n_features), name='s')  # input
            except:
                self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')    # input

        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'):
            self.q_eval_model = CNN(model=self.model, name='Actor', opt=self._optimizer, lr=self.lr, trainable=True)
            self.q_eval = self.q_eval_model.inference(self.s)

        with tf.variable_scope('target_net'):
            try:
                self.s_ = tf.placeholder(tf.float32, [None] + list(self.n_features), name='s_')    # input
            except:
                self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
            self.q_next = CNN(model=self.model, name='target_Actor', trainable=False).inference(self.s_)

        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net/Actor')
        self.eval_param = e_params
        assert len(e_params) > 0
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net/target_Actor')
        assert len(t_params) > 0

        with tf.variable_scope('replace_op'):
            self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def policy_gradients(self):
        with tf.variable_scope('eval_net'):
            with tf.variable_scope('parameters_gradients'):
                self.parameters_gradients = tf.gradients(self.q_eval, self.eval_param, -self.q_target)
            
            with tf.variable_scope('train'):
                self.update_q_net = self.q_eval_model.optimizer.method.apply_gradients(zip(self.parameters_gradients,self.eval_param))


class Critic(DQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_net(self):
        self.tau = 0.001
        with tf.variable_scope('qeval_input'):
            try:
                self.s = tf.placeholder(tf.float32, [None] + list(self.n_features), name='s')  # input
            except:
                self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')    # input
        
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'):
            self.a = tf.placeholder(tf.float32, [None, self.n_actions], name='a')
            self.q_eval_model = Critic_Net(model=self.model, name='Critic', opt=self._optimizer, lr=self.lr, trainable=True)
            self.q_eval = self.q_eval_model.inference(self.s, self.a)

            with tf.variable_scope('action_gradients'):
                self.action_gradients = tf.gradients(self.q_eval, self.a)

            with tf.variable_scope('loss'):
                self.loss = tf.losses.huber_loss(labels=self.q_target, predictions=self.q_eval)
                #self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
                self.loss_summary = tf.summary.scalar("loss", self.loss)
            
            with tf.variable_scope('train'):
                self.update_q_net = self.q_eval_model.optimize(self.loss)

        with tf.variable_scope('target_net'):
            try:
                self.s_ = tf.placeholder(tf.float32, [None] + list(self.n_features), name='s_')    # input
            except:
                self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
            self.a_ = tf.placeholder(tf.float32, [None, self.n_actions], name='a_') #target action
            self.q_next = Critic_Net(model=self.model, name='target_Critic', trainable=False).inference(self.s_, self.a_)

        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net/Critic')
        assert len(e_params) > 0
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net/target_Critic')
        assert len(t_params) > 0

        with tf.variable_scope('replace_op'):
            self.replace_target_op = [tf.assign(t, self.tau * e + (1 - self.tau) * t) for t, e in zip(t_params, e_params)]
            