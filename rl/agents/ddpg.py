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
        super().__init__(*args, **kwargs)
        self.actor_model, self.critic_model = kwargs['model']
        self.actor = Actor(model=self.actor_model, )
        self.critic = Critic()

        self.noise = OrnsteinUhlenbeckProcess(num_actions=self.n_actions)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.writer = Writer(self.sess)

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
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

        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net/Actor')
        self.eval_param = e_params
        assert len(e_params) > 0
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net/target_Actor')
        assert len(t_params) > 0

        with tf.variable_scope('replace_op'):
            self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def _build_net(self):
        with tf.variable_scope('eval_net'):
            self.q_eval_model = CNN(model=self.model, name='Actor', opt=self._optimizer, lr=self.lr, trainable=True)

            with tf.variable_scope('parameters_gradients'):
                self.parameters_gradients = tf.gradients(self.q_eval, self.eval_param, -self.q_target)
            
            with tf.variable_scope('train'):
                self.update_q_net = self.q_eval_model.optimizer.apply_gradients(zip(self.parameters_gradients,self.eval_param))

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


    def _build_net(self):
        with tf.variable_scope('eval_net'):
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
            self.q_next = Critic_Net(model=self.model, name='target_Critic', trainable=False).inference(self.s_, self.a_)
            