# -*- coding: utf-8 -*-
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utility'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../network'))
import numpy as np
import tensorflow as tf
from agent import Agent
from eager_nn import ActorNet, CriticNet
from optimizer import *
from OU_noise import OrnsteinUhlenbeckProcess

class DDPG(Agent):
    def __init__(self, *args, **kwargs):
        self.max_action = kwargs.pop('max_action')
        super().__init__(*args, **kwargs)
        self.noise = OrnsteinUhlenbeckProcess(num_actions=self.n_actions)
        self.tau = 0.01

    def _build_net(self):
        self.actor = ActorNet(model=self.model[0], out_dim=self.n_actions, name='ActorNet', opt=self._optimizer, lr=self.lr, trainable=self.trainable, max_action=self.max_action)
        self.actor_target = ActorNet(model=self.model[0], out_dim=self.n_actions, name='ActorNet_target', trainable=False, max_action=self.max_action)

        self.critic = CriticNet(model=self.model[1], out_dim=1, name='CriticNet', opt=self._optimizer, lr=self.lr, trainable=self.trainable)
        self.critic_target = CriticNet(model=self.model[1], out_dim=1, name='CriticNet_target',trainable=False)

    @tf.contrib.eager.defun
    def inference(self, state):
        return self.actor.inference(state)

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        action = self.inference(observation) + np.expand_dims(self.noise.generate(),axis=0)
        return np.array(action[0])
    
    def test_choose_action(self, observation):
        observation = observation[np.newaxis, :]
        action = self.inference(observation)
        return np.array(action[0])

    #@tf.contrib.eager.defun
    def update_q_net(self, replay_data, weights):
        bs, ba, done, bs_, br, p_idx = replay_data
        self.bs = np.array(bs, dtype=np.float32)
        bs_ = np.array(bs_, dtype=np.float32)
        eval_act_index = np.reshape(ba,(self.batch_size, self.n_actions))
        reward = np.reshape(br,(self.batch_size,1))
        done = np.reshape(done,(self.batch_size,1))
        p_idx = np.reshape(p_idx,(self.batch_size,1))

        global_step = tf.train.get_or_create_global_step()

        # update critic_net
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                critic_next, critic_eval = self.critic_target.inference([bs_, self.actor_target.inference(bs_)]), self.critic.inference([self.bs, eval_act_index])
                target_Q = tf.cast(reward + self.discount ** p_idx * critic_next * (1. - done), tf.float32)
                target_Q = tf.stop_gradient(target_Q)
                # ↓critic_loss
                error = tf.losses.huber_loss(labels=target_Q, predictions=critic_eval)
                self.td_error = tf.abs(tf.reduce_mean(target_Q - critic_eval, axis=1))
                critic_loss = tf.reduce_mean(error * weights, keep_dims=True)
            self.critic.optimize(critic_loss, global_step, tape)

            # update actor_net
            with tf.GradientTape() as tape:
                actor_eval = tf.cast(self.actor.inference(self.bs), tf.float32)
                actor_loss = -tf.reduce_mean(self.critic.inference([self.bs, actor_eval]))
            self.actor.optimize(actor_loss, global_step, tape)

            # check to replace target parameters
            self.update_target_net()

        return [critic_loss, actor_loss]

    
    def update_target_net(self):
        # update critic_target_net
        for param, target_param in zip(self.critic.weights, self.critic_target.weights):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)
        
        # update actor_target_net
        for param, target_param in zip(self.actor.weights, self.actor_target.weights):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)
        return


class TD3(Agent):
    def __init__(self, *args, **kwargs):
        self.max_action = kwargs.pop('max_action')
        super().__init__(*args, **kwargs)
        self.noise = OrnsteinUhlenbeckProcess(num_actions=self.n_actions)
        self.tau = 0.01
        self.policy_freq = 2

    def _build_net(self):
        self.actor = ActorNet(model=self.model[0], out_dim=self.n_actions, name='ActorNet', opt=self._optimizer, lr=self.lr, trainable=True, max_action=self.max_action)
        self.actor_target = ActorNet(model=self.model[0], out_dim=self.n_actions, name='ActorNet_target', trainable=False, max_action=self.max_action)

        self.critic1 = CriticNet(model=self.model[1], out_dim=1, name='CriticNet1', opt=self._optimizer, lr=self.lr, trainable=True)
        self.critic2 = CriticNet(model=self.model[1], out_dim=1, name='CriticNet2', opt=self._optimizer, lr=self.lr, trainable=True)
        self.critic_target1 = CriticNet(model=self.model[1], out_dim=1, name='CriticNet_target1',trainable=False)
        self.critic_target2 = CriticNet(model=self.model[1], out_dim=1, name='CriticNet_target2',trainable=False)

    @tf.contrib.eager.defun
    def inference(self, state):
        return self.actor.inference(state)

    
    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        action = self.inference(observation) + np.expand_dims(self.noise.generate(),axis=0)
        return np.array(action[0])

    def test_choose_action(self, observation):
        observation = observation[np.newaxis, :]
        action = self.inference(observation)
        return action[0]

    #@tf.contrib.eager.defun
    def update_q_net(self, replay_data, weights, noise_clip=0.5):
        bs, ba, done, bs_, br, p_idx = replay_data
        self.bs = np.array(bs, dtype=np.float32)
        bs_ = np.array(bs_, dtype=np.float32)
        eval_act_index = np.reshape(ba,(self.batch_size,self.n_actions))
        reward = np.reshape(br,(self.batch_size,1))
        done = np.reshape(done,(self.batch_size,1))
        p_idx = np.reshape(p_idx,(self.batch_size,1))

        with tf.device(self.device):
            global_step = tf.train.get_or_create_global_step()

            noise = tf.clip_by_value(tf.random.normal(shape=[self.batch_size,1],mean=0.0, stddev=0.2) , -noise_clip, noise_clip)
            next_action = tf.clip_by_value(self.actor_target.inference(bs_) + noise, -self.max_action, self.max_action)
            critic_next1, critic_next2 = self.critic_target1.inference([bs_, next_action]), self.critic_target2.inference([bs_, next_action])
            critic_next = tf.minimum(critic_next1, critic_next2)
            target_Q = reward + self.discount ** p_idx * critic_next * (1. - done)
            target_Q = tf.stop_gradient(target_Q)

            # update critic_net1
            with tf.GradientTape() as tape:
                critic_eval1 = self.critic1.inference([self.bs, eval_act_index])
                error = tf.losses.huber_loss(labels=target_Q, predictions=critic_eval1)
                self.td_error = tf.abs(tf.reduce_mean(target_Q - critic_eval1, axis=1))
                critic_loss1 = tf.reduce_mean(error * weights, keep_dims=True)
            self.critic1.optimize(critic_loss1, global_step, tape)

            with tf.GradientTape() as tape:
                critic_eval2 = self.critic2.inference([self.bs, eval_act_index])
                critic_loss2 = tf.reduce_mean(tf.losses.huber_loss(labels=target_Q, predictions=critic_eval2) * weights, keep_dims=True)
            self.critic2.optimize(critic_loss2, global_step, tape)

            if self._iteration % self.policy_freq == 0:
                # update actor_net
                with tf.GradientTape() as tape:
                    actor_eval = tf.cast(self.actor.inference(self.bs), tf.float32)
                    self.actor_loss = -tf.reduce_mean(self.critic1.inference([self.bs, actor_eval]))
                self.actor.optimize(self.actor_loss, global_step, tape)

                # check to replace target parameters
                self.update_target_net()

        self._iteration += 1

        return [critic_loss1, critic_loss2, self.actor_loss]

    def update_target_net(self):
        # update critic_target_net1
        for param, target_param in zip(self.critic1.weights, self.critic_target1.weights):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)

        # update critic_target_net2
        for param, target_param in zip(self.critic2.weights, self.critic_target2.weights):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)
        
        # update actor_target_net
        for param, target_param in zip(self.actor.weights, self.actor_target.weights):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)
        return