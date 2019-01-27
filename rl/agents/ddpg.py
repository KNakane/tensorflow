# -*- coding: utf-8 -*-
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utility'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../network'))
import numpy as np
import tensorflow as tf
from agent import Agent
from eager_cnn import ActorNet, CriticNet
from optimizer import *
from writer import Writer
from OU_noise import OrnsteinUhlenbeckProcess

class DDPG(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_net(self):
        self.actor = ActorNet(model=self.model, out_dim=self.n_actions, name='ActorNet', opt=self._optimizer, lr=self.lr, trainable=True, is_categorical=self.is_categorical)
        self.actor_target = ActorNet(model=self.model, out_dim=self.n_actions, name='ActorNet', opt=self._optimizer, lr=self.lr, trainable=False, is_categorical=self.is_categorical)

        self.critic = CriticNet(model=self.model, out_dim=1, name='CriticNet', opt=self._optimizer, lr=self.lr, trainable=True, is_categorical=self.is_categorical)
        self.critic_target = CriticNet(model=self.model, out_dim=1, name='CriticNet', opt=self._optimizer, lr=self.lr, trainable=False, is_categorical=self.is_categorical)

    def inference(self, state):
        return self.actor.inference(state)

    def update_q_net(self, replay_data):
        self.bs, ba, done, bs_, br = replay_data
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = ba
        reward = br
        done = done

        # check to replace target parameters
        if self._iteration % self.replace_target_iter == 0:
            self.update_target_net()

        global_step = tf.train.get_or_create_global_step()

        # update critic_net
        with tf.GradientTape() as tape:
            critic_next = self.critic_target.inference([bs_, self.actor_target.inference(bs_)])
            critic_eval = self.critic.inference([self.bs, eval_act_index])
            critic_target = np.array(critic_eval).copy()
            critic_target[batch_index, eval_act_index] = reward + self.gamma * critic_next * (1. - done)

            self.td_error = abs(critic_target[batch_index, eval_act_index] - np.array(critic_eval)[batch_index, eval_act_index])
            self.critic_loss = tf.losses.huber_loss(labels=critic_target, predictions=critic_eval)
        self.critic.optimize(self.critic_loss, global_step, tape)

        # update actor_net
        with tf.GradientTape() as tape:
            actor_eval = self.actor.inference(self.bs)
            self.actor_loss = -tf.reduce_mean(self.critic.inference([self.bs, actor_eval]))
        self.actor.optimize(self.actor_loss, global_step, tape)

        return

    
    def update_target_net(self):
        # update critic_target_net
        for param, target_param in zip(self.critic.weights, self.critic_target.weights):
            target_param.assign(param)
        
        # update actor_target_net
        for param, target_param in zip(self.actor.weights, self.actor_target.weights):
            target_param.assign(param)
        return