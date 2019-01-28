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
        self.max_action = kwargs.pop('max_action')
        self.min_action = kwargs.pop('min_action')
        super().__init__(*args, **kwargs)
        self.noise = OrnsteinUhlenbeckProcess(num_actions=self.n_actions)

    def _build_net(self):
        self.actor = ActorNet(model=self.model[0], out_dim=self.n_actions, name='ActorNet', opt=self._optimizer, lr=self.lr, trainable=True, max_action=self.max_action)
        self.actor_target = ActorNet(model=self.model[0], out_dim=self.n_actions, name='ActorNet', opt=self._optimizer, lr=self.lr, trainable=False)

        self.critic = CriticNet(model=self.model[1], out_dim=1, name='CriticNet', opt=self._optimizer, lr=self.lr, trainable=True)
        self.critic_target = CriticNet(model=self.model[1], out_dim=1, name='CriticNet', opt=self._optimizer, lr=self.lr, trainable=False)

    def inference(self, state):
        return self.actor.inference(state)

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.inference(observation)
            if self.on_policy:
                action = np.random.choice(self.actions_list, size=1, p=np.array(actions_value)[0])
            else:
                action = np.argmax(actions_value)
        else:
            action = np.random.uniform(self.min_action, self.max_action, 1)
        return action

    def update_q_net(self, replay_data):
        self.bs, ba, done, bs_, br, p_idx = replay_data
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
            critic_next, critic_eval = self.critic_target.inference([bs_, self.actor_target.inference(bs_)]), self.critic.inference([self.bs, eval_act_index])
            critic_next = reward + self.gamma ** p_idx * critic_next * (1. - done)
            critic_next = tf.stop_gradient(critic_next)
            self.td_error = abs(critic_next - critic_eval)
            self.critic_loss = tf.losses.huber_loss(labels=critic_next, predictions=critic_eval)
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