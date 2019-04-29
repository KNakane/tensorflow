import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utility'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../network'))
import numpy as np
import tensorflow as tf
from agent import Agent
from eager_nn import ActorNet, CriticNet
from eager_nn import A2CNet, A3CNet


class A3C(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_net(self):
        self.actor = ActorNet(model=self.model[0], out_dim=self.n_actions, name='ActorNet', opt=self._optimizer, lr=self.lr, trainable=self.trainable, max_action=self.max_action)
        #self.actor_target = ActorNet(model=self.model[0], out_dim=self.n_actions, name='ActorNet_target', trainable=False, max_action=self.max_action)

        self.critic = CriticNet(model=self.model[1], out_dim=1, name='CriticNet', opt=self._optimizer, lr=self.lr, trainable=self.trainable)
        #self.critic_target = CriticNet(model=self.model[1], out_dim=1, name='CriticNet_target',trainable=False)
 
    def inference(self, state):
        if self.is_categorical:
            return tf.argmax(tf.reduce_sum(self.actor.inference(state) * self.z_list_broadcasted, axis=2), axis=1)
        else:
            return self.actor.inference(state)

    def update_q_net(self, replay_data, weights): #Experience Replayを使用しないようにする
        self.bs, ba, done, bs_, br, p_idx = replay_data
        eval_act_index = ba
        reward = br
        done = done

        global_step = tf.train.get_or_create_global_step()

        with tf.GradientTape() as tape:
            action_eval, values = self.q_eval.inference(self.bs)
            neg_logs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_eval, labels=eval_act_index)
            advantage = reward - values

            policy_loss = tf.reduce_mean(neg_logs * tf.nn.softplus(advantage))
            value_loss = tf.losses.mean_squared_error(reward, values)
            action_entropy = tf.reduce_mean(self.categorical_entropy(action_eval))
            self.loss = policy_loss +  self.value_loss_weight * value_loss - self.entropy_weight * action_entropy
        self.q_eval.optimize(self.loss, global_step, tape)

        self.pull_global_net()

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max


        return

    def update_global_net(self):
        """
        Global Networkのparameter を更新する
        """
        for param, target_param in zip(self.q_eval.weights, self.q_next.weights):
            target_param.assign(param)
        return

    def pull_global_net(self):
        """
        Global Network から parameterを引き出す
        """
        for param, target_param in zip(self.q_eval.weights, self.q_next.weights):
            target_param.assign(param)
        return



class A2C(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_loss_weight = 1.0
        self.entropy_weight = 0.1

    def _build_net(self):
        self.q_eval = A2CNet(model=self.model, out_dim=self.n_actions, name='A2CNet', opt=self._optimizer, lr=self.lr, trainable=True)
        return

    def inference(self, state):
        return self.actor.inference(state)

    def update_q_net(self, replay_data, weights): #Experience Replayを使用しないようにする
        self.bs, ba, done, bs_, br, p_idx = replay_data
        eval_act_index = ba
        reward = br
        done = done

        global_step = tf.train.get_or_create_global_step()

        with tf.GradientTape() as tape:
            action_eval, values = self.q_eval.inference(self.bs)
            neg_logs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_eval, labels=eval_act_index)
            advantage = reward - values

            policy_loss = tf.reduce_mean(neg_logs * tf.nn.softplus(advantage))
            value_loss = tf.losses.mean_squared_error(reward, values)
            action_entropy = tf.reduce_mean(self.categorical_entropy(action_eval))
            self.loss = policy_loss +  self.value_loss_weight * value_loss - self.entropy_weight * action_entropy
        self.q_eval.optimize(self.loss, global_step, tape)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max


        return

    def categorical_entropy(self, logits):
        a0 = logits - tf.reduce_mean(logits, axis=1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_mean(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_mean(p0 * (tf.log(z0) - a0), axis=-1)