import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utility'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../network'))
import numpy as np
import tensorflow as tf
from agent import Agent
from eager_nn import A2CNet, A3CNet


class A3C(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_loss_weight = 1.0
        self.entropy_weight = 0.1

    def _build_net(self):
        self.q_eval = A3CNet(model=self.model, out_dim=self.n_actions, name='A3CNet', opt=self._optimizer, lr=self.lr, trainable=self.trainable)
 
    @property
    def var(self):
        v = []
        for l in self.q_eval._layers:
            v += l.get_weights()
        return v

    @tf.contrib.eager.defun
    def inference(self, state):
        action, _ = self.q_eval.inference(state)
        return action

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        actions_value = tf.keras.layers.Softmax()(self.inference(observation))
        action = np.random.choice(self.actions_list, size=1, p=np.array(actions_value).ravel())[0]
        return action

    def test_choose_action(self, observation):
        observation = observation[np.newaxis, :]
        actions_value = tf.keras.layers.Softmax()(self.inference(observation))
        action = np.random.choice(self.actions_list, size=1, p=np.array(actions_value).ravel())[0]
        return action

    def update_q_net(self, replay_data, weights): #Experience Replayを使用しないようにする
        self.bs, ba, done, bs_, br, p_idx = replay_data
        eval_act_index = ba
        reward = br
        done = done

        # normalize rewards
        reward = self._discount_and_norm_rewards(reward)

        global_step = tf.train.get_or_create_global_step()

        with tf.GradientTape() as tape:
            action_eval, values = self.q_eval.inference(self.bs)
            neg_logs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_eval, labels=eval_act_index)
            advantage = tf.convert_to_tensor(reward, dtype=tf.float32) - values

            policy_loss = tf.reduce_mean(neg_logs * tf.stop_gradient(advantage))
            value_loss = tf.losses.mean_squared_error(reward, values)
            action_entropy = tf.reduce_mean(self.categorical_entropy(action_eval))
            self.loss = policy_loss +  self.value_loss_weight * value_loss - self.entropy_weight * action_entropy
        gradients = self.q_eval.get_grads(self.loss, global_step, tape)
        
        return gradients

    def update_global_net(self, grad):
        """
        Global Networkのparameter を更新する
        """
        self.q_eval.optimize(grad)
        return

    def pull_global_net(self, global_para):
        """
        Global Network から parameterを引き出す
        """
        for param, target_param in zip(self.q_eval.weights, global_para):
            param.assign(target_param)
        return

    def categorical_entropy(self, logits):
        a0 = logits - tf.reduce_mean(logits, axis=1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_mean(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_mean(p0 * (tf.log(z0) - a0), axis=-1)


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

            policy_loss = tf.reduce_mean(neg_logs * advantage)
            value_loss = tf.losses.mean_squared_error(reward, values)
            action_entropy = tf.reduce_mean(self.categorical_entropy(action_eval))
            loss = policy_loss + self.value_loss_weight * value_loss - self.entropy_weight * action_entropy
        self.q_eval.optimize(loss, global_step, tape)

        # increasing epsilon
        self.epsilon = max(self.epsilon + self.epsilon_increment, self.epsilon_max)

        return loss

    def categorical_entropy(self, logits):
        a0 = logits - tf.reduce_mean(logits, axis=1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_mean(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_mean(p0 * (tf.log(z0) - a0), axis=-1)  