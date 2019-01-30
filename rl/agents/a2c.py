import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utility'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../network'))
import numpy as np
import tensorflow as tf
from agent import Agent
from eager_cnn import ActorNet, CriticNet

class A2C(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_net(self):
        self.actor = ActorNet(model=self.model[0], out_dim=self.n_actions, name='ActorNet', opt=self._optimizer, lr=self.lr*0.1, trainable=True)
        
        self.critic = CriticNet(model=self.model[1], out_dim=1, name='CriticNet', opt=self._optimizer, lr=self.lr, trainable=True)
        return

    def inference(self, state):
        return self.actor.inference(state)

    def update_q_net(self, replay_data):
        self.bs, ba, done, bs_, br = replay_data
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = ba
        reward = br
        done = done

        value_loss_weight = 1.0
        entropy_weight = 0.1


        global_step = tf.train.get_or_create_global_step()

        action_eval, values = self.actor.inference(self.bs), self.critic.inference(eval_act_index)
        neg_logs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_eval, labels=eval_act_index)
        advantage = reward - values

        policy_loss = tf.reduce_mean(neg_logs * tf.nn.softplus(advantage))
        value_loss = tf.losses.mean_squared_error(reward, values)
        action_entropy = tf.reduce_mean(self.categorical_entropy(action_eval))

        self.critic_loss = policy_loss +  value_loss_weight * value_loss 
        self.critic.optimize(self.critic_loss, global_step, tape)

        self.actor_loss = - entropy_weight * action_entropy
        self.actor.optimize(self.actor_loss, global_step, tape)

        with tf.GradientTape() as tape:
            actor_eval = self.actor.inference(self.bs)

        return

    def categorical_entropy(self, logits):
        a0 = logits - tf.reduce_mean(logits, axis=1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_mean(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_mean(p0 * (tf.log(z0) - a0), axis=-1)