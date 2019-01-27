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
        self.actor = ActorNet(model=self.model, out_dim=self.n_actions, name='ActorNet', opt=self._optimizer, lr=self.lr, trainable=True, is_categorical=self.is_categorical)
        
        self.critic = CriticNet(model=self.model, out_dim=1, name='CriticNet', opt=self._optimizer, lr=self.lr, trainable=True, is_categorical=self.is_categorical)
        return

    def inference(self, state):
        return self.actor.inference(state)

    def update_q_net(self, replay_data):
        self.bs, ba, done, bs_, br = replay_data
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = ba
        reward = br
        done = done

        global_step = tf.train.get_or_create_global_step()

        with tf.GradientTape() as tape:
            actor_eval = self.actor.inference(self.bs)

        return