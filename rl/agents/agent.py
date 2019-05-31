# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

class Agent(tf.contrib.checkpoint.Checkpointable):
    def __init__(
            self,
            model,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.99,
            e_greedy=0.1,
            replace_target_iter=300,
            batch_size=32,
            e_greedy_decrement=None,
            update_interval=4,
            optimizer=None,
            network=None,
            trainable=True,
            is_categorical=False,
            is_noise=False,
            gpu=0
    ):
        # Eager Mode
        tf.enable_eager_execution()
        self.model = model
        self.network = network
        self.n_actions = n_actions
        self.actions_list = list(range(n_actions))
        self.n_features = n_features
        self.lr = learning_rate
        self.discount = reward_decay #discount
        self.epsilon_min = e_greedy
        self.replace_target_iter = replace_target_iter
        self.batch_size = batch_size
        self._optimizer = optimizer
        self.is_categorical = is_categorical
        self.is_noise = is_noise
        self.trainable = trainable
        self.epsilon_decrement = e_greedy_decrement
        self.update_interval = update_interval
        self.epsilon = 0 if self.is_noise else max(e_greedy - self.epsilon_decrement * self.n_warmup,self.epsilon_min)
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"

        # total learning step
        self._iteration = 0

        # consist of [target_net, evaluate_net]
        self._build_net()
        
    def _build_net(self):
        if hasattr(self.__class__.__name__, 'q_eval') == False:
            raise Exception('please set build net')
        if self._optimizer is None:
            raise Exception('please set Optimizer')
    
    def inference(self, s):
        raise Exception('please Write inference function')

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)
        observation = observation[np.newaxis, :]

        if np.random.uniform() > self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.inference(observation)
            if self.is_categorical:
                return np.max(actions_value)
            else:
                return np.argmax(actions_value)
        else:
            return np.random.randint(0, self.n_actions)

    def test_choose_action(self, observation):
        observation = observation[np.newaxis, :]
        actions_value = self.inference(observation)
        if self.is_categorical:
            return np.max(actions_value)
        else:
            return np.argmax(actions_value)

    def update_q_net(self, replay_data):
        raise Exception('please set update_q_net')

    def _discount_and_norm_rewards(self, rewards):
        """
        割引報酬和の計算
        based on https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/4f9376de9273192c25dda7be589d6c3e30cf5aec/contents/7_Policy_gradient_softmax/RL_brain.py
        """
        discounted_ep_rs = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount + rewards[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        if np.std(discounted_ep_rs) > 0:
            discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs