import os, sys
import gym
import gym.spaces
import tensorflow as tf
from rl.env.pendulum_env import WrappedPendulumEnv

class Pendulum_setting():
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.judge_agent()
        self.env = gym.make(self.FLAGS.env)
        self.env = self.env.unwrapped
        self.env = WrappedPendulumEnv(self.env)
        self.FLAGS.step = 200

    def judge_agent(self):
        agent_list = ['DDPG', 'TD3']
        for correct_agent in agent_list:
            if self.FLAGS.agent == correct_agent:
                return
                
        raise NotImplementedError()

    def train(self):
        return

    def eval(self):
        return