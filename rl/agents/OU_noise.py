# -*- coding: utf-8 -*-
# Based on https://github.com/sykwer/tensorflow_ddpg/blob/master/ou_noise.py
import sys,os
import numpy as np

class OrnsteinUhlenbeckProcess():
    def __init__(self, num_actions, mu=0, theta=0.15, sigma=0.2, delta=0.5):
        self.num_actions = num_actions
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.delta = delta

        self.reset()

    def generate(self):
        prev_ou_level = self.ou_level
        drift = self.theta * (self.mu - prev_ou_level) * self.delta
        randomness = np.random.normal(loc=0, scale=np.sqrt(self.delta)*self.sigma, size=None) # Brownian motion
        self.ou_level = prev_ou_level + drift + randomness

        return self.ou_level

    def reset(self):
        self.ou_level = np.ones(self.num_actions) * self.mu