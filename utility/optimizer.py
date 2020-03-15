# -*- coding: utf-8 -*-
import tensorflow as tf


class SGD():
    def __init__(self, learning_rate=0.1, decay_step=None, decay_rate=0.95):
        if decay_step is not None:
            self.lr = tf.optimizers.schedules.ExponentialDecay(learning_rate, decay_steps=decay_step, decay_rate=decay_rate, staircase=True)
        else:
            self.lr = learning_rate
        self.method = tf.optimizers.SGD(learning_rate=self.lr)


class Momentum(SGD):
    def __init__(self, learning_rate=0.1, decay_step=None, decay_rate=0.95):
        super().__init__(learning_rate=learning_rate, decay_step=decay_step, decay_rate=decay_rate)
        self.method = tf.keras.optimizers.SGD(learning_rate=self.lr, momentum=0.9)


class Adadelta(SGD):
    def __init__(self, learning_rate=0.1, decay_step=None, decay_rate=0.95):
        super().__init__(learning_rate=learning_rate, decay_step=decay_step, decay_rate=decay_rate)
        self.method = tf.optimizers.Adadelta(self.lr, rho=0.95, epsilon=1e-06)


class Adagrad(SGD):
    def __init__(self, learning_rate=0.1, decay_step=None, decay_rate=0.95):
        super().__init__(learning_rate=learning_rate, decay_step=decay_step, decay_rate=decay_rate)
        self.method = tf.optimizers.Adagrad(self.lr, epsilon=1e-06)


class Adam(SGD):
    def __init__(self, learning_rate=0.1, decay_step=None, decay_rate=0.95):
        super().__init__(learning_rate=learning_rate, decay_step=decay_step, decay_rate=decay_rate)
        self.method = tf.optimizers.Adam(self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

class Adamax(SGD):
    def __init__(self, learning_rate=0.1, decay_step=None, decay_rate=0.95):
        super().__init__(learning_rate=learning_rate, decay_step=decay_step, decay_rate=decay_rate)
        self.method = tf.optimizers.Adamax(self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


class RMSprop(SGD):
    def __init__(self, learning_rate=0.1, decay_step=None, momentum=0.95, epsilon=0.01):
        super().__init__(learning_rate=learning_rate, decay_step=decay_step)
        self.method = tf.optimizers.RMSprop(self.lr, rho=0.9, epsilon=1e-06)
