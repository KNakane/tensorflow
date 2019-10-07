# -*- coding: utf-8 -*-
import tensorflow as tf


class SGD():
    def __init__(self, learning_rate=0.1, decay_step=None, decay_rate=0.95):
        self.lr = learning_rate
        self.method = tf.keras.optimizers.SGD(learning_rate=self.lr)

    def optimize(self, global_step=None, loss=None, var_list=None):
        if loss is None:
            NotImplementedError()
        else:
            return self.method.minimize(loss, global_step, var_list)


class Momentum(SGD):
    def __init__(self, learning_rate=0.1, decay_step=1000, decay_rate=0.95):
        super().__init__(learning_rate=learning_rate, decay_step=decay_step, decay_rate=decay_rate)
        self.method = tf.keras.optimizers.SGD(learning_rate=self.lr, momentum=0.9)


class Adadelta(SGD):
    def __init__(self, learning_rate=0.1, decay_step=1000, decay_rate=0.95):
        super().__init__(learning_rate=learning_rate, decay_step=decay_step, decay_rate=decay_rate)
        self.method = tf.keras.optimizers.Adadelta(self.lr)


class Adagrad(SGD):
    def __init__(self, learning_rate=0.1, decay_step=1000, decay_rate=0.95):
        super().__init__(learning_rate=learning_rate, decay_step=decay_step, decay_rate=decay_rate)
        self.method = tf.keras.optimizers.Adagrad(self.lr)


class Adam(SGD):
    def __init__(self, learning_rate=0.1, decay_step=1000, decay_rate=0.95):
        super().__init__(learning_rate=learning_rate, decay_step=decay_step, decay_rate=decay_rate)
        self.method = tf.keras.optimizers.Adam(self.lr, beta1=0.5)


class RMSProp(SGD):
    def __init__(self, learning_rate=0.1, decay_step=1000, momentum=0.95, epsilon=0.01):
        super().__init__(learning_rate=learning_rate, decay_step=decay_step)
        self.method = tf.keras.optimizers.RMSProp(self.lr, momentum=momentum, epsilon=epsilon)
