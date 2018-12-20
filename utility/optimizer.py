# -*- coding: utf-8 -*-
import tensorflow as tf


class SGD():
    def __init__(self, learning_rate=0.1):
        self.lr = learning_rate
        self.method = tf.train.GradientDescentOptimizer(self.lr)

    def optimize(self, global_step, loss=None):
        if loss is None:
            NotImplementedError()
        else:
            return self.method.minimize(loss, global_step)


class Momentum(SGD):
    def __init__(self, learning_rate=0.1):
        super().__init__(learning_rate=learning_rate)
        self.method = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)

    def optimize(self, global_step, loss=None):
        return self.method.minimize(loss, global_step)


class Adadelta(SGD):
    def __init__(self, learning_rate=0.001):
        super().__init__(learning_rate=learning_rate)
        self.method = tf.train.AdadeltaOptimizer(self.lr)

    def optimize(self, global_step, loss=None):
        return self.method.minimize(loss, global_step)


class Adagrad(SGD):
    def __init__(self, learning_rate=0.1):
        super().__init__(learning_rate=learning_rate)
        self.method = tf.train.AdagradOptimizer(self.lr)

    def optimize(self, global_step, loss=None):
        return self.method.minimize(loss, global_step)


class Adam(SGD):
    def __init__(self, learning_rate=0.1):
        super().__init__(learning_rate=learning_rate)
        self.method = tf.train.AdamOptimizer(self.lr)

    def optimize(self, global_step, loss=None):
        return self.method.minimize(loss, global_step)


class RMSProp(SGD):
    def __init__(self, learning_rate=0.1):
        super().__init__(learning_rate=learning_rate)
        self.method = tf.train.RMSPropOptimizer(self.lr)

    def optimize(self, global_step, loss=None):
        return self.method.minimize(loss, global_step)