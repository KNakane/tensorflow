# -*- coding: utf-8 -*-
import tensorflow as tf


class SGD():
    def __init__(self, learning_rate=0.1):
        self.lr = learning_rate

    def optimize(self, loss=None):
        if loss is None:
            NotImplementedError()
        else:
            return tf.train.GradientDescentOptimizer(self.lr).minimize(loss)


class Momentum(SGD):
    def __init__(self, learning_rate=0.1):
        super().__init__(learning_rate=learning_rate)

    def optimize(self, loss=None):
        return tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9).minimize(loss)


class Adadelta(SGD):
    def __init__(self, learning_rate=0.1):
        super().__init__(learning_rate=learning_rate)

    def optimize(self, loss=None):
        return tf.train.AdadeltaOptimizer(self.lr).minimize(loss)


class Adagrad(SGD):
    def __init__(self, learning_rate=0.1):
        super().__init__(learning_rate=learning_rate)

    def optimize(self, loss=None):
        return tf.train.AdagradOptimizer(self.lr).minimize(loss)


class Adam(SGD):
    def __init__(self, learning_rate=0.1):
        super().__init__(learning_rate=learning_rate)

    def optimize(self, loss=None):
        return tf.train.AdamOptimizer(self.lr).minimize(loss)


class RMSProp(SGD):
    def __init__(self, learning_rate=0.1):
        super().__init__(learning_rate=learning_rate)

    def optimize(self, loss=None):
        return tf.train.RMSPropOptimizer(self.lr).minimize(loss)