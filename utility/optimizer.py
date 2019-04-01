# -*- coding: utf-8 -*-
import tensorflow as tf


class SGD():
    def __init__(self, learning_rate=0.1, decay_step=None, decay_rate=0.95):
        global_step = tf.train.get_or_create_global_step()
        if decay_rate is not None:
            self.lr = tf.train.exponential_decay(learning_rate,global_step,decay_step,decay_rate,staircase=True)
        else:
            self.lr = learning_rate
        self.method = tf.train.GradientDescentOptimizer(self.lr)

    def optimize(self, global_step=None, loss=None, var_list=None):
        if loss is None:
            NotImplementedError()
        else:
            return self.method.minimize(loss, global_step, var_list)


class Momentum(SGD):
    def __init__(self, learning_rate=0.1, decay_step=1000, decay_rate=0.95):
        super().__init__(learning_rate=learning_rate, decay_step=decay_step, decay_rate=decay_rate)
        self.method = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)


class Adadelta(SGD):
    def __init__(self, learning_rate=0.1, decay_step=1000, decay_rate=0.95):
        super().__init__(learning_rate=learning_rate, decay_step=decay_step, decay_rate=decay_rate)
        self.method = tf.train.AdadeltaOptimizer(self.lr)


class Adagrad(SGD):
    def __init__(self, learning_rate=0.1, decay_step=1000, decay_rate=0.95):
        super().__init__(learning_rate=learning_rate, decay_step=decay_step, decay_rate=decay_rate)
        self.method = tf.train.AdagradOptimizer(self.lr)


class Adam(SGD):
    def __init__(self, learning_rate=0.1, decay_step=1000, decay_rate=0.95):
        super().__init__(learning_rate=learning_rate, decay_step=decay_step, decay_rate=decay_rate)
        self.method = tf.train.AdamOptimizer(self.lr, beta1=0.5)


class RMSProp(SGD):
    def __init__(self, learning_rate=0.1, decay_step=1000, decay_rate=0.95):
        super().__init__(learning_rate=learning_rate, decay_step=decay_step, decay_rate=decay_rate)
        self.method = tf.train.RMSPropOptimizer(self.lr)
