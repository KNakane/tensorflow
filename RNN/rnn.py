import numpy as np
import tensorflow as tf
from CNN.model import MyModel
from optimizer.optimizer import *

class RNN(MyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self):
        self.rnn1 = tf.keras.layers.SimpleRNN(128)
        self.out = tf.keras.layers.Dense(self.out_dim, activation='softmax')
        return

    @tf.function
    def __call__(self, x, trainable=True):
        with tf.name_scope(self.name):
            x = self.rnn1(x, training=trainable)
            x = self.out(x, training=trainable)
            return x

class LSTM(MyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self):
        self.lstm1 = tf.keras.layers.LSTM(128)
        self.out = tf.keras.layers.Dense(self.out_dim, activation='softmax')
        return

    @tf.function
    def __call__(self, x, trainable=True):
        with tf.name_scope(self.name):
            x = self.lstm1(x, training=trainable)
            x = self.out(x, training=trainable)
            return x

class GRU(MyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self):
        self.gru1 = tf.keras.layers.GRU(128)
        self.out = tf.keras.layers.Dense(self.out_dim, activation='softmax')
        return

    @tf.function
    def __call__(self, x, trainable=True):
        with tf.name_scope(self.name):
            x = self.gru1(x, training=trainable)
            x = self.out(x, training=trainable)
            return x