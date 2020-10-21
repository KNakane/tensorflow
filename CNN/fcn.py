import os, sys
import numpy as np
import tensorflow as tf
from CNN.model import MyModel
from optimizer.optimizer import *

class FCN(MyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self):
        self.conv1 = tf.keras.layers.Conv1D(32, kernel_size=7, padding='valid', activation='relu', kernel_regularizer=self.l2_regularizer)
        self.conv2 = tf.keras.layers.Conv1D(32, kernel_size=7, padding='valid', activation='relu', kernel_regularizer=self.l2_regularizer)
        self.conv3 = tf.keras.layers.Conv1D(32, kernel_size=5, padding='valid', activation='relu', kernel_regularizer=self.l2_regularizer)
        self.conv4 = tf.keras.layers.Conv1D(32, kernel_size=5, padding='valid', activation='relu', kernel_regularizer=self.l2_regularizer)
        self.gap = tf.keras.layers.GlobalAveragePooling1D()
        self.out = tf.keras.layers.Dense(self.out_dim, activation='softmax')
        return

    @tf.function
    def __call__(self, x, trainable=True):
        with tf.name_scope(self.name):
            x = self.conv1(x, training=trainable)
            x = self.conv2(x, training=trainable)
            x = self.conv3(x, training=trainable)
            x = self.conv4(x, training=trainable)
            x = self.gap(x)
            x = self.out(x, training=trainable)
            return x