import os, sys
import numpy as np
import tensorflow as tf
from CNN.model import MyModel
from optimizer.optimizer import *

# Paper : https://arxiv.org/pdf/1803.01271.pdf

class TCN(MyModel): # Temporal Convolutional Networks
    def __init__(self, *args, **kwargs):
        self.__levels = 8
        self.__num_channel = 25
        self.__kernel_size = 7
        self.__dropout_rate = 0.05
        self.__channel_sizes = [self.__num_channel] * self.__levels
        super().__init__(*args, **kwargs)

    def _build(self):
        self.temporal_layer = []
        for i in range(self.__levels):
            dilation_rate = 2 ** i
            layer = TemporalBlock(number=i+1, dilation_rate=dilation_rate, 
                                  filters=self.__channel_sizes[i], kernel_size=self.__kernel_size, stride=1, 
                                  padding='causal', dropout_rate=self.__dropout_rate, l2_reg_scale=self.l2_regularizer)
            self.temporal_layer.append(layer)

        self.out = tf.keras.layers.Dense(self.out_dim, activation='softmax')

    @tf.function
    def __call__(self, x, trainable=True):
        with tf.name_scope(self.name):
            for layer in self.temporal_layer:
                x = layer(x, trainable)
            x = self.out(x, training=trainable)
            return x


class TemporalBlock(tf.keras.Model):
    def __init__(self,
                 number,
                 dilation_rate,
                 filters,
                 kernel_size,
                 stride,
                 padding,
                 dropout_rate=0.2,
                 l2_reg_scale=None):
        super().__init__()
        assert padding in ['causal', 'same']
        self._name = 'TemporalBlock_{}'.format(number)
        self.__dilation_rate = dilation_rate
        self.__filters = filters
        self.__kernel_size = kernel_size
        self.__strides = stride
        self.__padding = padding
        self.__dropout_rate = dropout_rate
        self.l2_regularizer = l2_reg_scale
        self._build()

    def _build(self):
        # Block 1
        self.conv1 = tf.keras.layers.Conv1D(filters=self.__filters, kernel_size=self.__kernel_size, 
                                            strides=self.__strides, padding=self.__padding, dilation_rate=self.__dilation_rate,
                                            activation=None, kernel_regularizer=self.l2_regularizer)
        self.batch1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu1 = tf.keras.layers.ReLU()
        self.dropout1 = tf.keras.layers.Dropout(rate=self.__dropout_rate)

        # Block 2
        self.conv2 = tf.keras.layers.Conv1D(filters=self.__filters, kernel_size=self.__kernel_size, 
                                            strides=self.__strides, padding=self.__padding, dilation_rate=self.__dilation_rate,
                                            activation=None, kernel_regularizer=self.l2_regularizer)
        self.batch2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu2 = tf.keras.layers.ReLU()
        self.dropout2 = tf.keras.layers.Dropout(rate=self.__dropout_rate)

        self.downsample = tf.keras.layers.Conv1D(filters=self.__filters, kernel_size=1, 
                                                 padding='same', activation=None, kernel_regularizer=self.l2_regularizer)
        self.relu3 = tf.keras.layers.ReLU()

    @tf.function
    def __call__(self, x, trainable=True):
        with tf.name_scope(self._name):
            output = self.conv1(x, training=trainable)
            output = self.batch1(output, training=trainable)
            output = self.relu1(output, training=trainable)
            output = self.dropout1(output, training=trainable)
            
            output = self.conv2(output, training=trainable)
            output = self.batch2(output, training=trainable)
            output = self.relu2(output, training=trainable)
            output = self.dropout2(output, training=trainable)

            if tf.shape(output) != tf.shape(x):
                x = self.downsample(x)
            output = tf.math.add(x, output)
            output = self.relu3(output, training=trainable)
            return output