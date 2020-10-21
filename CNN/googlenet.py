import numpy as np
import tensorflow as tf
from CNN.model import MyModel
from optimizer.optimizer import *

class GoogLeNet(MyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self):
        self.conv1 = tf.keras.layers.Conv2D(64, (5,5), (1,1), 'same', kernel_regularizer=self.l2_regularizer)
        self.inception1 = Inception(number=1, filters=64, stride=1, padding='same', l2_reg_scale=self.l2_regularizer)
        self.inception2 = Inception(number=2, filters=64, stride=1, padding='same', l2_reg_scale=self.l2_regularizer)
        self.inception3 = Inception(number=3, filters=64, stride=1, padding='same', l2_reg_scale=self.l2_regularizer)
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.out = tf.keras.layers.Dense(self.out_dim, activation='softmax')
        return

    @tf.function
    def __call__(self, x, trainable=True):
        with tf.name_scope(self.name):
            x = self.conv1(x, training=trainable)
            x = self.inception1(x, training=trainable)
            x = self.inception2(x, training=trainable)
            x = self.inception3(x, training=trainable)
            x = self.gap(x, training=trainable)
            x = self.out(x, training=trainable)
            return x


class Inception(tf.keras.Model):
    def __init__(self,
                 number,
                 filters,
                 stride,
                 padding,
                 l2_reg_scale=None):
        super().__init__()
        self.name = 'Inception{}'.format(number)
        self.__filters = filters
        self.l2_regularizer = l2_reg_scale

    def _build(self):
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.conv1_pool = tf.keras.layers.Conv2D(self.__filters, (1,1), (2,2), 'same', kernel_regularizer=self.l2_regularizer)
        self.conv1 = tf.keras.layers.Conv2D(self.__filters, (1,1), (2,2), 'same', kernel_regularizer=self.l2_regularizer)
        self.conv1_3 = tf.keras.layers.Conv2D(self.__filters, (1,1), (2,2), 'same', kernel_regularizer=self.l2_regularizer)
        self.conv1_5 = tf.keras.layers.Conv2D(self.__filters, (1,1), (2,2), 'same', kernel_regularizer=self.l2_regularizer)
        self.conv3 = tf.keras.layers.Conv2D(self.__filters, (3,3), (2,2), 'same', kernel_regularizer=self.l2_regularizer)
        self.conv5 = tf.keras.layers.Conv2D(self.__filters, (5,5), (2,2), 'same', kernel_regularizer=self.l2_regularizer)
        return

    @tf.function
    def __call__(self, x, trainable=True):
        with tf.name_scope(self.name):
            # Path1
            path1 = self.conv1_5(x, training=trainable)
            path1 = self.conv5(path1, training=trainable)

            # Path2
            path2 = self.conv1_3(x, training=trainable)
            path2 = self.conv3(path2, training=trainable)

            # Path3
            path3 = self.maxpool(x, training=trainable)
            path3 = self.conv1_pool(path3, training=trainable)

            # Path4
            path4 = self.conv1(x, training=trainable)

            output = tf.concat([path1, path2, path3, path4])

            return output