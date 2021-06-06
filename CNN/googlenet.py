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

    def call(self, x, training=False):
        with tf.name_scope(self.name):
            x = self.conv1(x, training=training)
            x = self.inception1(x, training=training)
            x = self.inception2(x, training=training)
            x = self.inception3(x, training=training)
            x = self.gap(x, training=training)
            x = self.out(x, training=training)
            return x


class Inception(tf.keras.Model):
    def __init__(self,
                 number,
                 filters,
                 stride,
                 padding,
                 l2_reg_scale=None):
        super().__init__()
        self._name = 'Inception{}'.format(number)
        self.__filters = filters
        self.l2_regularizer = l2_reg_scale
        self._build()

    def _build(self):
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.conv1_pool = tf.keras.layers.Conv2D(self.__filters, (1,1), (2,2), 'same', kernel_regularizer=self.l2_regularizer)
        self.conv1 = tf.keras.layers.Conv2D(self.__filters, (1,1), (2,2), 'same', kernel_regularizer=self.l2_regularizer)
        self.conv1_3 = tf.keras.layers.Conv2D(self.__filters, (1,1), (2,2), 'same', kernel_regularizer=self.l2_regularizer)
        self.conv1_5 = tf.keras.layers.Conv2D(self.__filters, (1,1), (2,2), 'same', kernel_regularizer=self.l2_regularizer)
        self.conv3 = tf.keras.layers.Conv2D(self.__filters, (3,3), (2,2), 'same', kernel_regularizer=self.l2_regularizer)
        self.conv5 = tf.keras.layers.Conv2D(self.__filters, (5,5), (2,2), 'same', kernel_regularizer=self.l2_regularizer)
        return

    def call(self, x, training=False):
        with tf.name_scope(self._name):
            # Path1
            path1 = self.conv1_5(x, training=training)
            path1 = self.conv5(path1, training=training)

            # Path2
            path2 = self.conv1_3(x, training=training)
            path2 = self.conv3(path2, training=training)

            # Path3
            path3 = self.maxpool(x, training=training)
            path3 = self.conv1_pool(path3, training=training)

            # Path4
            path4 = self.conv1(x, training=training)

            output = tf.concat([path1, path2, path3, path4], axis=-1)

            return output