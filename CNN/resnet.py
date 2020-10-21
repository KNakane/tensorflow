import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from CNN.model import BasedResNet, ResidualBlock
from optimizer.optimizer import *

class ResNet18(BasedResNet):
    """
    https://arxiv.org/abs/1512.03385
    https://github.com/yusugomori/deeplearning-tf2/blob/master/models/resnet34_fashion_mnist.py
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self):
        self.block1 = [
            self._building_block(64) for i in range(2)
        ]
        self.block2 = [
            self._building_block(128) for i in range(2)
        ]
        self.block3 = [
            self._building_block(256) for i in range(2)
        ]
        self.block4 = [
            self._building_block(512) for i in range(2)
        ]
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(1000, activation='relu')
        self.out = tf.keras.layers.Dense(self.out_dim)
        return

    @tf.function
    def __call__(self, x, trainable=True):
        with tf.name_scope(self.name):
            x = self.conv1(x, training=trainable)
            x = self.bn1(x, training=trainable)
            x = self.relu1(x, training=trainable)
            x = self.pool1(x, training=trainable)
            for block in self.block1:
                x = block(x, training=trainable)
            for block in self.block2:
                x = block(x, training=trainable)
            for block in self.block3:
                x = block(x, training=trainable)
            for block in self.block4:
                x = block(x, training=trainable)
            x = self.gap(x, training=trainable)
            x = self.fc(x, training=trainable)
            x = self.out(x, training=trainable)
            return x


class ResNet34(BasedResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self):
        self.block1 = [
            self._building_block(64, True) if i == 2 else self._building_block(64) for i in range(3)
        ]
        self.block2 = [
            self._building_block(128, True) if i == 3 else self._building_block(128) for i in range(4)
        ]
        self.block3 = [
            self._building_block(256, True) if i == 5 else self._building_block(256) for i in range(6)
        ]
        self.block4 = [
            self._building_block(512, True) if i == 2 else self._building_block(512) for i in range(3)
        ]
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(1000, activation='relu')
        self.out = tf.keras.layers.Dense(self.out_dim)
        return

    @tf.function
    def __call__(self, x, trainable=True):
        with tf.name_scope(self.name):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.pool1(x)
            for block in self.block1:
                x = block(x)
            for block in self.block2:
                x = block(x)
            for block in self.block3:
                x = block(x)
            for block in self.block4:
                x = block(x)
            x = self.gap(x)
            x = self.fc(x)
            x = self.out(x)
            return x

class ResNet50(BasedResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self):
        self.block1 = [
            self._building_block(256, True) if i == 2 else self._building_block(256) for i in range(3)
        ]
        self.block2 = [
            self._building_block(512, True) if i == 3 else self._building_block(512) for i in range(4)
        ]
        self.block3 = [
            self._building_block(1024, True) if i == 5 else self._building_block(1024) for i in range(7)
        ]
        self.block4 = [
            self._building_block(2048, True) if i == 2 else self._building_block(2048) for i in range(4)
        ]
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(1000, activation='relu')
        self.out = tf.keras.layers.Dense(self.out_dim)
        return

    @tf.function
    def __call__(self, x, trainable=True):
        with tf.name_scope(self.name):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.pool1(x)
            for block in self.block1:
                x = block(x)
            for block in self.block2:
                x = block(x)
            for block in self.block3:
                x = block(x)
            for block in self.block4:
                x = block(x)
            x = self.gap(x)
            x = self.fc(x)
            x = self.out(x)
            return x