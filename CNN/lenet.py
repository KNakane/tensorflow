import os, sys
import numpy as np
import tensorflow as tf
from CNN.model import MyModel
from utility.optimizer import *

class LeNet(MyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = eval(kwargs['opt'])(learning_rate=kwargs['lr'], decay_step=None, decay_rate=0.95)

    def _build(self):
        self.conv1 = tf.keras.layers.Conv2D(6, kernel_size=(5, 5), padding='valid', activation='relu', kernel_regularizer=self.l2_regularizer)
        self.pooling1 = tf.keras.layers.MaxPooling2D(padding='same')
        self.conv2 = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), padding='valid', activation='relu', kernel_regularizer=self.l2_regularizer)
        self.pooling2 = tf.keras.layers.MaxPooling2D(padding='same')
        self.flat = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(120, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.fc2 = tf.keras.layers.Dense(84, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.out = tf.keras.layers.Dense(self.out_dim, activation='softmax')
        return
    
    def inference(self, x):
        x = self.conv1(x)
        x = self.pooling1(x)
        x = self.conv2(x)
        x = self.pooling2(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x


class VGG(MyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self):
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=self.l2_regularizer)
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=self.l2_regularizer)
        self.pooling1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')
        self.conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=self.l2_regularizer)
        self.conv4 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=self.l2_regularizer)
        self.pooling2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')
        self.conv5 = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=self.l2_regularizer)
        self.conv6 = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=self.l2_regularizer)
        self.pooling3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')
        self.flat = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(500, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.fc2 = tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.out = tf.keras.layers.Dense(self.out_dim, activation='softmax')
        return
    
    def inference(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pooling1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pooling2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pooling3(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x