import numpy as np
import tensorflow as tf
from GAN.model import BasedDiscriminator
from GAN.dcgan import Generator, DCGAN
from network.layers import SpectralNormalization

class SNGAN(DCGAN):
    """
    Spectral Normalization GAN
    https://arxiv.org/abs/1802.05957

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Discriminator(BasedDiscriminator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self):
        conv1 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(2,2), padding='same', activation=None, kernel_regularizer=self.l2_regularizer)
        self.snconv1 = SpectralNormalization(conv1)
        self.leaky_relu1 = tf.keras.layers.LeakyReLU()
        conv2 = tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(2,2), padding='same', activation=None, kernel_regularizer=self.l2_regularizer)
        self.snconv2 = SpectralNormalization(conv2)
        self.leaky_relu2 = tf.keras.layers.LeakyReLU()
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.flatten = tf.keras.layers.Flatten()
        self.fc3 = tf.keras.layers.Dense(1, activation=None, kernel_regularizer=self.l2_regularizer)

    def __call__(self, outputs, trainable=True):
        outputs = self.snconv1(outputs, training=trainable)
        outputs = self.leaky_relu1(outputs, training=trainable)
        outputs = self.snconv2(outputs, training=trainable)
        outputs = self.leaky_relu2(outputs, training=trainable)
        outputs = self.flatten(outputs, training=trainable)
        outputs = self.fc3(outputs, training=trainable)
        return outputs