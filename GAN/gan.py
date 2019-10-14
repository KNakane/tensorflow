import os, sys
import numpy as np
import tensorflow as tf
from GAN.model import BasedDiscriminator, BasedGenerator, BasedGAN

class Generator(BasedGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self):
        self.fc1 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu, kernel_regularizer=self.l2_regularizer)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.fc2 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, kernel_regularizer=self.l2_regularizer)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.fc3 = tf.keras.layers.Dense(self.size*self.size*self.channel, activation=tf.nn.tanh, kernel_regularizer=self.l2_regularizer)
        self.reshape = tf.keras.layers.Reshape((self.size,self.size,self.channel))

    def __call__(self, outputs, trainable=True):
        outputs = self.fc1(outputs, training=trainable)
        outputs = self.bn1(outputs, training=trainable)
        outputs = self.fc2(outputs, training=trainable)
        outputs = self.bn2(outputs, training=trainable)
        outputs = self.fc3(outputs, training=trainable)
        outputs = self.reshape(outputs, training=trainable)
        return outputs


class Discriminator(BasedDiscriminator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self):
        self.fc1 = tf.keras.layers.Dense(1024, activation=tf.nn.leaky_relu, kernel_regularizer=self.l2_regularizer)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.fc2 = tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu, kernel_regularizer=self.l2_regularizer)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.fc3 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, kernel_regularizer=self.l2_regularizer)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.fc4 = tf.keras.layers.Dense(1, activation=None, kernel_regularizer=self.l2_regularizer)

    def __call__(self, outputs, trainable=True):
        outputs = self.fc1(outputs, training=trainable)
        outputs = self.bn1(outputs, training=trainable)
        outputs = self.fc2(outputs, training=trainable)
        outputs = self.bn2(outputs, training=trainable)
        outputs = self.fc3(outputs, training=trainable)
        outputs = self.bn3(outputs, training=trainable)
        outputs = self.fc4(outputs, training=trainable)
        return outputs

class GAN(BasedGAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def _build(self):
        self.G = Generator(size=self.size, channel=self.channel, l2_reg=self._l2_reg, l2_reg_scale=self.l2_regularizer)
        self.D = Discriminator(l2_reg=self._l2_reg, l2_reg_scale=self.l2_regularizer)

    def inference(self, inputs, batch_size, labels=None, trainable=True):
        self.z = tf.random.normal((batch_size, self._z_dim), dtype=tf.float32)
        fake_img = self.G(self.combine_distribution(self.z, labels) if self.conditional else self.z, trainable=trainable)
        
        if self.conditional and labels is not None:
            fake_img = self.combine_image(fake_img, labels)
            inputs = self.combine_image(inputs, labels)
            """
            fake_img = self.combine_binary_image(fake_img, labels)
            inputs = self.combine_binary_image(inputs, labels)
            """
        real_logit = tf.nn.sigmoid(self.D(inputs, trainable=trainable))
        fake_logit = tf.nn.sigmoid(self.D(fake_img, trainable=trainable))
        return fake_logit, real_logit, fake_img

    def test_inference(self, inputs, batch_size, index=None, trainable=False):
        if self.conditional:
            indices = index if index is not None else np.array([x%self.class_num for x in range(batch_size)],dtype=np.int32)
            labels = tf.one_hot(indices, depth=self.class_num, dtype=tf.float32)
            inputs = self.combine_distribution(inputs, labels)
        return self.G(inputs, trainable=trainable)

    def loss(self, fake_logit, real_logit):
        eps = 1e-14
        d_loss = -tf.reduce_mean(tf.math.log(real_logit + eps) + tf.math.log(1. - fake_logit + eps))
        g_loss = -tf.reduce_mean(tf.math.log(fake_logit + eps))
        """
        # discriminator loss
        real_loss = self.cross_entropy(tf.ones_like(real_logit), real_logit)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_logit), fake_logit)
        d_loss = real_loss + fake_loss

        # generator loss
        g_loss = self.cross_entropy(tf.ones_like(fake_logit), fake_logit)
        """
        return d_loss, g_loss

    def generator_optimize(self, g_loss, tape=None):
        assert tape is not None, 'please set tape in opmize'
        g_grads = tape.gradient(g_loss, self.G.trainable_variables)
        self.g_optimizer.method.apply_gradients(zip(g_grads, self.G.trainable_variables))
        return

    def discriminator_optimize(self, d_loss, tape=None, n_update=1):
        assert tape is not None, 'please set tape in opmize'
        d_grads = tape.gradient(d_loss, self.D.trainable_variables)
        for _ in range(n_update):
            self.d_optimizer.method.apply_gradients(zip(d_grads, self.D.trainable_variables))
        return