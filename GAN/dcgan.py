import os, sys
import numpy as np
import tensorflow as tf
from GAN.model import BasedDiscriminator, BasedGenerator, BasedGAN

class Generator(BasedGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self):
        self.fc1 = tf.keras.layers.Dense(7*7*256, activation=None, kernel_regularizer=self.l2_regularizer)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.leaky_relu1 = tf.keras.layers.LeakyReLU()
        self.reshape = tf.keras.layers.Reshape((7, 7, 256))
        self.deconv1 = tf.keras.layers.Conv2DTranspose(128, kernel_size=(5,5), strides=(1,1), padding='same', kernel_regularizer=self.l2_regularizer)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.leaky_relu2 = tf.keras.layers.LeakyReLU()
        self.deconv2 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(5,5), strides=(2,2), padding='same', kernel_regularizer=self.l2_regularizer)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.leaky_relu3 = tf.keras.layers.LeakyReLU()
        self.deconv3 = tf.keras.layers.Conv2DTranspose(self.channel, kernel_size=(5,5), strides=(2,2), activation='tanh', padding='same', kernel_regularizer=self.l2_regularizer)

    @tf.function
    def __call__(self, outputs, trainable=True):
        outputs = self.fc1(outputs, training=trainable)
        outputs = self.bn1(outputs, training=trainable)
        outputs = self.leaky_relu1(outputs, training=trainable)
        outputs = self.reshape(outputs, training=trainable)
        outputs = self.deconv1(outputs, training=trainable)
        outputs = self.bn2(outputs, training=trainable)
        outputs = self.leaky_relu2(outputs, training=trainable)
        outputs = self.deconv2(outputs, training=trainable)
        outputs = self.bn3(outputs, training=trainable)
        outputs = self.leaky_relu3(outputs, training=trainable)
        outputs = self.deconv3(outputs, training=trainable)
        return outputs


class Discriminator(BasedDiscriminator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self):
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(2,2), padding='same', activation=None, kernel_regularizer=self.l2_regularizer)
        self.leaky_relu1 = tf.keras.layers.LeakyReLU()
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(2,2), padding='same', activation=None, kernel_regularizer=self.l2_regularizer)
        self.leaky_relu2 = tf.keras.layers.LeakyReLU()
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.flatten = tf.keras.layers.Flatten()
        self.fc3 = tf.keras.layers.Dense(1, activation=None, kernel_regularizer=self.l2_regularizer)

    @tf.function
    def __call__(self, outputs, trainable=True):
        outputs = self.conv1(outputs, training=trainable)
        outputs = self.leaky_relu1(outputs, training=trainable)
        outputs = self.dropout1(outputs, training=trainable)
        outputs = self.conv2(outputs, training=trainable)
        outputs = self.leaky_relu2(outputs, training=trainable)
        outputs = self.dropout2(outputs, training=trainable)
        outputs = self.flatten(outputs, training=trainable)
        outputs = self.fc3(outputs, training=trainable)
        return outputs

class DCGAN(BasedGAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = 1e-8

    def _build(self):
        self.G = Generator(size=self.size, channel=self.channel, l2_reg=self._l2_reg, l2_reg_scale=self.l2_regularizer)
        self.D = Discriminator(l2_reg=self._l2_reg, l2_reg_scale=self.l2_regularizer)

    def inference(self, inputs, batch_size, labels=None, trainable=True):
        self.z = tf.random.normal([batch_size, self._z_dim], dtype=tf.float32)
        fake_img = self.G(self.combine_distribution(self.z, labels) if self.conditional else self.z, trainable=trainable)
        
        if self.conditional and labels is not None:
            fake_img = self.combine_image(fake_img, labels)
            inputs = self.combine_image(inputs, labels)
            """
            fake_img = self.combine_binary_image(fake_img, labels)
            inputs = self.combine_binary_image(inputs, labels)
            """
        #real_logit = tf.nn.sigmoid(self.D(inputs, trainable=trainable))
        #fake_logit = tf.nn.sigmoid(self.D(fake_img, trainable=trainable))
        real_logit = self.D(inputs, trainable=trainable)
        fake_logit = self.D(fake_img, trainable=trainable)
        return fake_logit, real_logit, fake_img

    def test_inference(self, inputs, batch_size, index=None, trainable=False):
        if self.conditional:
            indices = index if index is not None else np.array([x%self.class_num for x in range(batch_size)],dtype=np.int32)
            labels = tf.one_hot(indices, depth=self.class_num, dtype=tf.float32)
            inputs = self.combine_distribution(inputs, labels)
        return self.G(inputs, trainable=trainable)

    def loss(self, fake_logit, real_logit):
        # discriminator loss
        real_loss = self.cross_entropy(tf.ones_like(real_logit), real_logit)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_logit), fake_logit)
        d_loss = real_loss + fake_loss

        # generator loss
        g_loss = self.cross_entropy(tf.ones_like(fake_logit), fake_logit)
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