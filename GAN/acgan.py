import os, sys
import numpy as np
import tensorflow as tf
from GAN.model import BasedDiscriminator, BasedGenerator
from GAN.dcgan import DCGAN, Generator

class Discriminator(BasedDiscriminator):
    def __init__(self, *args, **kwargs):
        self.class_num = kwargs.pop('class_num')
        super().__init__(*args, **kwargs)

    def _build(self):
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(2,2), padding='same', activation=None, kernel_regularizer=self.l2_regularizer)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.leaky_relu1 = tf.keras.layers.LeakyReLU()
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(2,2), padding='same', activation=None, kernel_regularizer=self.l2_regularizer)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.leaky_relu2 = tf.keras.layers.LeakyReLU()
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.flatten = tf.keras.layers.Flatten()
        self.fc3 = tf.keras.layers.Dense(1 + self.class_num, activation=None, kernel_regularizer=self.l2_regularizer)

    def __call__(self, outputs, trainable=True):
        outputs = self.conv1(outputs, training=trainable)
        outputs = self.bn1(outputs, training=trainable)
        outputs = self.leaky_relu1(outputs, training=trainable)
        outputs = self.conv2(outputs, training=trainable)
        outputs = self.bn2(outputs, training=trainable)
        outputs = self.leaky_relu2(outputs, training=trainable)
        outputs = self.flatten(outputs, training=trainable)
        outputs = self.fc3(outputs, training=trainable)
        return outputs

class ACGAN(DCGAN):
    """
    Auxiliary Classifier Generative Adversarial Network
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = tf.losses.CategoricalCrossentropy()

    def _build(self):
        self.G = Generator(input_shape=(self._z_dim + self.class_num,),
                           size=self.size,
                           channel=self.channel,
                           l2_reg=self._l2_reg,
                           l2_reg_scale=self.l2_regularizer)
        self.D = Discriminator(input_shape=(self.size, self.size, self.channel),
                               class_num=self.class_num,
                               l2_reg=self._l2_reg,
                               l2_reg_scale=self.l2_regularizer)

    def inference(self, inputs, batch_size, labels=None, trainable=True):
        assert labels is not None
        self.z = tf.random.normal([batch_size, self._z_dim], dtype=tf.float32)
        fake_img = self.G(self.combine_distribution(self.z, labels))

        real_logit, real_recognition = tf.split(self.D(inputs, trainable=trainable),[1, self.class_num], 1)
        fake_logit, fake_recognition = tf.split(self.D(fake_img, trainable=trainable), [1, self.class_num], 1)

        return (fake_logit, fake_recognition), (real_logit, real_recognition), fake_img

    def inference_generator(self, batch_size, labels=None, trainable=True):
        assert labels is not None
        self.z = tf.random.normal((batch_size, self._z_dim), dtype=tf.float32)
        fake_img = self.G(self.combine_distribution(self.z, labels))
        return fake_img

    def inference_discriminator(self, inputs, labels=None, trainable=True):
        logits, recognition = tf.split(self.D(inputs, trainable=trainable),[1, self.class_num], 1)
        recognition = tf.keras.layers.Softmax()(recognition)
        return (logits, recognition)

    def test_inference(self, inputs, batch_size, index=None, trainable=False):
        indices = index if index is not None else np.array([x%self.class_num for x in range(batch_size)],dtype=np.int32)
        labels = tf.one_hot(indices, depth=self.class_num, dtype=tf.float32)
        inputs = self.combine_distribution(inputs, labels)
        return self.G(inputs, trainable=trainable)

    def generator_loss(self, fake_logit, labels):
        (fake_logit, fake_recognition) = fake_logit
        g_loss = self.cross_entropy(tf.ones_like(fake_logit), fake_logit)
        fake_recog_loss = self.loss_function(y_true=labels, y_pred=fake_recognition)

        g_loss = g_loss + fake_recog_loss
        return g_loss

    def discriminator_loss(self, fake_logit, real_logit, labels):
        (fake_logit, fake_recognition) = fake_logit
        (real_logit, real_recognition) = real_logit

        # discriminator loss
        real_loss = self.cross_entropy(tf.ones_like(real_logit), real_logit)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_logit), fake_logit)
        d_loss = real_loss + fake_loss

        # Image Recognition
        fake_recog_loss = self.loss_function(y_true=labels, y_pred=fake_recognition)
        real_recog_loss = self.loss_function(y_true=labels, y_pred=real_recognition)

        d_loss = d_loss + fake_recog_loss + real_recog_loss
        return d_loss

    def accuracy(self, real_logit, fake_logit):
        (fake_logit, _) = fake_logit
        (real_logit, _) = real_logit
        return  (tf.reduce_mean(tf.cast(fake_logit < 0.5, tf.float32)) + tf.reduce_mean(tf.cast(real_logit > 0.5, tf.float32))) / 2.