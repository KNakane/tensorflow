import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from GAN.model import BasedGenerator
from GAN.dcgan import DCGAN, Generator
from AutoEncoder.encoder_decoder import Encoder, Decoder, Conv_Decoder, Conv_Encoder

class Discriminator():
    def __init__(self, input_shape=None, l2_reg=False, l2_reg_scale=0.0001):
        super().__init__()
        self.__shape = input_shape
        self.__outdim = 100
        self.__l2_reg = l2_reg
        self.__l2_reg_scale = 0.0001
        (self.size, self.size, self.channel) = input_shape
        self._build()

    def _build(self):
        # Encoder
        self.encode = Encoder(input_shape=self.__shape, out_dim=self.__outdim, l2_reg=self.__l2_reg, l2_reg_scale=self.__l2_reg_scale)

        # Decoder
        self.decode = Decoder(input_shape=(self.__outdim,), size=self.size, channel=self.channel, l2_reg=self.__l2_reg, l2_reg_scale=self.__l2_reg_scale)
        return

    def __call__(self, outputs, trainable=True):
        # Encoder
        outputs = self.encode(outputs, trainable=trainable)

        # Decoder
        outputs = self.decode(outputs, trainable=trainable)
        return outputs

class BEGAN(DCGAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kappa = tf.Variable(initial_value=0., trainable=False, dtype=tf.float32, name='alpha')
        self.gamma = tf.constant(0.5)
        self.lambda_k = tf.constant(0.001)

    def _build(self):
        self.G = Generator(input_shape=(self._z_dim + (self.conditional * self.class_num),),
                           size=self.size,
                           channel=self.channel,
                           l2_reg=self._l2_reg,
                           l2_reg_scale=self.l2_regularizer)
        
        self.D = Discriminator(input_shape=(self.size, self.size, self.channel),
                               l2_reg=self._l2_reg,
                               l2_reg_scale=self.l2_regularizer)
        return

    def inference_discriminator(self, inputs, labels=None, trainable=True):
        logits = self.D(inputs, trainable=trainable)
        return logits

    def generator_loss(self, fake_logit, inputs):
        g_loss = tf.reduce_mean(tf.abs(fake_logit - inputs))
        return g_loss

    def discriminator_loss(self, fake_logit, real_logit, inputs):
        real_loss = tf.reduce_mean(tf.abs(real_logit - inputs))
        fake_loss = tf.reduce_mean(tf.abs(fake_logit - inputs))
        
        d_loss = real_loss - self.kappa * fake_loss
        self.update_kappa(real_loss, fake_loss)
        return d_loss

    def discriminator_optimize(self, d_loss, tape=None):
        assert tape is not None, 'please set tape in opmize'
        d_grads = tape.gradient(d_loss, self.D.encode.trainable_variables + self.D.decode.trainable_variables)
        self.d_optimizer.method.apply_gradients(zip(d_grads, self.D.encode.trainable_variables + self.D.decode.trainable_variables))
        return

    def update_kappa(self, real_loss, fake_loss):
        self.kappa.assign(tf.clip_by_value(self.kappa + self.lambda_k * (self.gamma * real_loss - fake_loss), 0, 1))
        return