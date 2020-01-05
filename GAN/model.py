import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from utility.optimizer import *

class BasedGenerator(Model):
    def __init__(self,size=28, channel=1, l2_reg=False, l2_reg_scale=0.0001):
        super().__init__()
        self.size = size
        self.channel = channel
        self.l2_regularizer = l2_reg_scale if l2_reg else None
        self._build()

    def _build(self):
        raise NotImplementedError()

    def __call__(self, outputs, trainable=True):
        raise NotImplementedError()

    @property
    def weight(self):
        return


class BasedDiscriminator(Model):
    def __init__(self, l2_reg=False, l2_reg_scale=0.0001):
        super().__init__()
        self.l2_regularizer = l2_reg_scale if l2_reg else None
        self._build()

    def _build(self):
        raise NotImplementedError()

    def __call__(self, outputs, trainable=True):
        raise NotImplementedError()

    @property
    def weight(self):
        return

class BasedGAN(Model):
    def __init__(self,
                 z_dim=100,
                 size=28,
                 channel=1,
                 name='BasedGAN',
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 conditional=False,
                 class_num=10,
                 l2_reg=False,
                 l2_reg_scale=0.0001):
        super().__init__()
        self._z_dim = z_dim
        self.opt = opt
        self.lr = lr
        self.conditional = conditional
        self.size = size
        self.channel = channel
        self._l2_reg = l2_reg
        self.l2_regularizer = l2_reg_scale if l2_reg else None
        self.g_optimizer = eval(opt)(learning_rate=lr*5, decay_step=None)
        self.d_optimizer = eval(opt)(learning_rate=lr, decay_step=None)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        if self.conditional:
            self.class_num = class_num
        self._build()

    def _build(self):
        raise NotImplementedError()

    def inference(self, inputs, batch_size, trainable=True):
        raise NotImplementedError()

    def test_inference(self, inputs, trainable=False):
        raise NotImplementedError()

    def loss(self, real_logit, fake_logit):
        raise NotImplementedError()

    def generator_optimize(self, g_loss, tape=None):
        raise NotImplementedError()

    def discriminator_optimize(self, d_loss, tape=None):
        raise NotImplementedError()

    def combine_distribution(self, z, labels=None):
        """
        latent vector Z と label情報をConcatする
        parameters
        ----------
        z : 一様分布から生成した乱数
        label : labelデータ
        returns
        ----------
        image : labelをconcatしたデータ
        """
        assert labels is not None
        return tf.concat([z, labels], axis=1)

    def combine_image(self, image, labels=None):
        """
        Generatorで生成した画像とlabelをConcatする
        
        parameters
        ----------
        image : Generatorで生成した画像
        label : labelデータ
        returns
        ----------
        image : labelをconcatしたデータ
        """
        assert labels is not None
        labels = tf.reshape(labels, [-1, 1, 1, self.class_num])
        label_image = tf.ones((labels.shape[0], self.size, self.size, self.class_num))
        label_image = tf.multiply(labels, label_image)
        return tf.concat([image, label_image], axis=3)

    def combine_binary_image(self, image, labels=None):
        """
        Generatorで生成した画像とlabelを二進数に変換した画像をConcatする
        
        parameters
        ----------
        image : Generatorで生成した画像
        label : labelデータ
        returns
        ----------
        image : labelをconcatしたデータ
        """
        assert labels is not None
        batch_size = labels.shape[0]
        binary_size = format(self.class_num, 'b')
        binary_size = len([int(x) for x in list(str(binary_size))])
        label_int = tf.cast(tf.argmax(labels, axis=1), tf.int32)
        label_int = tf.cast(tf.mod(tf.bitwise.right_shift(tf.expand_dims(label_int,1), tf.range(binary_size)), 2), tf.float32)
        labels = tf.reshape(label_int, [-1, 1, 1, label_int.shape[-1]])
        label_image = tf.ones((batch_size, self.size, self.size, binary_size))
        label_image = tf.multiply(labels, label_image)
        return tf.concat([image, label_image], axis=3)

    def accuracy(self, real_logit, fake_logit):
        return  (tf.reduce_mean(tf.cast(fake_logit < 0.5, tf.float32)) + tf.reduce_mean(tf.cast(real_logit > 0.5, tf.float32))) / 2.
