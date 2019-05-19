import os,sys
sys.path.append('./network')
sys.path.append('./utility')
import tensorflow as tf
import math
import re
from module import Module
from optimizer import *

class Generator(Module):
    def __init__(self, model, l2_reg=False, l2_reg_scale=0.0001, name='Generator', trainable=True):
        super().__init__(l2_reg=l2_reg,l2_reg_scale=l2_reg_scale, trainable=trainable)
        self.model = model
        self.name = name

    def __call__(self, logits, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            for l in range(len(self.model)):
                logits = (eval('self.' + self.model[l][0])(logits, self.model[l][1:]))
            return logits

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

    def loss(self):
        return tf.losses.get_regularization_loss()


class Discriminator(Module):
    def __init__(self, model, l2_reg=False, l2_reg_scale=0.0001, name='Discriminator', trainable=True):
        super().__init__(l2_reg=l2_reg,l2_reg_scale=l2_reg_scale, trainable=trainable)
        self.model = model
        self.name = name

    def __call__(self, logits, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            for l in range(len(self.model)):
                logits = (eval('self.' + self.model[l][0])(logits, self.model[l][1:]))
            return logits

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

    @property
    def weight(self):
        return [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name) if re.search('kernel', v.name)]

    def loss(self):
        return tf.losses.get_regularization_loss()

class Classifier(Module):
    def __init__(self, model, l2_reg=False, l2_reg_scale=0.0001, name='Classifier', trainable=True):
        super().__init__(l2_reg=l2_reg,l2_reg_scale=l2_reg_scale, trainable=trainable)
        self.model = model
        self.name = name

    def __call__(self, logits, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            for l in range(len(self.model)):
                logits = (eval('self.' + self.model[l][0])(logits, self.model[l][1:]))
            return logits

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

    @property
    def weight(self):
        return [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name) if re.search('kernel', v.name)]

    def loss(self):
        return tf.losses.get_regularization_loss()


class BasedGAN(Module):
    def __init__(self,
                 z_dim=100,
                 size=28,
                 channel=1,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 conditional=False,
                 class_num=10,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 trainable=False
                 ):
        super().__init__(l2_reg=l2_reg,l2_reg_scale=l2_reg_scale, trainable=trainable)
        self._z_dim = z_dim
        self.opt = opt
        self.lr = lr
        self.trainable = trainable
        self.conditional = conditional
        self.size = size
        self.channel = channel
        self._l2_reg = l2_reg
        self.l2_reg_scale = l2_reg_scale
        if self._trainable:
            self.g_optimizer = eval(opt)(learning_rate=lr*5)
            self.d_optimizer = eval(opt)(learning_rate=lr)
        if self.conditional:
            self.class_num = class_num
        self.build()

    def conv_out_size_same(self, size, stride):
        return int(math.ceil(float(size) / float(stride)))

    def build(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

    def inference(self, inputs, batch_size):
        assert hasattr(self, 'generator')
        assert hasattr(self, 'discriminator')
        raise NotImplementedError()

    def loss(self, real_logit, fake_logit):
        raise NotImplementedError()

    def evaluate(self, real_logit, fake_logit):
        with tf.variable_scope('Accuracy'):
            return  (tf.reduce_mean(tf.cast(fake_logit < 0.5, tf.float32)) + tf.reduce_mean(tf.cast(real_logit > 0.5, tf.float32))) / 2.

    def optimize(self, d_loss, g_loss, global_step=None):
        with tf.variable_scope('Optimizer'):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                opt_D = self.d_optimizer.optimize(loss=d_loss, global_step=global_step, var_list=self.D.var)
                opt_G = self.g_optimizer.optimize(loss=g_loss, global_step=global_step, var_list=self.G.var)
                return opt_D, opt_G

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