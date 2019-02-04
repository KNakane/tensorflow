import os,sys
sys.path.append('./network')
sys.path.append('./utility')
import tensorflow as tf
import math
from module import Module
from optimizer import *

class Generator(Module):
    def __init__(self, model, l2_reg=False, l2_reg_scale=0.0001, name='Generator', trainable=True):
        super().__init__(l2_reg=l2_reg,l2_reg_scale=l2_reg_scale, trainable=trainable)
        self.model = model
        self.name = name

    def __call__(self, logits):
        with tf.variable_scope(self.name):
            for l in range(len(self.model)):
                logits = (eval('self.' + self.model[l][0])(logits, self.model[l][1:]))
            return logits

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)


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


class BasedGAN(Module):
    def __init__(self,
                 z_dim=100,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 trainable=False
                 ):
        super().__init__(l2_reg=l2_reg,l2_reg_scale=l2_reg_scale, trainable=trainable)
        self._z_dim = z_dim
        self.opt = opt
        self.lr = lr
        self.trainable = trainable
        self.eps = 1e-14
        self.size = 28     # あとで要修正
        self.channel = 1   # あとで要修正
        self.build()

    def conv_out_size_same(self, size, stride):
        return int(math.ceil(float(size) / float(stride)))

    def build(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

    def inference(self):
        assert hasattr(self, 'generator')
        assert hasattr(self, 'discriminator')
        raise NotImplementedError()

    def loss(self, real_logit, fake_logit):
        raise NotImplementedError()

    def optimize(self, d_loss, g_loss):
        raise NotImplementedError()