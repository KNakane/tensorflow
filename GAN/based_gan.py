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
        self.batch_size = 32
        self.img_chan = 1
        self.img_size = 28

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


class Discriminator(Module):
    def __init__(self, model, l2_reg=False, l2_reg_scale=0.0001, name='Discriminator', trainable=True):
        super().__init__(l2_reg=l2_reg,l2_reg_scale=l2_reg_scale, trainable=trainable)
        self.model = model
        self.name = name
        self.img_chan = 1

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
                 size=28,
                 channel=1,
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
        self.size = size
        self.channel = channel
        self.build()
        if self._trainable:
            self.optimizer = eval(opt)(learning_rate=lr)

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

    def optimize(self, d_loss, g_loss, global_step=None):
        opt_D = self.optimizer.optimize(loss=d_loss, global_step=global_step, var_list=self.D.var)
        opt_G = self.optimizer.optimize(loss=g_loss, global_step=global_step, var_list=self.G.var)
        return opt_D, opt_G