import os,sys
sys.path.append('./network')
sys.path.append('./utility')
import tensorflow as tf
import math
from module import Module
from optimizer import *
from layers import *

class Generator(Module):
    def __init__(self, model, l2_reg=False, l2_reg_scale=0.0001, name='Generator', trainable=True):
        super().__init__(l2_reg=l2_reg,l2_reg_scale=l2_reg_scale, trainable=trainable)
        self.model = model
        self.name = name
        self.batch_size = 32
        self.img_chan = 1
        self.img_size = 28

    def __call__(self, logits):
        """
        with tf.variable_scope(self.name):
            for l in range(len(self.model)):
                logits = (eval('self.' + self.model[l][0])(logits, self.model[l][1:]))
            return logits
        """
        with tf.variable_scope(self.name):
            with tf.variable_scope("linear"):
                inputs = tf.reshape(tf.nn.relu(fully_connected(logits, 4*4*512)), [self.batch_size, 4, 4, 512])
            with tf.variable_scope("deconv1"):
                inputs = tf.nn.relu(instanceNorm(deconv(inputs, [5, 5, 256, 512], [1, 2, 2, 1], [self.batch_size, 8, 8, 256])))
            with tf.variable_scope("deconv2"):
                inputs = tf.nn.relu(instanceNorm(deconv(inputs, [5, 5, 128, 256], [1, 2, 2, 1], [self.batch_size, 16, 16, 128])))
            with tf.variable_scope("deconv3"):
                inputs = tf.nn.relu(instanceNorm(deconv(inputs, [5, 5, 64, 128], [1, 2, 2, 1], [self.batch_size, 32, 32, 64])))
            with tf.variable_scope("deconv4"):
                stride = 1 if self.img_size <= 32 else 2
                inputs = tf.nn.tanh(deconv(inputs, [5, 5, self.img_chan, 64], [1, stride, stride, 1], [self.batch_size, self.img_size, self.img_size, self.img_chan]))
            return inputs

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
        """
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            for l in range(len(self.model)):
                logits = (eval('self.' + self.model[l][0])(logits, self.model[l][1:]))
            return logits
        """
        enable_sn=False
        with tf.variable_scope(self.name, reuse=reuse):
            with tf.variable_scope("conv1"):
                inputs = tf.nn.leaky_relu(conv(logits, [5, 5, self.img_chan, 64], [1, 2, 2, 1], enable_sn))
            with tf.variable_scope("conv2"):
                inputs = tf.nn.leaky_relu(instanceNorm(conv(inputs, [5, 5, 64, 128], [1, 2, 2, 1], enable_sn)))
            with tf.variable_scope("conv3"):
                inputs = tf.nn.leaky_relu(instanceNorm(conv(inputs, [5, 5, 128, 256], [1, 2, 2, 1], enable_sn)))
            with tf.variable_scope("conv4"):
                inputs = tf.nn.leaky_relu(instanceNorm(conv(inputs, [5, 5, 256, 512], [1, 2, 2, 1], enable_sn)))
            with tf.variable_scope("logits"):
                inputs = tf.layers.flatten(inputs)
            return fully_connected(inputs, 1, enable_sn)

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