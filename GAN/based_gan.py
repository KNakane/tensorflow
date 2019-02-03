import os,sys
sys.path.append('./network')
sys.path.append('./utility')
import tensorflow as tf
import math
from module import Module
from optimizer import *
from generator import Generator
from discriminator import Discriminator

class BasedGAN(Module):
    def __init__(self,
                 z_dim=100,
                 name='GAN',
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 trainable=False,
                 interval=2
                 ):
        super().__init__(l2_reg=l2_reg,l2_reg_scale=l2_reg_scale, trainable=trainable)
        self._z_dim = z_dim        
        self.name = name
        self.opt = opt
        self.lr = lr
        self.trainable = trainable
        self.gen_train_interval = interval
        self.eps = 1e-14
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

    def loss(self):
        raise NotImplementedError()

    def optimize(self):
        raise NotImplementedError()