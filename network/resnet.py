# -*- coding: utf-8 -*-
# Based on https://github.com/taki0112/ResNet-Tensorflow
# Based on https://github.com/taki0112/ResNeXt-Tensorflow
import sys
sys.path.append('./utility')
import tensorflow as tf
from cnn import CNN
from optimizer import *

class ResNet(CNN):
    def __init__(self, 
                 model=None,
                 name='ResNet',
                 out_dim=10,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 trainable=False):
        super().__init__(name=name, out_dim=out_dim, opt=opt, lr=lr, l2_reg=l2_reg, l2_reg_scale=l2_reg_scale, trainable=trainable)
        #resnet type -> '18, 34, 50, 101, 152'
        self.n_res = 18
        self.filter = 64

        if self.n_res < 50 :
            self.residual_block = self.resblock
        else :
            self.residual_block = self.bottle_resblock

    
    def inference(self, x):
        with tf.variable_scope(self.name):

            logits = x
            return logits

    

    def resblock(self, x, name):
        with tf.variable_scope(name) :
            logits = tf.nn.relu(tf.layers.batch_normalization(x))
            logits = tf.layers.conv2d(logits, self.filter, [3,3], strides=1, use_bias=True)
            logits = tf.nn.relu(tf.layers.batch_normalization(logits))
            logits = tf.layers.conv2d(logits, self.filter, [3,3], strides=1, use_bias=True)
            return logits + x
    
    def bottle_resblock(self):
        pass





class ResNeXt(CNN):
    def __init__(self, 
                 model=None,
                 name='ResNeXt',
                 out_dim=10,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 trainable=False):
        super().__init__(name=name, out_dim=out_dim, opt=opt, lr=lr, l2_reg=l2_reg, l2_reg_scale=l2_reg_scale, trainable=trainable)