# -*- coding: utf-8 -*-
import os,sys
sys.path.append('./utility')
sys.path.append('./network')
import tensorflow as tf
from cnn import CNN
from module import Module
from optimizer import *

class AutoEncoder(CNN):
    def __init__(self, 
                 encode=None,
                 decode=None,
                 name='AutoEncoder',
                 out_dim=10,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 trainable=False
                 ):
        assert encode is not None, "Please set encode model"
        assert decode is not None, "Please set decode model"
        super().__init__(name=name, out_dim=out_dim, opt=opt, lr=lr, trainable=trainable)
        self.encode = encode
        self.decode = decode

    def Encode(self, outputs):
        with tf.variable_scope('Encode'):
            for l in range(len(self.encode)):
                outputs = (eval('self.' + self.encode[l][0])(outputs, self.encode[l][1:]))
            return outputs
    
    def Decode(self, outputs):
        with tf.variable_scope('Decode'):
            for l in range(len(self.decode)):
                outputs = (eval('self.' + self.decode[l][0])(outputs, self.decode[l][1:]))
            return outputs

    def variable(self, outputs):
        return outputs

    def inference(self, outputs):
        with tf.variable_scope(self.name):
            outputs = self.Encode(outputs)
            outputs = self.Decode(outputs)
            return outputs
        
    def loss(self, logits, labels):
        loss = tf.reduce_mean(tf.square(logits - labels))
        return loss


class VAE(AutoEncoder):
    def __init__(self, 
                 encode=None,
                 decode=None,
                 name='VAE',
                 out_dim=10,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 trainable=False
                 ):
        super().__init__(encode=encode, decode=decode, name=name, out_dim=out_dim, opt=opt, lr=lr, trainable=trainable)

    def inference(self, outputs):
        with tf.variable_scope(self.name):
            outputs = self.Encode(outputs)
            outputs = self.Decode(outputs)
            return outputs