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

    def Encode(self, outputs, reuse=False):
        with tf.variable_scope('Encode'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            for l in range(len(self.encode)):
                outputs = (eval('self.' + self.encode[l][0])(outputs, self.encode[l][1:]))
            return outputs
    
    def Decode(self, outputs, reuse=False):
        with tf.variable_scope('Decode'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            for l in range(len(self.decode)):
                outputs = (eval('self.' + self.decode[l][0])(outputs, self.decode[l][1:]))
            return outputs

    def variable(self, outputs):
        return outputs

    def inference(self, outputs, reuse=False):
        with tf.variable_scope(self.name):
            outputs = self.Encode(outputs, reuse)
            outputs = self.Decode(outputs, reuse)
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

    def Encode(self, outputs, reuse=False):
        with tf.variable_scope('Encode'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            for l in range(len(self.encode)):
                outputs = (eval('self.' + self.encode[l][0])(outputs, self.encode[l][1:]))
            mu, var = tf.split(outputs, num_or_size_splits=2, axis=1)
            return mu, var

    def inference(self, outputs, reuse=False):
        with tf.variable_scope(self.name):
            self.mu, self.var = self.Encode(outputs, reuse)
            outputs = self.Decode(self.re_parameterization(self.mu, self.var), reuse)
            return outputs
    
    def re_parameterization(self, mu, var):
        return mu + tf.exp(var * .5) * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    def loss(self, logits, labels):
        KL_divergence = 0.5 * tf.reduce_sum(tf.square(self.mu) + tf.square(self.var) - tf.log(1e-8 + tf.square(self.var)) - 1, axis=1)
        loss = tf.reduce_sum(labels * tf.log(logits) + (1 - labels) * tf.log(1 - logits), 1)
        return -loss + KL_divergence
        """
        #loss = - tf.reduce_sum(labels * tf.log(logits) + (1.-self.x) * tf.log( tf.clip_by_value(1.-self.y,1e-20,1e+20)))
        KL_divergence = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(self.mu) + tf.square(self.var) - tf.log(1e-8 + tf.square(self.var)) - 1, 1))
        #loss = tf.reduce_mean(tf.square(logits - labels))
        loss = tf.reduce_mean(tf.reduce_sum(labels * tf.log(tf.clip_by_value(logits, 1e-10,1.0)) + (1 - labels) * tf.log(1 - tf.clip_by_value(logits, 1e-10,1.0)),1))
        return KL_divergence - loss 
        """