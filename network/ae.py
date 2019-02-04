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
            compose_img = self.re_parameterization(self.mu, self.var)
            outputs = tf.clip_by_value(self.Decode(compose_img, reuse), 1e-8, 1 - 1e-8)
            
            return outputs
    
    def re_parameterization(self, mu, var):
        with tf.variable_scope('re_parameterization'):
            std = tf.exp(0.5*var)
            eps = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
            return mu + std * eps

    def loss(self, logits, labels):
        with tf.variable_scope('loss'):
            if len(logits.shape) > 2:
                logits = tf.layers.flatten(logits)
            if len(labels.shape) > 2:
                labels = tf.layers.flatten(labels)
            reconstruct_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
            KL_divergence = -0.5 * tf.reduce_sum(1 + self.var - tf.pow(self.mu,2) - tf.exp(self.var))
            return tf.reduce_mean(reconstruct_loss + KL_divergence)
        