# -*- coding: utf-8 -*-
import os,sys
import math
import tensorflow as tf
sys.path.append('./utility')
from module import Module
from optimizer import *

class MDN(Module):
    def __init__(self, 
                 model=None,
                 name='MDN',
                 out_dim=10,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 trainable=False
                 ):
        super().__init__(l2_reg=l2_reg,l2_reg_scale=l2_reg_scale, trainable=trainable)
        self.model = model
        self._layers = []
        self.name = name
        self.out_dim = out_dim
        self.oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi)
        if self._trainable:
            self.optimizer = eval(opt)(learning_rate=lr)

    def inference(self, outputs, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            outputs = self.fc(outputs, [24, tf.nn.tanh])
            outputs = self.fc([24 * 3, None])
            return outputs

    def test_inference(self, outputs, reuse=False):
        return self.inference(outputs, reuse)

    @property
    def variables(self):
        v = []
        for l in self._layers:
            v += l.variables
        return v

    def tf_normal(self, y, mu, sigma):
        result = tf.subtract(y, mu)
        result = tf.multiply(result,tf.matrix_inverse(sigma))
        result = tf.negative(tf.square(result)/2)
        return tf.matmul(tf.exp(result),tf.matrix_inverse(sigma)) * self.oneDivSqrtTwoPI

    def loss(self, logits, labels):
        out_pi, out_sigma, out_mu = logits
        loss = self.tf_normal(labels, out_mu, out_sigma)
        loss = tf.multiply(loss, out_pi)
        loss = tf.reduce_sum(loss, 1, keep_dims=True)
        loss = -tf.log(loss)
        if self._l2_reg:
            loss += tf.losses.get_regularization_loss()  
        return loss

    def optimize(self, loss, global_step=None):
        return self.optimizer.optimize(loss=loss, global_step=global_step)

    def predict(self, logits):
        _, indices = tf.nn.top_k(logits, 1, sorted=False)
        return indices

    def evaluate(self, logits, labels):
        with tf.variable_scope('Accuracy'):
            return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(labels, logits))))