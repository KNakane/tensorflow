# -*- coding: utf-8 -*-
import os,sys
import math
import tensorflow as tf
sys.path.append('./utility')
from model import Model
from optimizer import *

class MDN(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi)
        
    def inference(self, outputs, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            outputs = self.fc(outputs, [24, tf.nn.tanh])
            outputs = self.fc(outputs, [24 * 3, None])
            out_pi, out_sigma, out_mu = tf.split(outputs, num_or_size_splits=3, axis=1)
            out_pi = tf.nn.softmax(out_pi)
            out_sigma = tf.exp(out_sigma)
            return [out_pi, out_sigma, out_mu]

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
        result = tf.multiply(result, tf.reciprocal(sigma))
        result = -tf.square(result)/2
        return tf.multiply(tf.exp(result), tf.reciprocal(sigma))*self.oneDivSqrtTwoPI

    def loss(self, logits, labels):
        [out_pi, out_sigma, out_mu] = logits
        loss = self.tf_normal(labels, out_mu, out_sigma)
        loss = tf.multiply(loss, out_pi)
        loss = tf.reduce_sum(loss, 1, keep_dims=True)
        loss = tf.reduce_mean(-tf.log(loss))
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