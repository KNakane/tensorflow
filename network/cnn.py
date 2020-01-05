# -*- coding: utf-8 -*-
import os,sys
import tensorflow as tf
from network.model import Model
from utility.optimizer import *

class CNN(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def inference(self, outputs, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            for l in range(len(self.model)):
                outputs = (eval('self.' + self.model[l][0])(outputs, self.model[l][1:]))
            outputs  = tf.identity(outputs, name="output_logits")
            return outputs

    def test_inference(self, outputs, reuse=False):
        return self.inference(outputs, reuse)

    @property
    def variables(self):
        v = []
        for l in self._layers:
            v += l.variables
        return v

    def loss(self, logits, labels):
        if self.name == 'AutoEncoder':
            loss = tf.reduce_mean(tf.square(logits - labels))
        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
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
            correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels, 1))
            return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
