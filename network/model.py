# -*- coding: utf-8 -*-
import os,sys
import tensorflow as tf
sys.path.append('./utility')
from module import Module
from optimizer import *

class DNN(Module):
    def __init__(self, 
                 model=None,
                 name='DNN',
                 out_dim=10,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 trainable=False
                 ):
        super().__init__(trainable=trainable)
        self.model = model
        self._layers = []
        self.name = name
        self.out_dim = out_dim
        if self._trainable:
            self.optimizer = eval(opt)(learning_rate=lr)

    def inference(self, outputs):
        with tf.variable_scope(self.name):
            for l in range(len(self.model)):
                outputs = (eval('self.' + self.model[l][0])(outputs, self.model[l][1:]))
            outputs  = tf.identity(outputs, name="output_logits")
            return outputs

    @property
    def variables(self):
        v = []
        for l in self._layers:
            v += l.variables
        return v

    def loss(self, logits, labels):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
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
