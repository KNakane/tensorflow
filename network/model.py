# -*- coding: utf-8 -*-
import os,sys
import tensorflow as tf
sys.path.append(os.path.join(os.path.dirname(__file__), './utility'))
from module import Module
from optimizer import *

class DNN(Module):
    def __init__(self, 
                 model=None,
                 name='DNN',
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 trainable=False):
        super().__init__(trainable=trainable)
        self.model = model
        self._layers = []
        self.name = name
        if self.trainable:
            self.optimizer = eval(opt)(learning_rate=lr)

    def inference(self, outputs):
        with tf.variable_scope(self.name):
            for l in range(len(self.model)):
                outputs = (eval('self.' + self.model[l][0])(outputs, self.model[l][1:]))
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