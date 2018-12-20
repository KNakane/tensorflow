# -*- coding: utf-8 -*-
import tensorflow as tf
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
        self.built = False
        self.optimizer = opt(learning_rate=lr).method
        self.trainable = trainable

    def build(self):
        for l in range(len(self.model)):
            self._layers.append(eval('self.'+self.model[l][0])(self.model[l][1:]))

        self.built = True

    @property
    def variables(self):
        v = []
        for l in self._layers:
            v += l.variables
        return v

    def call(self, inputs, is_train=True):
        outputs = tf.reshape(inputs, (-1, 28*28)) / 255.0
        for i in range(len(self._layers)):
            outputs = self._layers[i](outputs)
        return outputs

    def loss(self, logits, labels):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        return loss

    def optimize(self, loss, clipped_value=1.0):
        grads = self.optimizer.compute_gradients(loss, self.variables)
        clipped_grads = [(tf.clip_by_value(g, -clipped_value, clipped_value), v) for g, v in grads]
        train_op = self.optimizer.apply_gradients(clipped_grads)
        return train_op

    def predict(self, logits):
        _, indices = tf.nn.top_k(logits, 1, sorted=False)
        return indices

    def __call__(self, inputs, **kwargs):
        assert self.model is not None, 'Please confirm your model'
        with tf.variable_scope(self.name):
            if not self.built:
                self.build()
            return self.call(inputs, **kwargs)