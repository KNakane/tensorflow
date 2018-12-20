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

    def inference(self, inputs):
        outputs = tf.reshape(inputs, (-1, 28*28)) / 255.0
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

    def optimize(self, loss, global_step):
        return self.optimizer.minimize(loss=loss, global_step=global_step)

    """
    def optimize(self, loss, clipped_value=1.0):
        grads = self.optimizer.compute_gradients(loss, self.variables)
        clipped_grads = [(tf.clip_by_value(g, -clipped_value, clipped_value), v) for g, v in grads]
        train_op = self.optimizer.apply_gradients(clipped_grads)
        return train_op
    """

    def predict(self, logits):
        _, indices = tf.nn.top_k(logits, 1, sorted=False)
        return indices