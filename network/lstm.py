# -*- coding: utf-8 -*-
import os,sys
import tensorflow as tf
sys.path.append('./utility')
from model import Model
from optimizer import *

num_of_input_nodes = 1
num_of_hidden_nodes = 80
num_of_output_nodes = 1
length_of_sequences = 10
size_of_mini_batch = 32
forget_bias = 0.8

class LSTM(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self):
        self._layers.append(tf.keras.layers.LSTM(32))
        self._layers.append(tf.keras.layers.Dense(self.out_dim))
    
    def inference(self, outputs, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            for layer in self._layers:
                outputs = layer(outputs)
        return outputs

    def test_inference(self, outputs, reuse=False):
        return self.inference(outputs, reuse)

    def loss(self, logits, labels):
        #loss = tf.reduce_mean(tf.square(logits - labels))
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
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
            
            """
            correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels, 1))
            return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            """