# -*- coding: utf-8 -*-
import os,sys
import tensorflow as tf
sys.path.append('./utility')
from module import Module
from optimizer import *

num_of_input_nodes = 1
num_of_hidden_nodes = 80
num_of_output_nodes = 1
length_of_sequences = 10
size_of_mini_batch = 32
forget_bias = 0.8

class LSTM(Module):
    def __init__(self, 
                 model=None,
                 name='LSTM',
                 out_dim=10,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 trainable=False
                 ):
        super().__init__(l2_reg=l2_reg,l2_reg_scale=l2_reg_scale, trainable=trainable)
        self.model = model
        self.name = name
        self.out_dim = out_dim
        if self._trainable:
            self.optimizer = eval(opt)(learning_rate=lr)

        self.weight1_var = tf.Variable(tf.truncated_normal(
                [num_of_input_nodes, num_of_hidden_nodes], stddev=0.1), name="weight1")
        self.weight2_var = tf.Variable(tf.truncated_normal(
            [num_of_hidden_nodes, num_of_output_nodes], stddev=0.1), name="weight2")
        self.bias1_var = tf.Variable(tf.truncated_normal([num_of_hidden_nodes], stddev=0.1), name="bias1")
        self.bias2_var = tf.Variable(tf.truncated_normal([num_of_output_nodes], stddev=0.1), name="bias2")

    
    def inference(self, outputs, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            in1 = tf.transpose(outputs, [1, 0, 2])
            in2 = tf.reshape(in1, [-1, num_of_input_nodes])
            in3 = tf.matmul(in2, self.weight1_var) + self.bias1_var
            in4 = tf.split(in3, length_of_sequences, 0)

            cell = tf.nn.rnn_cell.LSTMCell(num_of_hidden_nodes, forget_bias=forget_bias, state_is_tuple=True)
            rnn_output, _ = tf.contrib.rnn.static_rnn(cell, in4, dtype=tf.float32)#, initial_state=tf.zeros(shape=(size_of_mini_batch, num_of_hidden_nodes * 2)))
            outputs = tf.matmul(rnn_output[-1], self.weight2_var) + self.bias2_var
        return outputs
        """
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            for l in range(len(self.model)):
                outputs = (eval('self.' + self.model[l][0])(outputs, self.model[l][1:]))
            outputs  = tf.identity(outputs, name="output_logits")
            return outputs
        """

    def test_inference(self, outputs, reuse=False):
        return self.inference(outputs, reuse)

    def loss(self, logits, labels):
        loss = tf.reduce_mean(tf.square(logits - labels))
        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
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
             tf.abs(logits - labels)
            
            """
            correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels, 1))
            return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            """