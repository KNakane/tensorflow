# -*- coding: utf-8 -*-
import sys
sys.path.append('./utility')
import tensorflow as tf
from cnn import CNN
from optimizer import *

class LeNet(CNN):
    def __init__(self, 
                 model=None,
                 name='LeNet',
                 out_dim=10,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 trainable=False):
        super().__init__(name=name,opt=opt,lr=lr,l2_reg=l2_reg,l2_reg_scale=l2_reg_scale,trainable=trainable, out_dim=out_dim)
        
    def inference(self, images, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            featmap = tf.layers.conv2d(images, 32, [5, 5], activation=tf.nn.relu, name="conv1")
            featmap = tf.layers.max_pooling2d(featmap, pool_size=2, strides=2,  name="pool1")
            featmap = tf.layers.conv2d(featmap, 64, [5, 5], activation=tf.nn.relu, name="conv2")
            featmap = tf.layers.max_pooling2d(featmap, pool_size=2, strides=2, name="pool2")

            features = tf.layers.flatten(featmap)
            features = tf.layers.dense(features, units=1024, activation=tf.nn.relu, name="fc3")
            features = tf.layers.dropout(features, rate=0.5, training=self._trainable)
            logits = tf.layers.dense(features, units=self.out_dim, name="fc4")
            logits  = tf.identity(logits, name="output_logits")

            return logits
