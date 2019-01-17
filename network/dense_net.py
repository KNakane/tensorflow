# -*- coding: utf-8 -*-
# Based on https://github.com/taki0112/Densenet-Tensorflow
import sys
sys.path.append('./utility')
import tensorflow as tf
from cnn import CNN
from optimizer import *

class DenseNet(CNN):
    def __init__(self, 
                 model=None,
                 name='DenseNet',
                 out_dim=10,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 trainable=False):
        super().__init__(name=name,opt=opt,lr=lr,trainable=trainable, out_dim=out_dim)

    def inference(self, images, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            featmap = tf.layers.conv2d(images, 32, [7, 7], (2,2), activation=tf.nn.relu, name="init_conv")
            featmap = self.dense_block(featmap, False, 'dense_1')
    
    def dense_block(self, x, n_layers, bottle_neck, name):
        with tf.name_scope(name):
            layers_concat = list()
            layers_concat.append(x)
            for i in range(n_layers):
                if bottle_neck:
                    x = tf.layers.batch_normalization(inputs=x,trainable=self._trainable,name='BN{}'.format(index),reuse=self._reuse)
                    x = tf.nn.relu(x)
                    x = tf.layers.conv2d(x, 32, [1, 1], activation=None, name="conv")
                    x = tf.layers.batch_normalization(inputs=x,trainable=self._trainable,name='BN{}'.format(index),reuse=self._reuse)
                    x = tf.nn.relu(x)
                    x = tf.layers.conv2d(x, 32, [3, 3], activation=None, name="conv")

                else:
                    x = tf.layers.batch_normalization(inputs=x,trainable=self._trainable,name='BN{}'.format(index),reuse=self._reuse)
                    x = tf.nn.relu(x)
                    x = tf.layers.conv2d(x, 32, [3, 3], activation=None, name="init_conv")
                layers_concat.append(x)
                x = tf.concat(layers_concat, axis=3)
        return x

    def transition(self, x):
        x = tf.layers.batch_normalization(inputs=x,trainable=self._trainable,name='BN{}'.format(index),reuse=self._reuse)
        x = tf.layers.conv2d(x, 32, [1, 1], activation=None, name="conv")
        x = tf.layers.average_pooling2d(x, pool_size=[2,2], strides=2)
        return x