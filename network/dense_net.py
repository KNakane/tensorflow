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
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 trainable=False):
        super().__init__(name=name, out_dim=out_dim, opt=opt, lr=lr, l2_reg=l2_reg, l2_reg_scale=l2_reg_scale, trainable=trainable)
        self.growth_k = 12
        self.nb_blocks = 2

    def inference(self, images, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            featmap = tf.layers.conv2d(images, self.growth_k * 2, [7, 7], strides=2, activation=tf.nn.relu, name="init_conv")
            featmap = tf.layers.max_pooling2d(featmap, pool_size=[3,3], strides=2, padding='VALID')
            for i in range(self.nb_blocks) :
                featmap = self.dense_block(featmap, n_layers=4, bottle_neck=True, name='dense_'+str(i))
                featmap = self.transition_layer(featmap, name='trans_'+str(i))

            featmap = self.dense_block(featmap, n_layers=31, bottle_neck=True, name='dense_final')
            featmap = tf.nn.relu(tf.layers.batch_normalization(inputs=featmap,trainable=self._trainable, name='BN1'))
            featmap = tf.layers.flatten(featmap, name='flatten')
            logits = tf.layers.dense(inputs=featmap, units=self.out_dim, activation=None, use_bias=True)

            return logits
    
    def dense_block(self, x, n_layers, bottle_neck, name):
        with tf.variable_scope(name):
            layers_concat = list()
            layers_concat.append(x)

            for index in range(n_layers):
                if bottle_neck:
                    x = tf.layers.batch_normalization(inputs=x,trainable=self._trainable,name='BN{}_1'.format(index))
                    x = tf.nn.relu(x)
                    x = tf.layers.conv2d(x, self.growth_k * 4, [1, 1], strides=1, padding='SAME', activation=None, name="conv{}_1".format(index))
                    x = tf.layers.batch_normalization(inputs=x,trainable=self._trainable,name='BN{}_2'.format(index))
                    x = tf.nn.relu(x)
                    x = tf.layers.conv2d(x, self.growth_k, [3, 3], strides=1, padding='SAME', activation=None, name="conv{}_2".format(index))

                else:
                    x = tf.layers.batch_normalization(inputs=x,trainable=self._trainable,name='BN{}_1'.format(index))
                    x = tf.nn.relu(x)
                    x = tf.layers.conv2d(x, self.growth_k * 4, [3, 3], activation=None, name="conv1")
                layers_concat.append(x)
                x = tf.concat(layers_concat, axis=3)
        return x

    def transition_layer(self, x, name):
        with tf.variable_scope(name):
            x = tf.nn.relu(tf.layers.batch_normalization(inputs=x,trainable=self._trainable,name='BN_1'))
            in_channel = x.get_shape().as_list()[3]
            x = tf.layers.conv2d(x, in_channel * 0.5, [1, 1], activation=None, name="conv")
            x = tf.layers.average_pooling2d(x, pool_size=[2,2], strides=2, padding='VALID')
            return x
