# -*- coding: utf-8 -*-
# Based on https://github.com/taki0112/Densenet-Tensorflow
import tensorflow as tf
from network.model import Model
from utility.optimizer import *

class DenseNet(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.growth_k = 12
        self.nb_blocks = 2

    def inference(self, images, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            featmap = self.conv(images, [7, self.growth_k * 2, 2, None], name="init_conv")
            featmap = self.max_pool(featmap, [3, 2, 'VALID'])
            for i in range(self.nb_blocks) :
                featmap = self.dense_block(featmap, n_layers=4, bottle_neck=True, name='dense_'+str(i))
                featmap = self.transition_layer(featmap, name='trans_'+str(i))

            featmap = self.dense_block(featmap, n_layers=31, bottle_neck=True, name='dense_final')
            featmap = self.ReLU(self.BN(featmap, [None]),[None])
            featmap = self.gap(featmap, [self.out_dim])
            featmap = self.fc(featmap, [self.out_dim,None])
            logits  = tf.identity(featmap, name="output_logits")

            return logits
    
    def dense_block(self, x, n_layers, bottle_neck, name):
        with tf.variable_scope(name):
            layers_concat = list()
            layers_concat.append(x)

            for _ in range(n_layers):
                if bottle_neck:
                    logits = self.ReLU(self.BN(x, [None]),[None])
                    logits = self.conv(logits, [1, self.growth_k * 4, 1, None])
                    logits = tf.layers.dropout(inputs=logits, rate=0.2, training=self._trainable)
                    logits = self.ReLU(self.BN(logits, [None]),[None])
                    logits = self.conv(logits, [3, self.growth_k, 1, None])
                    logits = tf.layers.dropout(inputs=logits, rate=0.2, training=self._trainable)

                else:
                    logits = self.ReLU(self.BN(x, [None]),[None])
                    logits = self.conv(logits, [3, self.growth_k * 4, 1, None])
                    logits = tf.layers.dropout(inputs=logits, rate=0.2, training=self._trainable)

                layers_concat.append(logits)
                x = tf.concat(layers_concat, axis=3)
            return x

    def transition_layer(self, x, name):
        with tf.variable_scope(name):
            x = self.ReLU(self.BN(x, [None]),[None])
            in_channel = x.get_shape().as_list()[3]
            x = self.conv(x, [1, in_channel * 0.5, 1, None])
            x = tf.layers.dropout(inputs=x, rate=0.2, training=self._trainable)
            x = self.avg_pool(x, [2, 2, 'VALID'])
            return x
