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
            featmap = self.conv(images, [7, self.growth_k * 2, 2, tf.nn.relu], name="init_conv")
            featmap = self.max_pool(featmap, [3, 2, 'VALID'])
            for i in range(self.nb_blocks) :
                featmap = self.dense_block(featmap, n_layers=4, bottle_neck=True, name='dense_'+str(i))
                featmap = self.transition_layer(featmap, name='trans_'+str(i))

            featmap = self.dense_block(featmap, n_layers=31, bottle_neck=True, name='dense_final')
            featmap = self.ReLU(self.BN(featmap, [None]),[None])
            logits = self.fc(featmap, [self.out_dim,None])

            return logits
    
    def dense_block(self, x, n_layers, bottle_neck, name):
        with tf.variable_scope(name):
            layers_concat = list()
            layers_concat.append(x)

            for _ in range(n_layers):
                if bottle_neck:
                    logits = self.ReLU(self.BN(x, [None]),[None])
                    logits = self.conv(logits, [1, self.growth_k * 4, 1, None])
                    logits = self.ReLU(self.BN(logits, [None]),[None])
                    logits = self.conv(logits, [3, self.growth_k, 1, None])

                else:
                    logits = self.ReLU(self.BN(x, [None]),[None])
                    logits = self.conv(logits, [3, self.growth_k * 4, 1, None])
                layers_concat.append(logits)
                x = tf.concat(layers_concat, axis=3)
            return x

    def transition_layer(self, x, name):
        with tf.variable_scope(name):
            x = self.ReLU(self.BN(x, [None]),[None])
            in_channel = x.get_shape().as_list()[3]
            x = self.conv(x, [1, in_channel * 0.5, 1, None])
            x = self.avg_pool(x, [2, 2, 'VALID'])
            return x
