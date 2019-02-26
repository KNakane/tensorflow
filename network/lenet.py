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

class VGG(CNN):
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
            featmap = self.conv(images, [3, 64, 1, tf.nn.relu], 'conv1')
            featmap = self.conv(featmap, [3, 64, 1, tf.nn.relu], 'conv2')
            featmap = self.max_pool(featmap, [2, 2, 'SAME'])

            featmap = self.conv(images, [3, 128, 1, tf.nn.relu], 'conv3')
            featmap = self.conv(featmap, [3, 128, 1, tf.nn.relu], 'conv4')
            featmap = self.max_pool(featmap, [2, 2, 'SAME'])

            featmap = self.conv(images, [3, 256, 1, tf.nn.relu], 'conv5')
            featmap = self.conv(featmap, [3, 256, 1, tf.nn.relu], 'conv6')
            featmap = self.conv(featmap, [3, 256, 1, tf.nn.relu], 'conv7')
            featmap = self.max_pool(featmap, [2, 2, 'SAME'])

            featmap = self.conv(images, [3, 512, 1, tf.nn.relu], 'conv8')
            featmap = self.conv(featmap, [3, 512, 1, tf.nn.relu], 'conv9')
            featmap = self.conv(featmap, [3, 512, 1, tf.nn.relu], 'conv10')
            featmap = self.max_pool(featmap, [2, 2, 'SAME'])

            featmap = self.conv(images, [3, 512, 1, tf.nn.relu], 'conv11')
            featmap = self.conv(featmap, [3, 512, 1, tf.nn.relu], 'conv12')
            featmap = self.conv(featmap, [3, 512, 1, tf.nn.relu], 'conv13')
            featmap = self.max_pool(featmap, [2, 2, 'SAME'])

            featmap = self.fc(featmap, [4096, tf.nn.relu])
            featmap = self.fc(featmap, [4096, tf.nn.relu])
            featmap = self.fc(featmap, [self.out_dim, tf.nn.relu])
            logits  = tf.identity(featmap, name="output_logits")
            return logits
