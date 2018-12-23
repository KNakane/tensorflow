# -*- coding: utf-8 -*-
import sys
sys.path.append('./utility')
import tensorflow as tf
from model import DNN
from optimizer import *

class LeNet(DNN):
    def __init__(self, 
                 model=None,
                 name='LeNet',
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 trainable=False):
        super().__init__(opt=opt,lr=lr,trainable=trainable)
        
    def inference(self, images):
        featmap = tf.layers.conv2d(images, 32, [5, 5], activation=tf.nn.relu, name="conv1")
        featmap = tf.layers.max_pooling2d(featmap, pool_size=2, strides=2,  name="pool1")
        featmap = tf.layers.conv2d(featmap, 64, [5, 5], activation=tf.nn.relu, name="conv2")
        featmap = tf.layers.max_pooling2d(featmap, pool_size=2, strides=2, name="pool2")

        features = tf.layers.flatten(featmap)
        features = tf.layers.dense(features, units=1024, activation=tf.nn.relu, name="fc3")
        features = tf.layers.dropout(features, rate=0.5, training=self.trainable)
        logits = tf.layers.dense(features, units=10, name="fc4")

        return logits
