# -*- coding: utf-8 -*-
import sys
sys.path.append('./utility')
import tensorflow as tf
from model import DNN
from optimizer import *

class UNet(DNN):
    def __init__(self, 
                 model=None,
                 name='U-Net',
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 trainable=False):
        super().__init__(opt=opt,lr=lr,trainable=trainable)
        
    def inference(self, featmap):
        # 
        featmap = tf.layers.conv2d(featmap, 64, [3, 3], activation=tf.nn.relu, padding='same', name="conv1")
        featmap1 = tf.layers.conv2d(featmap, 64, [3, 3], activation=tf.nn.relu, padding='same', name="conv2")
        featmap = tf.layers.max_pooling2d(featmap1, pool_size=2, strides=2,  name="pool1")

        featmap = tf.layers.conv2d(featmap, 128, [3, 3], activation=tf.nn.relu, padding='same', name="conv3")
        featmap2 = tf.layers.conv2d(featmap, 128, [3, 3], activation=tf.nn.relu, padding='same', name="conv4")
        featmap = tf.layers.max_pooling2d(featmap2, pool_size=2, strides=2,  name="pool2")

        featmap = tf.layers.conv2d(featmap, 256, [3, 3], activation=tf.nn.relu, padding='same', name="conv5")
        featmap3 = tf.layers.conv2d(featmap, 256, [3, 3], activation=tf.nn.relu, padding='same', name="conv6")
        featmap = tf.layers.max_pooling2d(featmap3, pool_size=2, strides=2,  name="pool3")

        featmap = tf.layers.conv2d(featmap, 512, [3, 3], activation=tf.nn.relu, padding='same', name="conv7")
        featmap4 = tf.layers.conv2d(featmap, 512, [3, 3], activation=tf.nn.relu, padding='same', name="conv8")
        featmap = tf.layers.max_pooling2d(featmap4, pool_size=2, strides=2,  name="pool4")

        featmap = tf.layers.conv2d(featmap, 1024, [3, 3], activation=tf.nn.relu, padding='same', name="conv9")
        featmap = tf.layers.conv2d(featmap, 1024, [3, 3], activation=tf.nn.relu, padding='same', name="conv10")
        featmap = tf.layers.conv2d_transpose(featmap, 512, strides=[2, 2], kernel_size=[2, 2], activation=tf.nn.relu, padding='same',name="upsamp1")
        featmap = tf.concat([featmap, featmap4], axis=3)

        featmap = tf.layers.conv2d(featmap, 512, [3, 3], activation=tf.nn.relu, padding='same', name="conv11")
        featmap = tf.layers.conv2d(featmap, 512, [3, 3], activation=tf.nn.relu, padding='same', name="conv12")
        featmap = tf.layers.conv2d_transpose(featmap, 256, strides=[2, 2], kernel_size=[2, 2], activation=tf.nn.relu, padding='same',name="upsamp2")
        featmap = tf.concat([featmap, featmap3], axis=3)

        featmap = tf.layers.conv2d(featmap, 256, [3, 3], activation=tf.nn.relu, padding='same', name="conv13")
        featmap = tf.layers.conv2d(featmap, 256, [3, 3], activation=tf.nn.relu, padding='same', name="conv14")
        featmap = tf.layers.conv2d_transpose(featmap, 128, strides=[2, 2], kernel_size=[2, 2], activation=tf.nn.relu, padding='same',name="upsamp3")
        featmap = tf.concat([featmap, featmap2], axis=3)

        featmap = tf.layers.conv2d(featmap, 128, [3, 3], activation=tf.nn.relu, padding='same', name="conv15")
        featmap = tf.layers.conv2d(featmap, 128, [3, 3], activation=tf.nn.relu, padding='same', name="conv16")
        featmap = tf.layers.conv2d_transpose(featmap, 64, strides=[2, 2], kernel_size=[2, 2], activation=tf.nn.relu, padding='same',name="upsamp4")
        featmap = tf.concat([featmap, featmap1], axis=3)

        featmap = tf.layers.conv2d(featmap, 64, [3, 3], activation=tf.nn.relu, padding='same', name="conv17")
        featmap = tf.layers.conv2d(featmap, 64, [3, 3], activation=tf.nn.relu, padding='same', name="conv18")
        featmap = tf.layers.conv2d(featmap, 3, [1, 1], activation=None, padding='same', name="conv19")

        return featmap
