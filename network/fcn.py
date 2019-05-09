# -*- coding: utf-8 -*-
import sys
sys.path.append('./utility')
import tensorflow as tf
from model import Model
from optimizer import *

class FCN(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def inference(self, outputs, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            outputs = self.conv1d(outputs, [8, 128, 1, tf.nn.relu])
            outputs = self.conv1d(outputs, [5, 256, 1, tf.nn.relu])
            outputs = self.conv1d(outputs, [3, 128, 1, tf.nn.relu])
            outputs = self.gap(outputs, [self.out_dim])
            #outputs = self.fc(outputs, [self.out_dim, None])
            outputs  = tf.identity(outputs, name="output_logits")
            return outputs