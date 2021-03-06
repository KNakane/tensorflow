# -*- coding: utf-8 -*-
import sys
sys.path.append('./utility')
import tensorflow as tf
from model import Model
from optimizer import *

class UNet(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def inference(self, featmap):
        with tf.variable_scope(self.name):

            featmap = self.conv(featmap, [3, 64, 1, tf.nn.relu])
            featmap1 = self.conv(featmap, [3, 64, 1, tf.nn.relu])
            featmap = self.max_pool(featmap1, [2, 2, 'SAME'])

            featmap = self.conv(featmap, [3, 128, 1, tf.nn.relu])
            featmap2 = self.conv(featmap, [3, 128, 1, tf.nn.relu])
            featmap = self.max_pool(featmap2, [2, 2, 'SAME'])

            featmap = self.conv(featmap, [3, 256, 1, tf.nn.relu])
            featmap3 = self.conv(featmap, [3, 256, 1, tf.nn.relu])
            featmap = self.max_pool(featmap3, [2, 2, 'SAME'])

            featmap = self.conv(featmap, [3, 512, 1, tf.nn.relu])
            featmap4 = self.conv(featmap, [3, 512, 1, tf.nn.relu])
            featmap = self.max_pool(featmap4, [2, 2, 'SAME'])

            featmap = self.conv(featmap, [3, 1024, 1, tf.nn.relu])
            featmap = self.conv(featmap, [3, 1024, 1, tf.nn.relu])
            featmap = self.deconv(featmap, [2, 512, 2, tf.nn.relu])
            featmap = tf.concat([featmap, featmap4], axis=3)

            featmap = self.conv(featmap, [3, 512, 1, tf.nn.relu])
            featmap = self.conv(featmap, [3, 512, 1, tf.nn.relu])
            featmap = self.deconv(featmap, [2, 512, 2, tf.nn.relu])
            featmap = tf.concat([featmap, featmap3], axis=3)

            featmap = self.conv(featmap, [3, 256, 1, tf.nn.relu])
            featmap = self.conv(featmap, [3, 256, 1, tf.nn.relu])
            featmap = self.deconv(featmap, [2, 128, 2, tf.nn.relu])
            featmap = tf.concat([featmap, featmap2], axis=3)

            featmap = self.conv(featmap, [3, 128, 1, tf.nn.relu])
            featmap = self.conv(featmap, [3, 128, 1, tf.nn.relu])
            featmap = self.deconv(featmap, [2, 64, 2, tf.nn.relu])
            featmap = tf.concat([featmap, featmap1], axis=3)

            featmap = self.conv(featmap, [3, 64, 1, tf.nn.relu])
            featmap = self.conv(featmap, [3, 64, 1, tf.nn.relu])
            featmap = self.conv(featmap, [1, self.output_dim, 1, None])

            return featmap
        
    def loss(self, logits, labels):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        if self._l2_reg:
            loss += tf.losses.get_regularization_loss()  
        return loss
    
    def optimize(self, loss, global_step=None):
        return self.optimizer.optimize(loss=loss, global_step=global_step)
