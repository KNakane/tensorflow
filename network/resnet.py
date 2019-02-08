# -*- coding: utf-8 -*-
# Based on https://github.com/taki0112/ResNet-Tensorflow
# Based on https://github.com/taki0112/ResNeXt-Tensorflow
import sys
sys.path.append('./utility')
import tensorflow as tf
from cnn import CNN
from optimizer import *

class ResNet(CNN):
    def __init__(self, 
                 model=None,
                 name='ResNet',
                 out_dim=10,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 trainable=False):
        super().__init__(name=name, out_dim=out_dim, opt=opt, lr=lr, l2_reg=l2_reg, l2_reg_scale=l2_reg_scale, trainable=trainable)
        #resnet type -> '18, 34, 50, 101, 152'
        self.n_res = 18
        self.filter = 64

        if self.n_res < 50 :
            self.residual_block = self.resblock
        else :
            self.residual_block = self.bottle_resblock

        self.residual_list = self.get_residual_layer()

    
    def inference(self, x, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            logits = self.conv(x, [3, self.filter, 1, None])
            for i in range(self.residual_list[0]):
                x = self.residual_block(x, channels=self.filter, downsample=False, name='resblock0_' + str(i))
            x = self.residual_block(x, channels=self.filter*2, downsample=True, name='resblock1_0')
            for i in range(1, self.residual_list[1]) :
                x = self.residual_block(x, channels=self.filter*2, downsample=False, name='resblock1_' + str(i))
            x = self.residual_block(x, channels=self.filter*4, downsample=True, name='resblock2_0')
            for i in range(1, self.residual_list[2]) :
                x = self.residual_block(x, channels=self.filter*4, downsample=False, name='resblock2_' + str(i))
            x = self.residual_block(x, channels=self.filter*8, downsample=True, name='resblock_3_0')
            for i in range(1, self.residual_list[3]) :
                x = self.residual_block(x, channels=self.filter*8, downsample=False, name='resblock_3_' + str(i))
            x = self.ReLU(self.BN(x, [None]),[None])
            x = self.gap(x,[self.out_dim])
            logits = self.fc(x, [self.out_dim, None])
            return logits

    def resblock(self, x, channels, downsample=False, name=None):
        with tf.variable_scope(name):
            logits = self.ReLU(self.BN(x, [None]),[None])
            if downsample:
                logits = self.conv(logits, [3, channels, 2, None])
                x = self.conv(x, [1, channels, 2, None])
            else:
                logits = self.conv(logits, [3, channels, 1, None])
            logits = self.ReLU(self.BN(logits, [None]),[None])
            logits = self.conv(logits, [3, channels, 1, None])
            return logits + x
    
    def bottle_resblock(self):
        pass

    def get_residual_layer(self) :
        if self.n_res == 18:
            return [2, 2, 2, 2]

        elif self.n_res == 34:
            return [3, 4, 6, 3]

        elif self.n_res == 50:
            return [3, 4, 6, 3]

        elif self.n_res == 101:
            return [3, 4, 23, 3]

        elif self.n_res == 152:
            return [3, 8, 36, 3]





class ResNeXt(CNN):
    def __init__(self, 
                 model=None,
                 name='ResNeXt',
                 out_dim=10,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 trainable=False):
        super().__init__(name=name, out_dim=out_dim, opt=opt, lr=lr, l2_reg=l2_reg, l2_reg_scale=l2_reg_scale, trainable=trainable)