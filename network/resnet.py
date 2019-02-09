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

        self.residual_list = [3, 4, 6, 3]
        self.cardinality = 32

    def inference(self, x, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            x = self.conv(x, [3, 64, 1, None])
            x = self.ReLU(self.BN(x, [None]),[None])
            for i in range(self.residual_list[0]):
                x = self.residual_layer(x, 128, name='resblock0_' + str(i))
            for i in range(self.residual_list[1]):
                x = self.residual_layer(x, 256, name='resblock1_' + str(i))
            for i in range(self.residual_list[2]):
                x = self.residual_layer(x, 512, name='resblock2_' + str(i))
            for i in range(self.residual_list[3]):
                x = self.residual_layer(x, 1024, name='resblock3_' + str(i))
            x = self.gap(x,[self.out_dim])
            logits = self.fc(x, [self.out_dim, None])
            return logits

    def residual_layer(self, x, channels, name=None):
        input_channel = x.get_shape().as_list()[-1]

        if input_channel * 2 == channels:
            flag = True
            stride = 2
        elif input_channel == channels:
            flag = False
            stride = 1
        else:
            raise ValueError('Output and input channel does not match in residual blocks!!!')

        with tf.variable_scope(name):
            logits = self.conv(x, [1, channels/2, stride, None])
            logits = self.ReLU(self.BN(logits, [None]),[None])
            """
            Group convolution
            """
            input_list = tf.split(logits, self.cardinality, axis=-1)
            logits_list = []
            for _, input_tensor in enumerate(input_list):
                logits_list.append(self.conv(input_tensor, [3, channels/2, 1, None]))
            logits = tf.concat(logits_list, axis=-1)
            logits = self.ReLU(self.BN(logits, [None]),[None])

            logits = self.conv(logits, [1, channels, 1, None])
            logits = self.BN(logits, [None])

            if flag is True :
                pad_input_x = self.avg_pool(x, [2, 2, 'SAME'])
                pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [0, int(channels/2)]]) # [?, height, width, channel]
            else :
                pad_input_x = x
                
            logits = self.ReLU(logits + pad_input_x, [None])

            return logits
