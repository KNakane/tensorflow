# -*- coding: utf-8 -*-
# Based on https://github.com/taki0112/ResNet-Tensorflow
# Based on https://github.com/taki0112/ResNeXt-Tensorflow
import sys
sys.path.append('./utility')
import numpy as np
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
                 is_stochastic_depth=False,
                 trainable=False):
        super().__init__(name=name, out_dim=out_dim, opt=opt, lr=lr, l2_reg=l2_reg, l2_reg_scale=l2_reg_scale, trainable=trainable)
        #resnet type -> '18, 34, 50, 101, 152'
        self.n_res = 18
        self.filter = 64
        self.p_L = 0.5 if is_stochastic_depth else 1.0

        if self.n_res < 50 :
            self.residual_block = self.resblock
        else :
            self.residual_block = self.bottle_resblock

        self.residual_list = self.get_residual_layer()

    
    def inference(self, x, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            x = self.conv(x, [3, self.filter, 1, None])
            for i in range(self.residual_list[0]):
                x = self.residual_block(x, channels=self.filter, layer_num=1, downsample=False, name='resblock0_' + str(i))
            x = self.residual_block(x, channels=self.filter*2, layer_num=2, downsample=True, name='resblock1_0')
            for i in range(1, self.residual_list[1]) :
                x = self.residual_block(x, channels=self.filter*2, layer_num=2, downsample=False, name='resblock1_' + str(i))
            x = self.residual_block(x, channels=self.filter*4, layer_num=3, downsample=True, name='resblock2_0')
            for i in range(1, self.residual_list[2]) :
                x = self.residual_block(x, channels=self.filter*4, layer_num=3, downsample=False, name='resblock2_' + str(i))
            x = self.residual_block(x, channels=self.filter*8, layer_num=4, downsample=True, name='resblock_3_0')
            for i in range(1, self.residual_list[3]) :
                x = self.residual_block(x, channels=self.filter*8, layer_num=4, downsample=False, name='resblock_3_' + str(i))
            x = self.ReLU(self.BN(x, [None]),[None])
            x = self.gap(x,[self.out_dim])
            x = self.fc(x, [self.out_dim, None])
            logits  = tf.identity(x, name="output_logits")
            return logits

    def resblock(self, x, channels, layer_num, downsample=False, name=None):
        with tf.variable_scope(name):
            if self.stochastic_depth(layer_num):
                return self.conv(x, [1, channels, 2, None]) if downsample else x
            else:
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

    def stochastic_depth(self, idx, L=4):
        """
        Base
        -----
            https://qiita.com/supersaiakujin/items/eb0553a1ef1d46bd03fa
        
        parameter
        -----
        idx : present layer
        L : value of Layers
        """
        if self.p_L == 1. or self._trainable is False:
            return False
        
        survival_probability = 1.0 - idx / L * (1.0 - self.p_L)
        if np.random.rand() >= survival_probability: # layer方向にDropout
            return True
        else:
            return False

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
            featmap = self.fc(x, [self.out_dim, None])
            logits  = tf.identity(featmap, name="output_logits")
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


class SENet(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def resblock(self, x, channels, layer_num, downsample=False, name=None):
        with tf.variable_scope(name):
            if self.stochastic_depth(layer_num):
                return self.conv(x, [1, channels, 2, None]) if downsample else x
            else:
                logits = self.ReLU(self.BN(x, [None]),[None])
                if downsample:
                    logits = self.conv(logits, [3, channels, 2, None])
                    x = self.conv(x, [1, channels, 2, None])
                else:
                    logits = self.conv(logits, [3, channels, 1, None])
                logits = self.ReLU(self.BN(logits, [None]),[None])
                logits = self.conv(logits, [3, channels, 1, None])

                # Squeeze-and-Excitation Block
                logits = self.squeeze_excitation_layer(logits, 4, name='selayer')
                
                return logits + x

    def squeeze_excitation_layer(self, x, ratio, name):
        with tf.variable_scope(name):
            channel = x.get_shape().as_list()[-1]
            logits = self.gap(x, [channel])
            logits = self.fc(logits, [channel/ratio,tf.nn.relu])
            logits = self.fc(logits, [channel, tf.nn.sigmoid])
            logits = tf.reshape(logits, [-1, 1, 1, channel])

            return x * logits

class sSENet(ResNet):
    """
    Channel Squeeze and Spatial Excitation Block
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def resblock(self, x, channels, layer_num, downsample=False, name=None):
        with tf.variable_scope(name):
            if self.stochastic_depth(layer_num):
                return self.conv(x, [1, channels, 2, None]) if downsample else x
            else:
                logits = self.ReLU(self.BN(x, [None]),[None])
                if downsample:
                    logits = self.conv(logits, [3, channels, 2, None])
                    x = self.conv(x, [1, channels, 2, None])
                else:
                    logits = self.conv(logits, [3, channels, 1, None])
                logits = self.ReLU(self.BN(logits, [None]),[None])
                logits = self.conv(logits, [3, channels, 1, None])

                # Squeeze-and-Excitation Block
                logits = self.channel_squeeze_and_spatial_excitation(logits, 4, name='sselayer')
                
                return logits + x

    def channel_squeeze_and_spatial_excitation(self, x, ratio, name):
        with tf.variable_scope(name):
            logits = self.conv(x, [1, 1, 1, tf.nn.sigmoid])
            return x * logits


class scSENet(ResNet):
    """
    Spatial and Channel Squeeze & Excitation 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def resblock(self, x, channels, layer_num, downsample=False, name=None):
        with tf.variable_scope(name):
            if self.stochastic_depth(layer_num):
                return self.conv(x, [1, channels, 2, None]) if downsample else x
            else:
                logits = self.ReLU(self.BN(x, [None]),[None])
                if downsample:
                    logits = self.conv(logits, [3, channels, 2, None])
                    x = self.conv(x, [1, channels, 2, None])
                else:
                    logits = self.conv(logits, [3, channels, 1, None])
                logits = self.ReLU(self.BN(logits, [None]),[None])
                logits = self.conv(logits, [3, channels, 1, None])

                # Squeeze-and-Excitation Block
                logits = self.concurrent_spatial_and_channel_se(logits, 4, name='scselayer')
                
                return logits + x

    def concurrent_spatial_and_channel_se(self, x, ratio, name):
        with tf.variable_scope(name):
            cse = self.squeeze_excitation_layer(x, ratio, name='selayer')
            sse = self.channel_squeeze_and_spatial_excitation(x, 4, name='cselayer')
            return cse + sse

    def squeeze_excitation_layer(self, x, ratio, name):
        with tf.variable_scope(name):
            channel = x.get_shape().as_list()[-1]
            logits = self.gap(x, [channel])
            logits = self.fc(logits, [channel/ratio,tf.nn.relu])
            logits = self.fc(logits, [channel, tf.nn.sigmoid])
            logits = tf.reshape(logits, [-1, 1, 1, channel])

            return x * logits


    def channel_squeeze_and_spatial_excitation(self, x, ratio, name):
        with tf.variable_scope(name):
            logits = self.conv(x, [1, 1, 1, tf.nn.sigmoid])
            return x * logits