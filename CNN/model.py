import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from optimizer.optimizer import *

class MyModel(Model):
    def __init__(self, 
                 model=None,
                 name='Model',
                 input_shape=None,
                 out_dim=10,
                 opt="Adam",   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001
                 ):
        super().__init__()
        self.model_name = name
        self.input_shape = input_shape
        self.out_dim = out_dim
        self.optimizer = eval(opt)(learning_rate=lr, decay_step=None, decay_rate=0.95)
        self.l2_regularizer = tf.keras.regularizers.l2(l2_reg_scale) if l2_reg else None
        self._build()
        self.loss_function = tf.losses.CategoricalCrossentropy()
        self.accuracy_function = tf.keras.metrics.CategoricalAccuracy()
        with tf.device("/cpu:0"):
            self(x=tf.constant(tf.zeros(shape=(1,)+self.input_shape,
                                             dtype=tf.float32)))

    def _build(self):
        raise NotImplementedError()

    def __call__(self, x, trainable=True):
        raise NotImplementedError()

    def test_inference(self, x, trainable=False):
        return self.__call__(x, trainable=trainable)

    def loss(self, logits, answer):
        return self.loss_function(y_true=answer, y_pred=logits)

    def optimize(self, loss, tape=None):
        assert tape is not None, 'please set tape in opmize'
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.method.apply_gradients(zip(grads, self.trainable_variables))
        return

    def accuracy(self, logits, answer):
        self.accuracy_function(y_true=answer, y_pred=logits)
        return self.accuracy_function.result()


class BasedResNet(MyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._first_layer()

    def _first_layer(self):
        self.conv1 = tf.keras.layers.Convolution2D(64, (7,7), (2,2), 'same', kernel_regularizer=self.l2_regularizer)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2, 2), padding='same')
        return
    
    def _building_block(self, channel_out=64, downsample=False):
        return ResidualBlock(channel_out=channel_out, downsample=downsample, l2_reg_scale=self.l2_regularizer)


class ResidualBlock(Model):
    def __init__(self, channel_out=64, downsample=False, l2_reg_scale=None):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            self.conv1 = tf.keras.layers.Convolution2D(channel_out, (3,3), (2,2), kernel_regularizer=l2_reg_scale)
            self.downsampling = tf.keras.layers.Convolution2D(channel_out, (1,1), (2,2))
        else:
            self.conv1 = tf.keras.layers.Convolution2D(channel_out, (3,3), padding='same', kernel_regularizer=l2_reg_scale)
            self.downsampling = tf.keras.layers.Convolution2D(channel_out, (1,1))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Convolution2D(channel_out, (3,3), padding='same', kernel_regularizer=l2_reg_scale)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.add = tf.keras.layers.Add()
        self.relu2 = tf.keras.layers.ReLU()

    def call(self, x, training=True):
        x = self.downsampling(x)
        h = self.conv1(x, training=training)
        h = self.bn1(h, training=training)
        h = self.relu1(h, training=training)
        h = self.conv2(h, training=training)
        h = self.bn2(h, training=training)
        h = self.add([x, h])
        h = self.relu2(h)
        return h