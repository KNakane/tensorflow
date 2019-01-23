# -*- coding: utf-8 -*-
import sys
import numpy as np
import tensorflow as tf
sys.path.append('./utility')
from eager_module import EagerModule
from optimizer import *

class EagerCNN(EagerModule):
    def __init__(self, 
                 model=None,
                 name='CNN',
                 out_dim=10,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 trainable=False
                 ):
        super().__init__(l2_reg=l2_reg,l2_reg_scale=l2_reg_scale, trainable=trainable)
        self.model = model
        self._layers = []
        if self._trainable:
            self.optimizer = eval(opt)(learning_rate=lr)
        self._build()

    def _build(self):
        for l in range(len(self.model)):
            my_layer = eval('self.' + self.model[l][0])(self.model[l][1:])
            self._layers.append(my_layer)

    def inference(self, x):
        for my_layer in self._layers:
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            try:
                x = my_layer(x, training=self._trainable)
            except:
                x = my_layer(x)
        return x

    def loss(self, logits, labels):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        return loss

    def optimize(self, loss, global_step, tape=None):
        assert tape is not None, 'please set tape in opmize'
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.method.apply_gradients(zip(grads, self.trainable_variables),global_step)


class Dueling_Net(EagerCNN):
    def __init__(self, 
                 model=None,
                 name='CNN',
                 out_dim=10,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 trainable=False
                 ):
        self.out_dim = out_dim
        super().__init__(model=model,opt=opt,l2_reg=l2_reg,l2_reg_scale=l2_reg_scale,trainable=trainable)

    def _build(self):
        for l in range(len(self.model)):
            if l == len(self.model) - 1:
                self.model[l][1] = self.out_dim + 1   # 状態価値V用に1unit追加
            my_layer = eval('self.' + self.model[l][0])(self.model[l][1:])
            self._layers.append(my_layer)

    def inference(self, x):
        for i, my_layer in enumerate(self._layers):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            try:
                x = my_layer(x, training=self._trainable)
            except:
                x = my_layer(x)
                
        # Dueling part
        V = tf.reshape(x[:,0], (x.shape[0], 1))
        V = tf.tile(V, [1, self.out_dim])
        x = x[:, 1:] + V - tf.tile(tf.reshape(np.average(x[:,1:], axis=1), (x.shape[0], 1)), [1, self.out_dim])
        return x