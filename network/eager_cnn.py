# -*- coding: utf-8 -*-
import sys
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
        self.out_dim = out_dim
        if self._trainable:
            self.optimizer = eval(opt)(learning_rate=lr)
        self._build()

    def _build(self):
        for l in range(len(self.model)):
            my_layer = eval('self.' + self.model[l][0])(self.model[l][1:])
            self._layers.append(my_layer)

    def inference(self, x):
        for my_layer in self._layers:
            x = tf.convert_to_tensor(x)
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