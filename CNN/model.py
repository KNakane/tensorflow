import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from utility.optimizer import *

class MyModel(Model):
    def __init__(self, 
                 model=None,
                 name='Model',
                 out_dim=10,
                 opt="Adam",   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001
                 ):
        super().__init__()
        self.model_name = name
        self.out_dim = out_dim
        self.optimizer = eval(opt)(learning_rate=lr, decay_step=None, decay_rate=0.95)
        self.l2_regularizer = l2_reg_scale if l2_reg else None
        self._build()
        self.loss_function = tf.losses.CategoricalCrossentropy()
        self.accuracy_function = tf.keras.metrics.CategoricalAccuracy()

    def _build(self):
        raise NotImplementedError()

    def inference(self, x):
        raise NotImplementedError()

    def test_inference(self, x):
        return self.inference(x)

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