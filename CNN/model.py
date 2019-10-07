import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

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
        self.opt = opt
        self._build()

    def _build(self):
        NotImplementedError

    def inference(self, x):
        return

    def loss(self, logits, answer, regression=False):
        return tf.keras.losses.MeanSquaredError() if regression else tf.losses.CategoricalCrossentropy()

    def optimize(self, loss, tape=None):
        assert tape is not None, 'please set tape in opmize'
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.method.apply_gradients(zip(grads, self.trainable_variables))
        return

    def accuracy(self, logits, answer):
        accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        return accuracy(logits, answer)