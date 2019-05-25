# -*- coding: utf-8 -*-
import sys
import numpy as np
import tensorflow as tf
sys.path.append('./utility')
from optimizer import *
from eager_nn import BasedEagerNN

class LeNet(BasedEagerNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self):
        self._layers = [tf.keras.layers.Conv2D(6, kernel_size=(5, 5), padding='valid', activation='relu'),
                        tf.keras.layers.MaxPooling2D(padding='same'),
                        tf.keras.layers.Conv2D(16, kernel_size=(5, 5), padding='valid', activation='relu'),
                        tf.keras.layers.MaxPooling2D(padding='same'),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(120, activation='relu'),
                        tf.keras.layers.Dense(84, activation='relu'),
                        tf.keras.layers.Dense(self.out_dim)
                        ]

    @tf.contrib.eager.defun
    def inference(self, x, softmax=True):
        for my_layer in self._layers:
            try:
                x = my_layer(x, training=self._trainable)
            except:
                x = my_layer(x)
        return x
