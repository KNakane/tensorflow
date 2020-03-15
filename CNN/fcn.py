import os, sys
import numpy as np
import tensorflow as tf
from CNN.model import MyModel
from utility.optimizer import *

class FCN(MyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self):
        self.conv1 = tf.keras.layers.Conv1D(6, kernel_size=(5, 5), padding='valid', activation='relu', kernel_regularizer=self.l2_regularizer)
        self.out = tf.keras.layers.Dense(self.out_dim, activation='softmax')
        return