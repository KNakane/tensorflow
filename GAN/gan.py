import os,sys
sys.path.append('./network')
import tensorflow as tf
from model import DNN

class GAN(DNN):
    def __init__(self):
        super().__init__()
        z = tf.random_normal((batch_size, self._z_dim), dtype=tf.float32)
