import os,sys
sys.path.append('./CNN')
import tensorflow as tf
from keras.datasets import *
from keras.utils import np_utils
from data_load import Load

class GAN_Load(Load):
    def __init__(self,name):
        super().__init__(name=name)

    def g_load(self, batch_size):
        def preprocess_fn(image, label):
            '''A transformation function to preprocess raw data
            into trainable input. '''
            x = tf.reshape(tf.cast(image, tf.float32), (self.size, self.size, self.channel))
            y = tf.one_hot(tf.cast(label, tf.uint8), self.output_dim)
            return x, y
        
        self.z = tf.random.normal(shape=(batch_size, 100))
        self.features_placeholder = tf.placeholder(images.dtype, images.shape, name='input_noize')
        self.labels_placeholder = tf.placeholder(labels.dtype, labels.shape, name='labels')

        return