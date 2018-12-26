import os,sys
sys.path.append('./CNN')
import numpy as np
import tensorflow as tf
from keras.datasets import *
from keras.utils import np_utils
from data_load import Load

class GAN_Load(Load):
    def __init__(self, name, batch_size):
        super().__init__(name=name)
        self.z = tf.random.normal(shape=[batch_size,100])
        self.correct_label = np.ones(batch_size)
        self.fake_label = np.zeros(batch_size)

    def load(self, images, labels, batch_size, buffer_size=1000, is_noise=False, is_training=False):
        def preprocess_fn(image, label):
            '''A transformation function to preprocess raw data
            into trainable input. '''
            if is_noise:
                x = tf.cast(image, tf.float32)
            else:
                x = tf.reshape(tf.cast(image, tf.float32), (self.size, self.size, self.channel))
            y = tf.cast(label, tf.uint8)
            return x, y

        self.feature_placeholder = tf.placeholder(images.dtype, images.shape, name='input')
        self.labels_placeholder = tf.placeholder(labels.dtype, labels.shape, name='label')
        dataset = tf.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))

        # Transform and batch data at the same time
        dataset = dataset.apply(tf.data.experimental.map_and_batch( #tf.contrib.data.map_and_batch(
            preprocess_fn, batch_size,
            num_parallel_batches=4,  # cpu cores
            drop_remainder=True if is_training else False))

        if is_training:
            dataset = dataset.shuffle(buffer_size).repeat()  # depends on sample size
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset