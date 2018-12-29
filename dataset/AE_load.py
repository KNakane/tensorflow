import os,sys
import numpy as np
import tensorflow as tf
from load import Load

class AE_Load(Load):
    def __init__(self, name):
        super().__init__(name=name)
        self.y_train = self.x_train.copy()
        self.y_test = self.x_test.copy()

    def load(self, images, labels, batch_size, buffer_size=1000, is_training=False):
        def preprocess_fn(image, label):
            '''A transformation function to preprocess raw data
            into trainable input. '''
            x = tf.reshape(tf.cast(image, tf.float32), (self.size, self.size, self.channel)) / 255.0
            y = tf.reshape(tf.cast(image, tf.float32), (self.size, self.size, self.channel)) / 255.0
            return x, y

        self.features_placeholder = tf.placeholder(images.dtype, images.shape, name='input_images')
        self.labels_placeholder = tf.placeholder(images.dtype, images.shape, name='correct_images')
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

    def load_test(self, images, labels):
        x = tf.reshape(tf.cast(images, tf.float32), (-1, self.size, self.size, self.channel)) / 255.0
        y = tf.reshape(tf.cast(images, tf.float32), (-1, self.size, self.size, self.channel)) / 255.0
        return x, y