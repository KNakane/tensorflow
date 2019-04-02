import os, sys
import numpy as np
from tqdm import tqdm
import tensorflow as tf

class MDN_Load():
    def __init__(self, name):
        self.name = name
        if self.name == 'sample':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = self.get_sample(10000)
            self.output_dim = 3
        else:
            NotImplementedError

    
    def load(self, inputs, answer, batch_size, buffer_size=1000, is_training=False):
        with tf.variable_scope('{}_dataset'.format('training' if is_training is True else 'validation')):
            def preprocess_fn(inputs, answer):
                '''A transformation function to preprocess raw data
                into trainable input. '''
                x = tf.cast(inputs, tf.float32)
                y = tf.cast(answer, tf.float32)
                return x, y

            if is_training: # training dataset
                self.x_train, self.y_train = inputs, answer
                self.features_placeholder = tf.placeholder(self.x_train.dtype, self.x_train.shape, name='input_images')
                self.labels_placeholder = tf.placeholder(self.y_train.dtype, self.y_train.shape, name='labels')
                dataset = tf.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))
            else:           # validation dataset
                self.x_test, self.y_test = inputs, answer
                self.valid_placeholder = tf.placeholder(self.x_test.dtype, self.x_test.shape, name='valid_inputs')
                self.valid_labels_placeholder = tf.placeholder(self.y_test.dtype, self.y_test.shape, name='valid_labels')
                dataset = tf.data.Dataset.from_tensor_slices((self.valid_placeholder, self.valid_labels_placeholder))

            # Transform and batch data at the same time
            dataset = dataset.apply(tf.data.experimental.map_and_batch(
                preprocess_fn, batch_size,
                num_parallel_batches=4,  # cpu cores
                drop_remainder=True if is_training else False))

            dataset = dataset.shuffle(buffer_size).repeat()  # depends on sample size
            dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

            return dataset

    
    def get_sample(self, NSAMPLE):
        y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
        y_data = np.random.permutation(y_data)
        r_data = np.float32(np.random.normal(size=(NSAMPLE,1))) # random noise
        x_data = np.float32(np.sin(0.75*y_data)*7.0+y_data*0.5+r_data*1.0)
        return (x_data[:8000], y_data[:8000]), (x_data[8000:], y_data[8000:])
