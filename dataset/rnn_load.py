import os, sys
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from keras.datasets import *

class RNN_Load():
    def __init__(self, name):
        if name == "imdb":
            (self.x_train, self.y_train), (self.x_test, self.y_test) = imdb.load_data(num_words=10000)
        elif name == 'sample':
            self.x_train, self.y_train = self.create_data(1000, 10)
            self.x_test, self.y_test = self.create_data(1000, 10)
            self.output_dim = 1
        else:
            NotImplementedError

    def load(self, inputs, answer, batch_size, buffer_size=1000, is_training=False, augmentation=None):
        with tf.variable_scope('{}_dataset'.format('training' if is_training is True else 'validation')):
            def preprocess_fn(inputs, answer):
                '''A transformation function to preprocess raw data
                into trainable input. '''
                x = tf.reshape(tf.cast(inputs, tf.float32), (10, 1))
                y = tf.cast(answer, tf.float32)
                return x, y

            answer = answer.reshape(answer.shape[0])

            """
            if augmentation is not None and is_training:
                augment = Augment(images, labels)
                images, labels = eval('augment.'+ augmentation)()
            """

            if is_training: # training dataset
                self.x_train, self.y_train = inputs, answer
                self.features_placeholder = tf.placeholder(self.x_train.dtype, self.x_train.shape, name='input_data')
                self.labels_placeholder = tf.placeholder(self.y_train.dtype, self.y_train.shape, name='answer')
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


    def create_data(self, nb_of_samples, sequence_len):
        X = np.zeros((nb_of_samples, sequence_len))
        for row_idx in range(nb_of_samples):
            X[row_idx, :] = np.around(np.random.rand(sequence_len)).astype(int)
        # Create the targets for each sequence
        t = np.sum(X, axis=1)
        return X, t