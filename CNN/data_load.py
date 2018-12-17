import os,sys
import tensorflow as tf
from keras.datasets import *

class Load():
    def __init__(self,name):
        self.name = 'tf.keras.datasets.'+name
        self.datasets = eval(self.name)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.get()
        if name == 'mnist':
            self.size, self.channel = 28, 1
            self.output_dim = 10
        elif name == 'cifar10':
            self.size, self.channel = 32, 3
            self.output_dim = 10
        elif name == 'cifar100':
            self.size, self.channel = 32, 3
            self.output_dim = 100
        else:
            NotImplementedError

    def get(self):
        try:
            return self.datasets.load_data(label_mode='fine')
        except:
            return self.datasets.load_data()

    def load(self, images, labels, batch_size, buffer_size=1000, is_training=False):
        def preprocess_fn(image, label):
            '''A transformation function to preprocess raw data
            into trainable input. '''
            x = tf.reshape(tf.cast(image, tf.float32), (self.size, self.size, self.channel))
            y = tf.one_hot(tf.cast(label, tf.uint8), self.output_dim)
            return x, y

        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        # Transform and batch data at the same time
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            preprocess_fn, batch_size,
            num_parallel_batches=4,  # cpu cores
            drop_remainder=True if is_training else False))

        if is_training:
            dataset = dataset.shuffle(buffer_size).repeat()  # depends on sample size
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset