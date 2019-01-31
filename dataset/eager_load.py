import os,sys
import numpy as np
import tensorflow as tf
from keras.datasets import *
from Augmentation import Augment

class Load():
    def __init__(self, name):
        if name == "kuzushiji":
            self.get_kuzushiji()
        else:
            self.name = 'tf.keras.datasets.'+ name
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

    def get_kuzushiji(self):
        train_image = np.load('./dataset/k49-train-imgs.npz')
        train_label = np.load('./dataset/k49-train-labels.npz')
        test_image = np.load('./dataset/k49-test-imgs.npz')
        test_label = np.load('./dataset/k49-test-labels.npz')
        self.x_train = train_image['arr_0']
        self.y_train = train_label['arr_0']
        self.x_test = test_image['arr_0']
        self.y_test = test_label['arr_0']
        self.size, self.channel = 28, 1
        self.output_dim = 49

    def load(self, images, labels, batch_size, buffer_size=1000, is_training=False, augmentation=None):
        def preprocess_fn(image, label):
            '''A transformation function to preprocess raw data
            into trainable input. '''
            x = tf.reshape(tf.cast(image, tf.float32), (self.size, self.size, self.channel))
            y = tf.one_hot(tf.cast(label, tf.uint8), self.output_dim)
            return x, y

        labels = labels.reshape(labels.shape[0])
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        # Transform and batch data at the same time
        dataset = dataset.apply(tf.data.experimental.map_and_batch(
            preprocess_fn, batch_size,
            num_parallel_batches=4,  # cpu cores
            drop_remainder=True if is_training else False))

        if is_training:
            dataset = dataset.shuffle(buffer_size).repeat()  # depends on sample size
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset