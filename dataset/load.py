import os,sys
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.datasets import *
from dataset.augmentation import Augmentation
from sklearn.model_selection import train_test_split

class Load():
    def __init__(self, name):
        if name == "kuzushiji":
            (x_train, y_train), (self.x_test, self.y_test) = self.get_kuzushiji()
            self.size, self.channel = 28, 1
            self.output_dim = 49
        else:
            (x_train, y_train), (self.x_test, self.y_test) = self.get(name)

        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(x_train, y_train, test_size=0.175)

    @property
    def input_shape(self):
        return (self.size, self.size, self.channel)

    def get(self, name):
        train_dataset, info = tfds.load(name=name, with_info=True, split=tfds.Split.TRAIN, batch_size=-1)
        test_dataset = tfds.load(name=name, split=tfds.Split.TEST, batch_size=-1)

        train = tfds.as_numpy(train_dataset)
        test = tfds.as_numpy(test_dataset)

        x_train, y_train = train["image"], train["label"]
        x_test, y_test = test["image"], test["label"]

        # Information
        shape = info.features['image'].shape
        self.size, self.channel = shape[0], shape[2]
        self.output_dim = info.features['label'].num_classes
        return (x_train, y_train), (x_test, y_test)

    def get_kuzushiji(self):
        train_image = np.load('./dataset/k49-train-imgs.npz')
        train_label = np.load('./dataset/k49-train-labels.npz')
        test_image = np.load('./dataset/k49-test-imgs.npz')
        test_label = np.load('./dataset/k49-test-labels.npz')
        x_train = train_image['arr_0']
        y_train = train_label['arr_0']
        x_test = test_image['arr_0']
        y_test = test_label['arr_0']
        return (x_train, y_train), (x_test, y_test)

    def load(self, batch_size=32, buffer_size=1000, is_training=False, augmentation=None):
        def preprocess_fn(image, label):
            '''A transformation function to preprocess raw data
            into trainable input. '''
            x = tf.reshape(tf.cast(image, tf.float32), (self.size, self.size, self.channel)) / 255.0
            y = tf.one_hot(tf.cast(label, tf.uint8), self.output_dim)
            return x, y

        if is_training:
            dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((self.x_valid, self.y_valid))

        # Transform and batch data at the same time
        dataset = dataset.map(map_func=preprocess_fn, num_parallel_calls=4)
        if augmentation is not None:
            aug = Augmentation(augmentation)
            dataset = dataset.map(aug)
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size, drop_remainder=True if is_training else False)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def next_batch(self, batch_size):
        '''
        Return a total of `batch_size` random samples and labels. 
        '''
        idx = np.arange(0 , len(self.x_train))
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        data_shuffle = [self.x_train[ i] for i in idx]
        labels_shuffle = [self.y_train[ i] for i in idx]

        return idx, np.asarray(data_shuffle), np.asarray(labels_shuffle)


class SeqLoad(Load):
    def __init__(self, name, pix_by_pix=False):
        self._pix_by_pix = pix_by_pix
        if name == "kuzushiji":
            (x_train, y_train), (self.x_test, self.y_test) = self.get_kuzushiji()
            self.timestep, self.data_num, channel = 28, 28, 1
            self.output_dim = 49
        else:
            dataset_name = 'tf.keras.datasets.'+ name
            (x_train, y_train), (self.x_test, self.y_test) = self.get(dataset_name)
            if name == 'mnist':
                self.timestep, self.data_num, channel = 28, 28, 1
                self.output_dim = 10
            else:
                NotImplementedError

        if pix_by_pix:
            x_train = x_train.reshape((-1, self.timestep * self.data_num, channel))
            self.x_test = self.x_test.reshape((-1, self.timestep * self.data_num, channel))
            self.timestep = self.timestep * self.data_num
            self.data_num = channel

        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(x_train, y_train, test_size=0.175)

    @property
    def input_shape(self):
        return (self.timestep, self.data_num)

    def load(self, batch_size=32, buffer_size=1000, is_training=False):
        def preprocess_fn(inputs, label):
            x = tf.reshape(tf.cast(inputs, tf.float32), (self.timestep, self.data_num)) / 255.0
            y = tf.one_hot(tf.cast(label, tf.uint8), self.output_dim)
            return x, y

        if is_training:
            dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((self.x_valid, self.y_valid))

        dataset = dataset.map(map_func=preprocess_fn, num_parallel_calls=4)
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size, drop_remainder=True if is_training else False)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset