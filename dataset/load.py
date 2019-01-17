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
        with tf.variable_scope('{}_dataset'.format('training' if is_training is True else 'validation')):
            def preprocess_fn(image, label):
                '''A transformation function to preprocess raw data
                into trainable input. '''
                x = tf.reshape(tf.cast(image, tf.float32), (self.size, self.size, self.channel)) / 255.0
                y = tf.one_hot(tf.cast(label, tf.uint8), self.output_dim)
                return x, y

            labels = labels.reshape(labels.shape[0])

            if augmentation != 'None' and is_training:
                augment = Augment(images, labels)
                images, labels = eval('augment.'+augmentation)()

            if is_training: # training dataset
                self.x_train, self.y_train = images, labels
                self.features_placeholder = tf.placeholder(self.x_train.dtype, self.x_train.shape, name='input_images')
                self.labels_placeholder = tf.placeholder(self.y_train.dtype, self.y_train.shape, name='labels')
                dataset = tf.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))
            else:           # validation dataset
                self.x_test, self.y_test = images, labels
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

    def load_test(self, images, labels):
        x = tf.reshape(tf.cast(images, tf.float32), (-1, self.size, self.size, self.channel)) / 255.0
        y = tf.one_hot(tf.cast(labels, tf.uint8), self.output_dim)
        return x, y


if __name__ == '__main__':
    data = Load('mnist')
    #dataset = data.load(data.x_train, data.y_train, batch_size=32, is_training=True, is_augment=True)
    augment = Augment(data.x_train, data.y_train)
    #images, labels = augment.shift(v=3, h=3) #上下左右に3ピクセルずつランダムにずらす
    images, labels = augment.random_erace() #上下左右に3ピクセルずつランダムにずらす
    import matplotlib.pyplot as plt
    plt.imshow(images[0])
    plt.show()
    plt.imshow(images[60000])
    plt.show()