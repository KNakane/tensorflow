import os,sys
import requests
import numpy as np
from tqdm import tqdm
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
        if not os.path.isfile('./dataset/k49-train-imgs.npz'):
            self.down_load_kuzushiji()
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
                drop_remainder=True))

            dataset = dataset.shuffle(buffer_size).repeat()  # depends on sample size
            dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

            return dataset

    def load_test(self, images, labels, batch_size):
        def preprocess_fn(image, label):
            x = tf.reshape(tf.cast(image, tf.float32), (self.size, self.size, self.channel)) / 255.0
            y = tf.one_hot(tf.cast(label, tf.uint8), self.output_dim)
            return x, y

        self.x_test, self.y_test = images, labels
        self.test_placeholder = tf.placeholder(self.x_test.dtype, self.x_test.shape, name='valid_inputs')
        self.test_labels_placeholder = tf.placeholder(self.y_test.dtype, self.y_test.shape, name='valid_labels')
        dataset = tf.data.Dataset.from_tensor_slices((self.test_placeholder, self.test_labels_placeholder))

        dataset = dataset.apply(tf.data.experimental.map_and_batch(
                preprocess_fn, batch_size,
                num_parallel_batches=4,  # cpu cores
                drop_remainder=False))
        
        return dataset

    def down_load_kuzushiji(self):
        url_list = ['http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz',
                    'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz',
                    'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz',
                    'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz']
        for url in url_list:
            path = url.split('/')[-1]
            r = requests.get(url, stream=True)
            with open("./dataset/"+path, 'wb') as f:
                total_length = int(r.headers.get('content-length'))
                print('Downloading {} - {:.1f} MB'.format(path, (total_length / 1024000)))
                for chunk in tqdm(r.iter_content(chunk_size=1024), total=int(total_length / 1024) + 1, unit="KB"):
                    if chunk:
                        f.write(chunk)



if __name__ == '__main__':
    data = Load('cifar100')
    #dataset = data.load(data.x_train, data.y_train, batch_size=32, is_training=True, is_augment=True)
    augment = Augment(data.x_train, data.y_train)
    #images, labels = augment.shift(v=3, h=3) #上下左右に3ピクセルずつランダムにずらす
    images, labels = augment.shift() #上下左右に3ピクセルずつランダムにずらす
    import matplotlib.pyplot as plt
    plt.imshow(images[0])
    plt.show()
    plt.imshow(images[60000])
    plt.show()