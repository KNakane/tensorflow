import os, sys
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from keras.datasets import imdb

class RNN_Load():
    def __init__(self, name):
        if name == "imdb":
            (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
        else:
            NotImplementedError

    def load(self):
        return