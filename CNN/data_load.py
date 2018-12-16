import os,sys
from keras.datasets import *

class Load():
    def __init__(self,name):
        self.name = 'tf.keras.datasets'+name
        self.datasets = eval(self.name)

    def get(self):
        try:
            return self.datasets.load_data(label_mode='fine')
        except:
            return self.datasets.load_data()