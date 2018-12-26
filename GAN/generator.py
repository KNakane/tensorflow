import os,sys
sys.path.append('./network')
import tensorflow as tf
from model import DNN

class Generator(DNN):
    def __init__(self):
        super().__init__()
        self.z = tf.random.normal(shape=[])

    def inference(self):
        return