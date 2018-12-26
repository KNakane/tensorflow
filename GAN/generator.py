import os,sys
sys.path.append('./network')
import tensorflow as tf
from model import DNN

class Generator(DNN):
    def __init__(self, 
                 model,
                 name='Generator',
                 trainable=False):
        super().__init__(model=model, name=name, trainable=trainable)

    def inference(self, outputs):
        with tf.variable_scope(self.name):
            for l in range(len(self.model)):
                outputs = (eval('self.' + self.model[l][0])(outputs, self.model[l][1:]))
            return outputs