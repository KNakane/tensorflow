import os,sys
sys.path.append('./network')
sys.path.append('./utility')
import tensorflow as tf
from cnn import CNN
from optimizer import *

class Generator(CNN):
    def __init__(self, 
                 model,
                 opt=Adam,
                 name='Generator',
                 trainable=False):
        super().__init__(model=model, name=name, opt=opt, trainable=trainable)

    def inference(self, outputs):
        with tf.variable_scope(self.name):
            for l in range(len(self.model)):
                outputs = (eval('self.' + self.model[l][0])(outputs, self.model[l][1:]))
            return outputs