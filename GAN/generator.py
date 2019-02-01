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

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    def inference(self, outputs):
        with tf.variable_scope(self.name):
            for l in range(len(self.model)):
                outputs = (eval('self.' + self.model[l][0])(outputs, self.model[l][1:]))
            return outputs

    def optimize(self, loss, global_step=None):
        #print(self.name,self.var)
        #sys.exit()
        return self.optimizer.optimize(loss=loss, global_step=global_step)#, var_list=self.var)