import os,sys
sys.path.append('./network')
sys.path.append('./utility')
import tensorflow as tf
from cnn import CNN
from optimizer import *

class Discriminator(CNN):
    def __init__(self, 
                 model,
                 opt=Adam,
                 name='Discriminator',
                 trainable=False):
        super().__init__(model=model, name=name, opt=opt, trainable=trainable)
    
    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    def inference(self, logits, reuse=False):
        with tf.variable_scope("discriminator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            for l in range(len(self.model)):
                logits = (eval('self.' + self.model[l][0])(logits, self.model[l][1:]))
            return tf.nn.sigmoid(logits) ,logits

    def optimize(self, loss, global_step=None):
        return self.optimizer.optimize(loss=loss, global_step=global_step)#, var_list=self.var)