# -*- coding: utf-8 -*-
import os,sys
sys.path.append('./utility')
sys.path.append('./network')
import tensorflow as tf
from cnn import CNN
from module import Module
from optimizer import *

class Critic_Net(CNN):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def inference(self, states, actions, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            for l in range(len(self.model)):
                if self.model[l][0] == 'fc':
                    states = tf.concat([states, actions],1)
                states = (eval('self.' + self.model[l][0])(states, self.model[l][1:]))

        return states
