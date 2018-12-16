# -*- coding: utf-8 -*-
#tensorboard --logdir ./logs
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../practice/program'))
import tensorflow as tf
from cnn_model import CNN


def CNNFunction(output_dim=1, output_activation=tf.identity, initializer=None):
    def get_model(inputs, name="CNNMLP", trainable=True):
        model = CNN(model='variableCNN',
                    inputs=inputs,
                    select=[('conv', 8, 32, 4),
                            ('conv', 4, 32, 2),
                            ('conv', 3, 64, 1),
                            ('fc', 512)
                            ],  # cnn = (func_name, kernel_size, filter, strides), pool = (func_name, kernel_size, strides)
                    output_dim=output_dim,
                    output_activation=output_activation,
                    name = name,
                    trainable=trainable,
                    initializer=initializer)
        return model()
    return get_model


def NNFunction(output_dim=1, output_activation=tf.identity, initializer=None):
    def get_model(inputs, name="CNNMLP", trainable=True):
        model=  CNN(model='variableCNN',
                    inputs=inputs,
                    select=[('fc', 16),
                            ('fc', 16)
                            ],  # cnn = (func_name, kernel_size, filter, strides), pool = (func_name, kernel_size, strides)
                    output_dim=output_dim,
                    output_activation=output_activation,
                    name = name,
                    trainable=trainable,
                    initializer=initializer)
        return model()
    return get_model