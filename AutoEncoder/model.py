import os, sys
import tensorflow as tf
from tensorflow.keras.models import Model
from utility.optimizer import *

class Encoder(Model):
    def __init__(self, 
                 model=None,
                 name='Encoder',
                 out_dim=10,
                 l2_reg=False,
                 l2_reg_scale=0.0001
                 ):
        super().__init__()
        self.model_name = name
        self.out_dim = out_dim
        self.l2_regularizer = l2_reg_scale if l2_reg else None
        self._build()

    def _build(self):
        self.flat = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.fc2 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.fc3 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.out = tf.keras.layers.Dense(self.out_dim, activation='relu', kernel_regularizer=self.l2_regularizer)
        return

    def __call__(self, x):
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.out(x)
        return x


class Decoder(Model):
    def __init__(self, 
                 model=None,
                 name='Encoder',
                 out_dim=10,
                 l2_reg=False,
                 l2_reg_scale=0.0001
                 ):
        super().__init__()
        self.model_name = name
        self.out_dim = out_dim
        self.l2_regularizer = l2_reg_scale if l2_reg else None
        self._build()

    def _build(self):
        self.fc1 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.fc2 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.fc3 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.out = tf.keras.layers.Reshape((28,28,1))
        return

    def __call__(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.out(x)
        return x


class AutoEncoder(Model):
    def __init__(self, 
                 denoise=False,
                 name='AutoEncoder',
                 size=28,
                 channel=1,
                 out_dim=10,
                 opt="Adam",   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001
                 ):
        super().__init__()
        self.model_name = name
        self.encode = Encoder(l2_reg, l2_reg_scale)
        self.decode = Decoder(l2_reg, l2_reg_scale)
        self.denoise = denoise
        self.out_dim = out_dim
        self.optimizer = eval(opt)(learning_rate=lr, decay_step=None, decay_rate=0.95)
        self.l2_regularizer = l2_reg_scale if l2_reg else None

    def noise(self, outputs):
        outputs += tf.random_normal(tf.shape(outputs))
        return tf.clip_by_value(outputs, 1e-8, 1 - 1e-8)
    
    def inference(self, outputs):
        self.inputs = outputs
        if self.denoise:
            outputs = self.noise(outputs)
        outputs = self.encode(outputs)
        outputs = self.decode(outputs)
        return outputs

    def loss(self, logits, anser):
        loss = tf.reduce_mean(tf.square(logits - anser))
        return loss


class VAE(AutoEncoder):
    def __init__(self, 
                 encode=None,
                 decode=None,
                 denoise=False,
                 name='VAE',
                 out_dim=10,
                 opt="Adam",   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001
                 ):
        super.__init__(encode=encode, decode=decode, denoise=denoise, name=name, out_dim=out_dim, opt=opt, lr=lr, l2_reg=l2_reg, l2_reg_scale=l2_reg_scale)

