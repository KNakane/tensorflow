import os, sys
import numpy as np
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
                 size=28,
                 channel=1,
                 l2_reg=False,
                 l2_reg_scale=0.0001
                 ):
        super().__init__()
        self.model_name = name
        self.size = size
        self.channel = channel
        self.l2_regularizer = l2_reg_scale if l2_reg else None
        self._build()

    def _build(self):
        self.fc1 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.fc2 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.fc3 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.fc4 = tf.keras.layers.Dense(self.size**2 * self.channel, activation='sigmoid', kernel_regularizer=self.l2_regularizer)
        self.out = tf.keras.layers.Reshape((self.size,self.size,self.channel))
        return

    def __call__(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
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
        self.encode = Encoder(out_dim=out_dim, l2_reg=l2_reg, l2_reg_scale=l2_reg_scale)
        self.decode = Decoder(size=size, channel=channel, l2_reg=l2_reg, l2_reg_scale=l2_reg_scale)
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
    
    def optimize(self, loss, tape=None):
        assert tape is not None, 'please set tape in opmize'
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.method.apply_gradients(zip(grads, self.trainable_variables))
        return

    def accuracy(self, logits, answer, eps=1e-10):
        marginal_likelihood = tf.reduce_mean(answer * tf.math.log(logits + eps) + (1 - answer) * tf.math.log(1 - logits + eps),
                                            [1, 2])
        neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)
        return neg_loglikelihood


class VAE(AutoEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def inference(self, outputs):
        self.inputs = outputs
        if self.denoise:
            outputs = self.noise(outputs)
        outputs = self.encode(outputs)
        self.mu, self.var = tf.split(outputs, num_or_size_splits=2, axis=1)
        compose_img = self.re_parameterization(self.mu, self.var)
        outputs = tf.clip_by_value(self.decode(compose_img), 1e-8, 1 - 1e-8)
        return outputs

    def test_inference(self, outputs):
        batch_size = tf.constant(outputs.shape[0], dtype=tf.int32)
        compose_img = self.gaussian(batch_size,20)
        outputs = tf.clip_by_value(self.decode_(compose_img), 1e-8, 1 - 1e-8)
        return outputs

    def loss(self, logits, answer):
        epsilon = 1e-10
        reconstruct_loss = tf.reduce_mean(-tf.reduce_sum(answer * tf.math.log(epsilon + logits) + (1 - answer) * tf.math.log(epsilon + 1 - logits), axis=1))
        KL_divergence = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + self.var - tf.square(self.mu) - tf.exp(self.var), axis=1))      
        return reconstruct_loss + KL_divergence

    def re_parameterization(self, mu, var):
        """
        Reparametarization trick
        parameters
        ---
        mu, var : numpy array or tensor
            mu is average, var is variance
        """
        std = var**0.5
        eps = tf.random.normal(tf.shape(var), 0, 1, dtype=tf.float32)
        return mu + tf.sqrt(tf.exp(std)) * eps

    def gaussian(self, batch_size, n_dim, mean=0, var=1):
        z = tf.random.normal(shape=(batch_size, n_dim), mean=mean, stddev=var)
        return z