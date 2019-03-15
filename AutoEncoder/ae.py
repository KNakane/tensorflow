# -*- coding: utf-8 -*-
import os,sys
sys.path.append('../utility')
sys.path.append('../network')
import numpy as np
import tensorflow as tf
from cnn import CNN
from module import Module
from optimizer import *

class Encode(Module):
    def __init__(self, model, l2_reg=False, l2_reg_scale=0.0001, name='Encode', trainable=True):
        super().__init__(l2_reg=l2_reg,l2_reg_scale=l2_reg_scale, trainable=trainable)
        self.model = model
        self.name = name

    def __call__(self, outputs, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            for l in range(len(self.model)):
                outputs = (eval('self.' + self.model[l][0])(outputs, self.model[l][1:]))
            return outputs


class Decode(Module):
    def __init__(self, model, l2_reg=False, l2_reg_scale=0.0001, name='Decode', trainable=True):
        super().__init__(l2_reg=l2_reg,l2_reg_scale=l2_reg_scale, trainable=trainable)
        self.model = model
        self.name = name

    def __call__(self, outputs, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            for l in range(len(self.model)):
                outputs = (eval('self.' + self.model[l][0])(outputs, self.model[l][1:]))
            return outputs


class AutoEncoder(CNN):
    def __init__(self, 
                 encode=None,
                 decode=None,
                 denoise=False,
                 name='AutoEncoder',
                 out_dim=10,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 trainable=False
                 ):
        assert encode is not None, "Please set encode model"
        assert decode is not None, "Please set decode model"
        super().__init__(name=name, out_dim=out_dim, opt=opt, lr=lr, l2_reg=l2_reg, l2_reg_scale=l2_reg_scale, trainable=trainable)
        self.encode = Encode(encode, l2_reg, l2_reg_scale)
        self.decode = Decode(decode, l2_reg, l2_reg_scale)
        self.decode_ = Decode(decode, trainable=False)
        self.denoise = denoise

    def noise(self, outputs):
        outputs += tf.random_normal(tf.shape(outputs))
        return tf.clip_by_value(outputs, 1e-8, 1 - 1e-8)

    def inference(self, outputs, reuse=False):
        self.inputs = outputs
        with tf.variable_scope(self.name):
            if self.denoise:
                outputs = self.noise(outputs)
            outputs = self.encode(outputs, reuse)
            outputs = self.decode(outputs, reuse)
            return outputs

    def test_inference(self, outputs, reuse):
        return self.inference(outputs, reuse)
        
    def loss(self, logits, labels):
        loss = tf.reduce_mean(tf.square(logits - self.inputs))
        return loss


class VAE(AutoEncoder):
    # Based on https://github.com/shaohua0116/VAE-Tensorflow
    def __init__(self, 
                 encode=None,
                 decode=None,
                 denoise=False,
                 name='VAE',
                 out_dim=10,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 trainable=False
                 ):
        super().__init__(encode=encode, decode=decode, name=name, out_dim=out_dim, opt=opt, lr=lr, l2_reg=l2_reg, l2_reg_scale=l2_reg_scale, trainable=trainable)

    def inference(self, outputs, reuse=False):
        self.inputs = outputs
        with tf.variable_scope(self.name):
            if self.denoise:
                outputs = self.noise(outputs)
            outputs = self.encode(outputs, reuse)
            self.mu, self.var = tf.split(outputs, num_or_size_splits=2, axis=1)
            compose_img = self.re_parameterization(self.mu, self.var)
            outputs = tf.clip_by_value(self.decode(compose_img, reuse), 1e-8, 1 - 1e-8)
            
            return outputs

    def test_inference(self, outputs, reuse=True):
        n = 32
        x = tf.convert_to_tensor(np.linspace(0.05, 0.95, n))
        z = tf.tile(tf.expand_dims(x, 0), [2,1])
        return self.decode_(z, reuse)

    
    def re_parameterization(self, mu, var):
        with tf.variable_scope('re_parameterization'):
            std = tf.exp(0.5 * var)
            eps = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
            return mu + tf.sqrt(tf.exp(std)) * eps

    def loss(self, logits, labels):
        epsilon = 1e-10
        with tf.variable_scope('loss'):
            if len(logits.shape) > 2:
                logits = tf.layers.flatten(logits)
            if len(self.inputs.shape) > 2:
                self.inputs = tf.layers.flatten(self.inputs)
            reconstruct_loss = -tf.reduce_mean(tf.reduce_sum(self.inputs * tf.log(epsilon + logits) + (1 - self.inputs) * tf.log(epsilon + 1 - logits), axis=1))
            KL_divergence = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + self.var - tf.square(self.mu - tf.exp(self.var)), axis=1))
            return reconstruct_loss + KL_divergence
        

class CVAE(VAE):
    def __init__(self, 
                 encode=None,
                 decode=None,
                 denoise=False,
                 name='VAE',
                 out_dim=10,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 trainable=False
                 ):
        super().__init__(encode=encode, decode=decode, name=name, out_dim=out_dim, opt=opt, lr=lr, l2_reg=l2_reg, l2_reg_scale=l2_reg_scale, trainable=trainable)

    def combine_distribution(self, z, labels=None):
        """
        latent vector Z と label情報をConcatする

        parameters
        ----------
        z : 一様分布から生成した乱数

        label : labelデータ

        returns
        ----------
        image : labelをconcatしたデータ
        """
        assert labels is not None
        return tf.concat([z, labels], axis=1)
    
    def combine_image(self, image, labels=None):
        """
        Generatorで生成した画像とlabelをConcatする
        
        parameters
        ----------
        image : Generatorで生成した画像

        label : labelデータ

        returns
        ----------
        image : labelをconcatしたデータ

        """
        assert labels is not None
        labels = tf.reshape(labels, [-1, 1, 1, self.class_num])
        label_image = tf.ones((labels.shape[0], self.size, self.size, self.class_num))
        label_image = tf.multiply(labels, label_image)
        return tf.concat([image, label_image], axis=3)

    def inference(self, outputs, labels, reuse=False):
        with tf.variable_scope(self.name):
            if self.denoise:
                outputs = self.noise(outputs)
            outputs = self.combine_distribution(outputs, labels)
            self.mu, self.var = self.Encode(outputs, reuse)
            compose_img = self.re_parameterization(self.mu, self.var)
            compose_img = self.combine_image(compose_img, labels)
            outputs = tf.clip_by_value(self.Decode(compose_img, reuse), 1e-8, 1 - 1e-8)
            
            return outputs

    def test_inference(self, outputs, labels, reuse):
        return self.inference(outputs, labels, reuse)