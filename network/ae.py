# -*- coding: utf-8 -*-
import os,sys
sys.path.append('../utility')
import numpy as np
import tensorflow as tf
from model import Model
from optimizer import *

class Encode(Model):
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


class Decode(Model):
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


class AutoEncoder(Model):
    def __init__(self, 
                 encode=None,
                 decode=None,
                 denoise=False,
                 size=28,
                 channel=1,
                 name='AutoEncoder',
                 out_dim=10,
                 opt="Adam",   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
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
        self.size = size
        self.channel = channel

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

    def test_inference(self, outputs, reuse=True):
        return self.predict(outputs, reuse)

    def predict(self, outputs, reuse=True):
        with tf.variable_scope(self.name):
            if self.denoise:
                outputs = self.noise(outputs)
            outputs = self.encode(outputs, reuse)
            outputs = self.decode_(outputs, reuse)
            return outputs
        
    def loss(self, logits, labels):
        loss = tf.reduce_mean(tf.square(logits - labels))
        return loss

    def evaluate(self, logits, input_data):
        with tf.variable_scope('Accuracy'):
            marginal_likelihood = tf.reduce_sum(input_data * tf.log(logits) + (1 - input_data) * tf.log(1 - logits),
                                            [1, 2])
            neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)
            return neg_loglikelihood


class VAE(AutoEncoder):
    # Based on https://github.com/shaohua0116/VAE-Tensorflow
    def __init__(self, 
                 encode=None,
                 decode=None,
                 denoise=False,
                 size=28,
                 channel=1,
                 name='VAE',
                 out_dim=10,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 trainable=False
                 ):
        super().__init__(encode=encode, decode=decode, denoise=denoise, size=size, channel=channel, name=name, out_dim=out_dim, opt=opt, lr=lr, l2_reg=l2_reg, l2_reg_scale=l2_reg_scale, trainable=trainable)

    def inference(self, outputs, reuse=False):
        with tf.variable_scope(self.name):
            if self.denoise:
                outputs = self.noise(outputs)
            outputs = self.encode(outputs, reuse)
            self.mu, self.var = tf.split(outputs, num_or_size_splits=2, axis=1)
            compose_img = self.re_parameterization(self.mu, self.var)
            outputs = tf.clip_by_value(self.decode(compose_img, reuse), 1e-8, 1 - 1e-8)
            
            return outputs

    def predict(self, outputs, reuse=True):
        with tf.variable_scope(self.name):
            batch_size = tf.constant(outputs.shape[0], dtype=tf.int32)
            compose_img = self.gaussian(batch_size,20)
            outputs = tf.clip_by_value(self.decode_(compose_img, reuse), 1e-8, 1 - 1e-8)
            return outputs
    
    def re_parameterization(self, mu, var):
        with tf.variable_scope('re_parameterization'):
            std = var ** 0.5
            eps = tf.random_normal(tf.shape(var), 0, 1, dtype=tf.float32)
            return mu + tf.sqrt(tf.exp(std)) * eps

    def loss(self, logits, labels):
        epsilon = 1e-10
        with tf.variable_scope('loss'):
            if len(logits.shape) > 2:
                logits = tf.layers.flatten(logits)
            if len(labels.shape) > 2:
                labels = tf.layers.flatten(labels)
            with tf.variable_scope('reconstruct_loss'):
                reconstruct_loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(epsilon + logits) + (1 - labels) * tf.log(epsilon + 1 - logits), axis=1))
            with tf.variable_scope('KL_divergence'):
                KL_divergence = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + self.var - tf.square(self.mu) - tf.exp(self.var), axis=1))
            return reconstruct_loss + KL_divergence

    def gaussian(self, batch_size, n_dim, mean=0, var=1):
        z = tf.random_normal(shape=(batch_size, n_dim), mean=mean, stddev=var)
        return z
        

class CVAE(VAE):
    def __init__(self, 
                 encode=None,
                 decode=None,
                 denoise=False,
                 size=28,
                 channel=1,
                 name='VAE',
                 out_dim=10,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 trainable=False
                 ):
        super().__init__(encode=encode, decode=decode, denoise=denoise, size=size, channel=channel, name=name, out_dim=out_dim, opt=opt, lr=lr, l2_reg=l2_reg, l2_reg_scale=l2_reg_scale, trainable=trainable)
        self.class_num = out_dim

    def inference(self, outputs, labels=None, reuse=False):
        with tf.variable_scope(self.name):
            if self.denoise:
                outputs = self.noise(outputs)
            outputs = self.combine_image(outputs, labels)
            outputs = self.encode(outputs, reuse)
            self.mu, self.var = tf.split(outputs, num_or_size_splits=2, axis=1)
            compose_img = self.re_parameterization(self.mu, self.var)
            compose_img = self.combine_distribution(compose_img, labels)
            outputs = tf.clip_by_value(self.decode(compose_img, reuse), 1e-8, 1 - 1e-8)
            
            return outputs

    def test_inference(self, outputs, labels=None, reuse=True):
        return self.inference(outputs, labels, reuse)

    def predict(self, outputs, reuse=True):
        with tf.variable_scope(self.name):
            batch_size = outputs.shape[0]
            indices = np.array([x%self.class_num for x in range(batch_size)],dtype=np.int32)
            labels = tf.one_hot(indices, depth=self.class_num, dtype=tf.float32)
            compose_img = tf.convert_to_tensor(self.gaussian(batch_size, n_dim=20), dtype=tf.float32)
            compose_img = self.combine_distribution(compose_img, labels)
            outputs = tf.clip_by_value(self.decode_(compose_img, reuse), 1e-8, 1 - 1e-8)
            return outputs

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