import os, sys
import numpy as np
import tensorflow as tf
from GAN.dcgan import Discriminator, Generator, DCGAN

class WGAN(DCGAN):
    """
    Wasserstein GAN
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def weight_clipping(self):
        with tf.name_scope('weight_clipping'):
            for p in self.D.weights:
                p.assign(tf.clip_by_value(p, -0.01, 0.01))
        return
    
    def generator_loss(self, fake_logit):
        g_loss = -tf.reduce_mean(fake_logit + self.eps)
        return g_loss

    def discriminator_loss(self, fake_logit, real_logit):
        d_loss = -tf.reduce_mean(real_logit + self.eps) + tf.reduce_mean(fake_logit + self.eps)
        return d_loss

    def discriminator_optimize(self, d_loss, tape=None):
        assert tape is not None, 'please set tape in optimize'
        d_grads = tape.gradient(d_loss, self.D.trainable_variables)
        self.d_optimizer.method.apply_gradients(zip(d_grads, self.D.trainable_variables))
        self.weight_clipping()
        return


class WGANGP(WGAN):
    """
    Wesserstein GAN + Gradient penalty
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_penalty_weight = tf.constant(10., dtype=tf.float32)

    def inference(self, inputs, batch_size, labels=None, trainable=True):
        fake_logit, real_logit, fake_img = super().inference(inputs, batch_size, labels, trainable)
        e = tf.random.uniform([batch_size, 1, 1, 1], 0, 1)
        x_hat = e * inputs + (1 - e) * fake_img
        x_hat_logit = self.D(x_hat, trainable=trainable)
        grad = tf.gradients(x_hat_logit, [x_hat])[0]
        return [fake_logit, grad], real_logit, fake_img

    def loss(self, fake_logits, real_logit):
        fake_logit, grad = fake_logits[0], fake_logits[1]
        with tf.name_scope('loss'):
            with tf.name_scope('Discriminator_loss'):
                d_loss = tf.reduce_mean(fake_logit - real_logit)
                gp = tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3])) - 1))
                d_loss += self.grad_penalty_weight * gp
            with tf.name_scope('Generator_loss'):
                g_loss = -tf.reduce_mean(fake_logit + self.eps)
            return d_loss, g_loss

    def accuracy(self, real_logit, fake_logits):
        fake_logit = fake_logits[0]
        return  (tf.reduce_mean(tf.cast(fake_logit < 0.5, tf.float32)) + tf.reduce_mean(tf.cast(real_logit > 0.5, tf.float32))) / 2.
