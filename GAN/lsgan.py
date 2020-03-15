import os, sys
import numpy as np
import tensorflow as tf
from GAN.dcgan import Discriminator, Generator, DCGAN

class LSGAN(DCGAN):
    """
    Least Square GAN
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss(self, fake_logit, real_logit):
        # discriminator loss
        d_loss = tf.reduce_mean(0.5 * tf.square(real_logit - tf.ones_like(real_logit)) + 0.5 * tf.square(fake_logit))

        # generator loss
        g_loss = tf.reduce_mean(0.5 * tf.square(fake_logit - tf.ones_like(fake_logit)))
        return d_loss, g_loss
