import os,sys
sys.path.append('./network')
sys.path.append('./utility')
import tensorflow as tf
import math
from module import Module
from based_gan import BasedGAN, Discriminator, Generator

class DCGAN(BasedGAN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self):
        """
        s_h = 28
        s_h2 = self.conv_out_size_same(s_h, 2)
        s_h4 = self.conv_out_size_same(s_h2, 2)
        s_h8 = self.conv_out_size_same(s_h4, 2)
        s_h16 = self.conv_out_size_same(s_h8, 2)

        gen_model = [['fc', 512*s_h16*s_h16, None],
                     ['reshape', [-1, s_h16, s_h16, 512]],
                     ['BN'],
                     ['ReLU'],
                     ['deconv', s_h8, 256, 2, None],
                     ['BN'],
                     ['ReLU'],
                     ['deconv', s_h4, 128, 2, None],
                     ['BN'],
                     ['ReLU'],
                     ['deconv', s_h2, 64, 2, None],
                     ['BN'],
                     ['ReLU'],
                     ['deconv', s_h, 1, 2, None],
                     ['tanh']]


        dis_model = [['conv', 5, 64, 2, tf.nn.leaky_relu],
                     ['conv', 5, 128, 2, None],
                     ['BN'],
                     ['Leaky_ReLU'],
                     ['conv', 5, 256, 2, None],
                     ['BN'],
                     ['Leaky_ReLU'],
                     ['conv', 5, 512, 2, None],
                     ['BN'],
                     ['Leaky_ReLU'],
                     ['fc', 1, None]
                     ]
        """

        gen_model = [
            ['fc', 4*4*512, None],
            ['reshape', [-1, 4, 4, 512]],
            ['BN'],
            ['Leaky_ReLU'],
            ['deconv', 5, 256, 3, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['deconv', 5, 128, 2, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['deconv', 5, 1, 1, None, 'valid'],
            ['tanh']]

        dis_model = [
            ['conv', 5, 64, 2, None],
            ['Leaky_ReLU'],
            ['conv', 5, 128, 2, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['reshape', [-1, 4*4*256]],
            ['fc', 1, None]
        ]

        self.D = Discriminator(dis_model)
        self.G = Generator(gen_model)

    def inference(self, inputs, batch_size):
        self.z = tf.random_normal((batch_size, self._z_dim), dtype=tf.float32)
        fake_img = self.G(self.z)

        real_logit = tf.nn.sigmoid(self.D(inputs))
        fake_logit = tf.nn.sigmoid(self.D(fake_img, reuse=True))
        return real_logit, fake_logit, fake_img

    def predict(self, inputs):
        return self.G(inputs, reuse=True)

    def loss(self, real_logit, fake_logit):
        d_loss = -tf.reduce_mean(tf.log(real_logit) + tf.log(1. - fake_logit))
        g_loss = -tf.reduce_mean(tf.log(fake_logit))
        return d_loss, g_loss

