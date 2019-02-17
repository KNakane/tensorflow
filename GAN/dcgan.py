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

        self.D = Discriminator(dis_model)
        self.G = Generator(gen_model)

    def inference(self, inputs, batch_size):
        self.z = tf.random_normal((batch_size, self._z_dim), dtype=tf.float32)
        fake_img = self.G(self.z)
        real_logit = self.D(inputs)
        fake_logit = self.D(fake_img, reuse=True)
        return real_logit, fake_logit, fake_img

    def predict(self):
        return self.generator.inference(self.z)

    def loss(self, real_logit, fake_logit):
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit, labels=tf.ones_like(real_logit)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.zeros_like(fake_logit)))
        d_loss = d_loss_real + d_loss_fake
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.ones_like(fake_logit)))

        return d_loss, g_loss

    def optimize(self, d_loss, g_loss):
        global_step = tf.train.get_or_create_global_step()
        opt_D = tf.train.AdamOptimizer(1e-4, beta1=0.5).minimize(d_loss, global_step, var_list=self.D.var)
        opt_G = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(g_loss, global_step, var_list=self.G.var)
        return opt_D, opt_G

