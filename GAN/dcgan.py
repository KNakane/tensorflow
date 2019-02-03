import os,sys
sys.path.append('./network')
sys.path.append('./utility')
import tensorflow as tf
import math
from module import Module
from optimizer import *
from based_gan import BasedGAN
from generator import Generator
from discriminator import Discriminator

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

        self.generator = Generator(model=gen_model, opt=self.opt, lr=self.lr, scope_name=self.name, trainable=self.trainable)
        self.discriminator = Discriminator(model=dis_model, opt=self.opt, lr=self.lr, scope_name=self.name, trainable=self.trainable)

    def inference(self, inputs, batch_size):
        with tf.variable_scope(self.name):
            self.z = tf.random_normal((batch_size, self._z_dim), dtype=tf.float32)
            self.G = self.generator.inference(self.z)
            
            self.D_logits = self.discriminator.inference(inputs)               # input Correct data
            self.D_logits_ = self.discriminator.inference(self.G, reuse=True) # input Fake data

            return self.D_logits, self.D_logits_, self.G

    def predict(self):
        return self.generator.inference(self.z)

    def loss(self, img, fake_img):
        real = tf.nn.sigmoid(img)
        fake = tf.nn.sigmoid(fake_img)
        """
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(img)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake_img)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.ones_like(fake_img)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        """
        self.d_loss = - (tf.reduce_mean(tf.log(real + self.eps)) + tf.reduce_mean(tf.log(1 - fake + self.eps)))
        self.g_loss = - tf.reduce_mean(tf.log(fake + self.eps))

        return self.d_loss, self.g_loss

    def optimize(self, dis_loss, gen_loss):
        global_steps = tf.train.get_or_create_global_step()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            return self.generator.optimize(loss=gen_loss, global_step=global_steps), self.discriminator.optimize(loss=dis_loss, global_step=global_steps)
    

