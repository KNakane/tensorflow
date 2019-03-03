import os,sys
import tensorflow as tf
from based_gan import BasedGAN, Discriminator, Generator

class GAN(BasedGAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self):
        
        self.generator_model = [['fc', 128, tf.nn.leaky_relu],
                                ['BN'],
                                ['fc', 256, tf.nn.leaky_relu],
                                ['BN'],
                                ['fc', self.size*self.size*self.channel, tf.nn.tanh],
                                ['reshape', [-1, self.size, self.size, self.channel]]]

        self.discriminator_model = [['fc', 1024, tf.nn.leaky_relu],
                                    ['BN'],
                                    ['fc', 512, tf.nn.leaky_relu],
                                    ['BN'],
                                    ['fc', 256, tf.nn.leaky_relu],
                                    ['BN'],
                                    ['fc', 1, None]]

        self.D = Discriminator(self.discriminator_model)
        self.G = Generator(self.generator_model)

    def predict(self, inputs):
        return self.G(inputs, reuse=True)

    def inference(self, inputs, batch_size):
        self.z = tf.random_normal((batch_size, self._z_dim), dtype=tf.float32)
        fake_img = self.G(self.z)

        real_logit = self.D(inputs)
        fake_logit = self.D(fake_img, reuse=True)
        return real_logit, fake_logit, fake_img

    def loss(self, real_logit, fake_logit):
        d_loss = -tf.reduce_mean(tf.log(real_logit) + tf.log(1. - fake_logit))
        g_loss = -tf.reduce_mean(tf.log(fake_logit))
        return d_loss, g_loss


