import os,sys
import tensorflow as tf
from based_gan import BasedGAN, Discriminator, Generator

class GAN(BasedGAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self):
        
        self.generator_model = [['fc', 256, tf.nn.leaky_relu],
                                ['BN'],
                                ['fc', 512, tf.nn.leaky_relu],
                                ['BN'],
                                ['fc', 1024, tf.nn.leaky_relu],
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
        pass

    def inference(self, inputs, batch_size):
        self.z = tf.random_normal((batch_size, self._z_dim), dtype=tf.float32)
        fake_img = self.G(self.z)

        real_logit = self.D(inputs)
        fake_logit = self.D(fake_img, reuse=True)#tf.nn.sigmoid(self.D(fake_img))
        #real_logit = self.D(inputs, reuse=True)#tf.nn.sigmoid(self.D(inputs, reuse=True))
        return real_logit, fake_logit, fake_img

    def loss(self, real_logit, fake_logit):
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit, labels=tf.ones_like(real_logit)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.zeros_like(fake_logit)))
        d_loss = d_loss_real + d_loss_fake
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.ones_like(fake_logit)))
        """
        d_loss = - (tf.reduce_mean(tf.log(real_logit + self.eps)) + tf.reduce_mean(tf.log(tf.ones_like(fake_logit) - fake_logit + self.eps)))
        g_loss = - tf.reduce_mean(tf.log(fake_logit + self.eps))
        """
        return d_loss, g_loss
    
    def optimize(self, d_loss, g_loss):
        global_step = tf.train.get_or_create_global_step()
        opt_D = tf.train.AdamOptimizer(1e-4, beta1=0.5).minimize(d_loss, global_step, var_list=self.D.var)
        opt_G = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(g_loss, global_step, var_list=self.G.var)
        return opt_D, opt_G


