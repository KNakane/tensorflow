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

        real_logit = tf.nn.sigmoid(self.D(inputs))
        fake_logit = tf.nn.sigmoid(self.D(fake_img, reuse=True))
        return real_logit, fake_logit, fake_img

    def loss(self, real_logit, fake_logit):
        d_loss = -tf.reduce_mean(tf.log(real_logit) + tf.log(1. - fake_logit))
        g_loss = -tf.reduce_mean(tf.log(fake_logit))
        return d_loss, g_loss

class DCGAN(GAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

class WGAN(GAN):
    """
    Wesserstein GAN
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def inference(self, inputs, batch_size):
        self.z = tf.random_normal((batch_size, self._z_dim), dtype=tf.float32)
        fake_img = self.G(self.z)

        real_logit = self.D(inputs)
        fake_logit = self.D(fake_img, reuse=True)
        return real_logit, fake_logit, fake_img

    def weight_clipping(self):
        clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.D.weight]
        return clip_D

    def loss(self, real_logit, fake_logit):
        d_loss = -(tf.reduce_mean(real_logit) - tf.reduce_mean(fake_logit))
        g_loss = -tf.reduce_mean(fake_logit)
        return d_loss, g_loss

    def optimize(self, d_loss, g_loss, global_step=None):
        clip_D = self.weight_clipping()
        opt_D = self.optimizer.optimize(loss=d_loss, global_step=global_step, var_list=self.D.var)
        opt_G = self.optimizer.optimize(loss=g_loss, global_step=global_step, var_list=self.G.var)
        return tf.group([opt_D, clip_D]), opt_G

class WGAN_GP(WGAN):
    """
    Wesserstein GAN + Gradient penalty
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CGAN(GAN):
    """
    Conditional GAN
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def combine_distribution(self, z, labels):
        assert labels is not None
        return tf.concat([z, labels], axis=0)
    
    def combine_image(self, image, labels):
        assert labels is not None
        return tf.concat([image, labels], axis=3)

    def inference(self, inputs, batch_size, labels=None):
        z = tf.random_normal((batch_size, self._z_dim), dtype=tf.float32)
        self.z = self.combine_distribution(z, labels)

        fake_img = self.G(self.z)
        fake = self.combine_image(fake_img, labels)


        real_logit = self.D(inputs)
        fake_logit = self.D(fake, reuse=True)
        return real_logit, fake_logit, fake_img

