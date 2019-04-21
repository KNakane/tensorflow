import os,sys
import numpy as np
import tensorflow as tf
from based_gan import BasedGAN, Discriminator, Generator

class GAN(BasedGAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = 1e-14

    def build(self):
        
        self.generator_model = [['fc', 128, tf.nn.leaky_relu],
                                ['BN'],
                                ['fc', 256, tf.nn.leaky_relu],
                                ['BN'],
                                ['fc', self.size*self.size*self.channel, tf.nn.sigmoid],
                                ['reshape', [-1, self.size, self.size, self.channel]]]

        self.discriminator_model = [['fc', 1024, tf.nn.leaky_relu],
                                    ['BN'],
                                    ['fc', 512, tf.nn.leaky_relu],
                                    ['BN'],
                                    ['fc', 256, tf.nn.leaky_relu],
                                    ['BN'],
                                    ['fc', 1, None]]

        self.D = Discriminator(self.discriminator_model, self.l2_reg, self.l2_reg_scale)
        self.G = Generator(self.generator_model, self.l2_reg, self.l2_reg_scale)
        self.G_ = Generator(self.generator_model, trainable=False)

    def predict(self, inputs, batch_size):
        if self.conditional:
            indices = np.array([x%self.class_num for x in range(batch_size)],dtype=np.int32)
            labels = tf.one_hot(indices, depth=self.class_num, dtype=tf.float32)
            inputs = self.combine_distribution(inputs, labels)
        return self.G_(inputs, reuse=True)

    def inference(self, inputs, batch_size, labels=None):
        self.z = tf.random_normal((batch_size, self._z_dim), dtype=tf.float32)
        fake_img = self.G(self.combine_distribution(self.z, labels) if self.conditional else self.z)
        
        if self.conditional and labels is not None:
            """
            fake_img = self.combine_image(fake_img, labels)
            inputs = self.combine_image(inputs, labels)
            """

            fake_img = self.combine_binary_image(fake_img, labels)
            inputs = self.combine_binary_image(inputs, labels)
            

        real_logit = tf.nn.sigmoid(self.D(inputs))
        fake_logit = tf.nn.sigmoid(self.D(fake_img, reuse=True))
        return real_logit, fake_logit#, fake_img

    def loss(self, real_logit, fake_logit):
        with tf.variable_scope('loss'):
            d_loss = -tf.reduce_mean(tf.log(real_logit + self.eps) + tf.log(1. - fake_logit + self.eps))
            g_loss = -tf.reduce_mean(tf.log(fake_logit + self.eps))
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
            ['deconv', 5, 256, 3, None],#['deconv', 5, 256, 2, None], #['deconv', 5, 256, 3, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['deconv', 5, 128, 2, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['deconv', 5, self.channel, 1, None, 'valid'],#['deconv', 5, self.channel, 2, None], #['deconv', 5, self.channel, 1, None, 'valid'],
            ['sigmoid']]

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
        self.G_ = Generator(gen_model, trainable=False)

class WGAN(GAN):
    """
    Wesserstein GAN
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def build(self):
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
        self.G_ = Generator(gen_model, trainable=False)
    
    def inference(self, inputs, batch_size, labels=None):
        self.z = tf.random_normal((batch_size, self._z_dim), dtype=tf.float32)
        fake_img = self.G(self.combine_distribution(self.z, labels) if self.conditional else self.z)

        if self.conditional and labels is not None:
            """
            fake_img = self.combine_image(fake_img, labels)
            inputs = self.combine_image(inputs, labels)

            """
            fake_img = self.combine_binary_image(fake_img, labels)
            inputs = self.combine_binary_image(inputs, labels)

        real_logit = self.D(inputs)
        fake_logit = self.D(fake_img, reuse=True)
        return real_logit, fake_logit

    def weight_clipping(self):
        with tf.variable_scope('weight_clipping'):
            clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.D.weight]
            return clip_D

    def loss(self, real_logit, fake_logit):
        with tf.variable_scope('loss'):
            d_loss = -(tf.reduce_mean(real_logit + self.eps) - tf.reduce_mean(fake_logit + self.eps))
            g_loss = -tf.reduce_mean(fake_logit + self.eps)
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

    def build(self):
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
        self.G_ = Generator(gen_model, trainable=False)

    def inference(self, inputs, batch_size, labels=None):
        self.z = tf.random_normal((batch_size, self._z_dim), dtype=tf.float32)
        fake_img = self.G(self.combine_distribution(self.z, labels) if self.conditional else self.z)

        if self.conditional and labels is not None:
            """
            fake_img = self.combine_image(fake_img, labels)
            inputs = self.combine_image(inputs, labels)

            """
            fake_img = self.combine_binary_image(fake_img, labels)
            inputs = self.combine_binary_image(inputs, labels)

        e = tf.random_uniform([batch_size, 1, 1, 1], 0, 1)
        x_hat = e * inputs + (1 - e) * fake_img

        real_logit = self.D(inputs)
        fake_logit = self.D(fake_img, reuse=True)
        x_hat_logit = self.D(x_hat, reuse=True)
        self.grad = tf.gradients(x_hat_logit, x_hat)[0]
        return real_logit, fake_logit#, fake_img

    def loss(self, real_logit, fake_logit):
        d_loss = tf.reduce_mean(fake_logit - real_logit) + 10 * tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_sum(tf.square(self.grad), axis=[1, 2, 3])) - 1))
        g_loss = -tf.reduce_mean(fake_logit)
        return d_loss, g_loss

    def optimize(self, d_loss, g_loss, global_step=None):
        opt_D = self.optimizer.optimize(loss=d_loss, global_step=global_step, var_list=self.D.var)
        opt_G = self.optimizer.optimize(loss=g_loss, global_step=global_step, var_list=self.G.var)
        return opt_D, opt_G

class LSGAN(GAN):
    """
    Least Square GAN
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss(self, real_logit, fake_logit):
        with tf.variable_scope('loss'):
            d_loss = tf.reduce_mean(0.5 * tf.square(real_logit - 1) + 0.5 * tf.square(fake_logit))
            g_loss = tf.reduce_mean(0.5 * tf.square(fake_logit - 1))
            return d_loss, g_loss


class ACGAN(DCGAN):
    """
    Auxiliary Classifier Generative Adversarial Network
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conditional = True

    def build(self):
        gen_model = [
            ['fc', 4*4*512, None],
            ['reshape', [-1, 4, 4, 512]],
            ['BN'],
            ['Leaky_ReLU'],
            ['deconv', 5, 256, 3, None],#['deconv', 5, 256, 2, None], #['deconv', 5, 256, 3, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['deconv', 5, 128, 2, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['deconv', 5, self.channel, 1, None, 'valid'],#['deconv', 5, self.channel, 2, None], #['deconv', 5, self.channel, 1, None, 'valid'],
            ['sigmoid']]

        dis_model = [
            ['conv', 5, 64, 2, None],
            ['Leaky_ReLU'],
            ['conv', 5, 128, 2, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['reshape', [-1, 4*4*256]],
            ['fc', 1 + self.class_num, None]
        ]

        self.D = Discriminator(dis_model)
        self.G = Generator(gen_model)
        self.G_ = Generator(gen_model, trainable=False)

    def inference(self, inputs, batch_size, labels=None):
        self.z = tf.random_normal((batch_size, self._z_dim), dtype=tf.float32)
        fake_img = self.G(self.combine_distribution(self.z, labels) if self.conditional else self.z)
        
        if self.conditional and labels is not None:
            """
            fake_img = self.combine_image(fake_img, labels)
            inputs = self.combine_image(inputs, labels)

            """
            fake_img = self.combine_binary_image(fake_img, labels)
            inputs = self.combine_binary_image(inputs, labels)
            
        real_logit, real_recognition = tf.split(self.D(inputs), [1, self.class_num], 1)
        fake_logit, fake_recognition = tf.split(self.D(fake_img, reuse=True), [1, self.class_num], 1)

        # Real or Fake
        real_logit = tf.nn.sigmoid(real_logit)
        fake_logit = tf.nn.sigmoid(fake_logit)

        # Recognition
        real_recognition = tf.nn.softmax(real_recognition)
        fake_recognition = tf.nn.softmax(fake_recognition)

        return real_logit, fake_logit

    def loss(self, real_logit, fake_logit, labels):
        with tf.variable_scope('loss'):
            d_loss = -(tf.reduce_mean(real_logit + self.eps) - tf.reduce_mean(fake_logit + self.eps))
            g_loss = -tf.reduce_mean(fake_logit + self.eps)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
            return d_loss, g_loss

    def evaluate(self, real_logit, fake_logit):
        with tf.variable_scope('Accuracy'):
            return  (tf.reduce_mean(tf.cast(fake_logit < 0.5, tf.float32)) + tf.reduce_mean(tf.cast(real_logit > 0.5, tf.float32))) / 2.