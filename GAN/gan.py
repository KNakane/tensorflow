import os,sys
import numpy as np
import tensorflow as tf
sys.path.append('./utility')
from optimizer import *
from based_gan import BasedGAN, Discriminator, Generator, Classifier

class GAN(BasedGAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = 1e-14
        self.name = 'GAN'

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

        self.D = Discriminator(self.discriminator_model, self._l2_reg, self.l2_reg_scale)
        self.G = Generator(self.generator_model, self._l2_reg, self.l2_reg_scale)
        self.G_ = Generator(self.generator_model, trainable=False)

    def predict(self, inputs, batch_size, index=None):
        if self.conditional:
            indices = index if index is not None else np.array([x%self.class_num for x in range(batch_size)],dtype=np.int32)
            labels = tf.one_hot(indices, depth=self.class_num, dtype=tf.float32)
            inputs = self.combine_distribution(inputs, labels)
        return self.G_(inputs, reuse=True)

    def inference(self, inputs, batch_size, labels=None):
        self.z = tf.random_normal((batch_size, self._z_dim), dtype=tf.float32)
        fake_img = self.G(self.combine_distribution(self.z, labels) if self.conditional else self.z)
        
        if self.conditional and labels is not None:
            
            fake_img = self.combine_image(fake_img, labels)
            inputs = self.combine_image(inputs, labels)
            """
            
            fake_img = self.combine_binary_image(fake_img, labels)
            inputs = self.combine_binary_image(inputs, labels)
            """

        real_logit = tf.nn.sigmoid(self.D(inputs))
        fake_logit = tf.nn.sigmoid(self.D(fake_img, reuse=True))
        return real_logit, fake_logit#, fake_img

    def loss(self, real_logit, fake_logit):
        with tf.variable_scope('loss'):
            with tf.variable_scope('Discriminator_loss'):
                d_loss = -tf.reduce_mean(tf.log(real_logit + self.eps) + tf.log(1. - fake_logit + self.eps))
            with tf.variable_scope('Generator_loss'):
                g_loss = -tf.reduce_mean(tf.log(fake_logit + self.eps))
            if self._l2_reg:
                d_loss += self.D.loss()
                g_loss += self.G.loss()
            return d_loss, g_loss

class UnrolledGAN(GAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'UnrolledGAN'

    def set_parameter(self, copy_para):
        return [tf.assign(target_param, param) for param, target_param in zip(copy_para, self.D.weight)]

    def loss(self, real_logit, fake_logit):
        with tf.variable_scope('loss'):
            with tf.variable_scope('Discriminator_loss'):
                d_loss = -tf.reduce_mean(tf.log(real_logit + self.eps) + tf.log(1. - fake_logit + self.eps))
            with tf.variable_scope('Generator_loss'):
                g_loss = -tf.reduce_mean(tf.log(fake_logit + self.eps))
            if self._l2_reg:
                d_loss += self.D.loss()
                g_loss += self.G.loss()
            return d_loss, g_loss


class DCGAN(GAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'DCGAN'

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
            ['fc', 4*4*256, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['reshape', [-1, 4, 4, 256]],
            ['deconv', 2, 128, 2, None], # cifar
            ['deconv', 2, 128, 1, None], # cifar
            #['deconv', 5, 256, 3, None], # mnist
            ['BN'],
            ['Leaky_ReLU'],
            ['deconv', 3, 64, 2, None],
            ['deconv', 3, 64, 1, None], # cifar
            ['BN'],
            ['Leaky_ReLU'],
            ['deconv', 5, self.channel, 2, tf.nn.tanh]] # cifar
            #['deconv', 5, self.channel, 1, None, 'valid'], # mnist

        """
        dis_model = [
            ['conv', 5, 64, 2, None],
            ['Leaky_ReLU'],
            ['conv', 5, 128, 2, None],
            ['BN'],
            ['Leaky_ReLU'],
            #['reshape', [-1, 7*7*128]], # mnist
            ['reshape', [-1, 8*8*128]], # cifar10
            ['fc', 1, None]
        ]
        

        gen_model = [
            ['fc', 128 * 8 * 8, tf.nn.relu],
            ['BN'],
            ['Leaky_ReLU'],
            ['reshape', [-1, 8, 8, 128]],
            ['conv', 4, 128, 1, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['deconv', 4, 128, 2, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['conv', 5, 128, 1, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['deconv', 4, 128, 2, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['conv', 5, 128, 1, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['conv', 5, 128, 1, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['conv', 5, self.channel, 1, tf.nn.tanh]
        ]
        """

        dis_model = [
            ['conv', 3, 128, 1, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['conv', 4, 128, 2, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['conv', 4, 128, 2, None],
            ['BN'],
            ['Leaky_ReLU'],
            #['reshape', [-1, 7*7*128]], # mnist
            ['reshape', [-1, 8*8*128]], # cifar10
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
        self.name = 'WGAN'
    
    def build(self):
        gen_model = [
            ['fc', 2*2*512, None],
            ['reshape', [-1, 2, 2, 512]],
            ['BN'],
            ['Leaky_ReLU'],
            ['deconv', 5, 256, 2, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['deconv', 5, 128, 2, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['deconv', 5, 64, 2, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['deconv', 5, self.channel, 2, tf.nn.tanh]]

        dis_model = [
            ['conv', 5, 64, 2, tf.nn.leaky_relu],
            ['conv', 5, 128, 2, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['conv', 5, 256, 2, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['conv', 5, 512, 2, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['reshape', [-1, 2*2*512]],
            ['fc', 1, tf.nn.sigmoid]
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
            with tf.variable_scope('Discriminator_loss'):
                d_loss = - tf.reduce_mean(real_logit + self.eps) + tf.reduce_mean(fake_logit + self.eps)
            with tf.variable_scope('Generator_loss'):
                g_loss = -tf.reduce_mean(fake_logit + self.eps)
            if self._l2_reg:
                d_loss += self.D.loss()
                g_loss += self.G.loss()
            return d_loss, g_loss

    def optimize(self, d_loss, g_loss, global_step=None):
        with tf.variable_scope('Optimizer'):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                clip_D = self.weight_clipping()
                opt_D = self.d_optimizer.optimize(loss=d_loss, global_step=global_step, var_list=self.D.var)
                opt_G = self.g_optimizer.optimize(loss=g_loss, global_step=global_step, var_list=self.G.var)
                return tf.group([opt_D, clip_D]), opt_G

class WGAN_GP(WGAN):
    """
    Wesserstein GAN + Gradient penalty
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'WGAN_GP'

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

        # Gradient Penalty
        e = tf.random_uniform([batch_size, 1, 1, 1], 0, 1)
        x_hat = e * inputs + (1 - e) * fake_img

        real_logit = self.D(inputs)
        fake_logit = self.D(fake_img, reuse=True)
        x_hat_logit = self.D(x_hat, reuse=True)
        self.grad = tf.gradients(x_hat_logit, [x_hat])[0]
        return real_logit, fake_logit#, fake_img

    def loss(self, real_logit, fake_logit):
        with tf.variable_scope('loss'):
            d_loss = tf.reduce_mean(fake_logit - real_logit) + 10 * tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_sum(tf.square(self.grad), axis=[1, 2, 3])) - 1))
            g_loss = -tf.reduce_mean(fake_logit)
            if self._l2_reg:
                d_loss += self.D.loss()
                g_loss += self.G.loss()
            return d_loss, g_loss

    def optimize(self, d_loss, g_loss, global_step=None):
        with tf.variable_scope('Optimizer'):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                opt_D = self.d_optimizer.optimize(loss=d_loss, global_step=global_step, var_list=self.D.var)
                opt_G = self.g_optimizer.optimize(loss=g_loss, global_step=global_step, var_list=self.G.var)
                return opt_D, opt_G

class LSGAN(DCGAN):
    """
    Least Square GAN
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'LSGAN'

    def inference(self, inputs, batch_size, labels=None):
        self.z = tf.random_normal((batch_size, self._z_dim), dtype=tf.float32)
        fake_img = self.G(self.combine_distribution(self.z, labels) if self.conditional else self.z)
        
        if self.conditional and labels is not None:
            
            fake_img = self.combine_image(fake_img, labels)
            inputs = self.combine_image(inputs, labels)
            """

            fake_img = self.combine_binary_image(fake_img, labels)
            inputs = self.combine_binary_image(inputs, labels)
            """

        real_logit = self.D(inputs)
        fake_logit = self.D(fake_img, reuse=True)
        return real_logit, fake_logit#, fake_img

    def loss(self, real_logit, fake_logit):
        with tf.variable_scope('loss'):
            d_loss = tf.reduce_mean(0.5 * tf.square(real_logit - 1) + 0.5 * tf.square(fake_logit))
            g_loss = tf.reduce_mean(0.5 * tf.square(fake_logit - 1))
            if self._l2_reg:
                d_loss += self.D.loss()
                g_loss += self.G.loss()
            return d_loss, g_loss


class ACGAN(DCGAN):
    """
    Auxiliary Classifier Generative Adversarial Network
    """
    def __init__(self, *args, **kwargs):
        kwargs['conditional'] = True
        super().__init__(*args, **kwargs)
        self.name = 'ACGAN'

    def build(self):
        gen_model = [
            ['fc', 4*4*256, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['reshape', [-1, 4, 4, 256]],
            ['deconv', 2, 128, 2, None], # cifar
            ['deconv', 2, 128, 1, None], # cifar
            #['deconv', 5, 256, 3, None], # mnist
            ['BN'],
            ['Leaky_ReLU'],
            ['deconv', 3, 64, 2, None],
            ['deconv', 3, 64, 1, None], # cifar
            ['BN'],
            ['Leaky_ReLU'],
            ['deconv', 5, self.channel, 2, tf.nn.tanh]] # cifar
            #['deconv', 5, self.channel, 1, None, 'valid'], # mnist

        dis_model = [
            ['conv', 5, 64, 2, tf.nn.leaky_relu],
            ['conv', 5, 128, 2, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['conv', 5, 256, 2, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['conv', 5, 512, 2, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['reshape', [-1, 2*2*512]],
            ['fc', 1 + self.class_num, None]
        ]

        self.D = Discriminator(dis_model)
        self.G = Generator(gen_model)
        self.G_ = Generator(gen_model, trainable=False)

    def inference(self, inputs, batch_size, labels=None):
        self.z = tf.random_normal((batch_size, self._z_dim), dtype=tf.float32)
        fake_img = self.G(self.combine_distribution(self.z, labels))
        
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

        return (real_logit, real_recognition), (fake_logit, fake_recognition)

    def loss(self, real_logit, fake_logit, labels):
        with tf.variable_scope('loss'):
            d_loss = -tf.reduce_mean(tf.log(real_logit[0] + self.eps) + tf.log(1. - fake_logit[0] + self.eps))
            g_loss = -tf.reduce_mean(tf.log(fake_logit[0] + self.eps))

            real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=real_logit[1], labels=labels))
            fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fake_logit[1], labels=labels))
            d_loss = d_loss +  real_loss + fake_loss
            g_loss = g_loss + fake_loss
            if self._l2_reg:
                d_loss += self.D.loss()
                g_loss += self.G.loss()
            return d_loss, g_loss

    def evaluate(self, real_logit, fake_logit):
        with tf.variable_scope('Accuracy'):
            return  (tf.reduce_mean(tf.cast(fake_logit[0] < 0.5, tf.float32)) + tf.reduce_mean(tf.cast(real_logit[0] > 0.5, tf.float32))) / 2.


class infoGAN(DCGAN):
    """
    Information Maximizing Generative Adversarial Networks
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'infoGAN'

    def build(self):
        gen_model = [
            ['fc', 4*4*512, None],
            ['reshape', [-1, 4, 4, 512]],
            ['BN'],
            ['Leaky_ReLU'],
            #['deconv', 5, 256, 2, None], # cifar
            ['deconv', 5, 256, 3, None], # mnist
            ['BN'],
            ['Leaky_ReLU'],
            ['deconv', 5, 128, 2, None],
            ['BN'],
            ['Leaky_ReLU'],
            #['deconv', 5, self.channel, 2, None], # cifar
            ['deconv', 5, self.channel, 1, None, 'valid'], # mnist
            ['tanh']]

        dis_model = [
            ['conv', 5, 64, 2, None],
            ['Leaky_ReLU'],
            ['conv', 5, 128, 2, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['reshape', [-1, 7*7*128]], # mnist
            #['reshape', [-1, 8*8*128]], # cifar10
            ['fc', 1, None]
        ]

        cls_model = [
            ['conv', 5, 64, 2, None],
            ['Leaky_ReLU'],
            ['conv', 5, 128, 2, None],
            ['BN'],
            ['Leaky_ReLU'],
            ['reshape', [-1, 7*7*128]], # mnist
            #['reshape', [-1, 8*8*128]], # cifar10
            ['fc', 2 + self.class_num, None]
        ]

        self.D = Discriminator(dis_model)
        self.G = Generator(gen_model)
        self.G_ = Generator(gen_model, trainable=False)
        self.C = Classifier(cls_model)

    def predict(self, inputs, batch_size):
        if self.conditional:
            indices = np.array([x%self.class_num for x in range(batch_size)],dtype=np.int32)
            labels = tf.one_hot(indices, depth=self.class_num, dtype=tf.float32)
            inputs = self.combine_distribution(inputs, labels)
        return self.G_(inputs, reuse=True)

    def inference(self, inputs, batch_size, labels=None):
        self.z = tf.random_normal((batch_size, self._z_dim), dtype=tf.float32)
        if labels is None:
            batch_labels = np.random.multinomial(1,
                                                self.class_num * [float(1.0 / self.class_num)],
                                                size=[batch_size])
        else:
            batch_labels = labels
        batch_codes = np.concatenate((batch_labels, np.random.uniform(-1, 1, size=(batch_size, 2))),axis=1)

        fake_img = self.G(self.combine_distribution(self.z, batch_codes))        

        real_logit = tf.nn.sigmoid(self.D(inputs))
        fake_logit = tf.nn.sigmoid(self.D(fake_img, reuse=True))
        return real_logit, fake_logit

    def loss(self, real_logit, fake_logit):
        with tf.variable_scope('loss'):
            with tf.variable_scope('Discriminator_loss'):
                d_loss = -tf.reduce_mean(tf.log(real_logit + self.eps) + tf.log(1. - fake_logit + self.eps))
            with tf.variable_scope('Generator_loss'):
                g_loss = -tf.reduce_mean(tf.log(fake_logit + self.eps))
            if self._l2_reg:
                d_loss += self.D.loss()
                g_loss += self.G.loss()
            return d_loss, g_loss

class DRAGAN(DCGAN):
    """
    On Convergence and Stability of GANs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # DRAGAN parameter
        self.lambd = 0.25
        self.name = 'DRAGAN'

    def get_perturbed_batch(self, minibatch):
        return minibatch + 0.5 * tf.keras.backend.std(minibatch) * tf.random_uniform(shape=minibatch.get_shape())

    def inference(self, inputs, batch_size, labels=None):
        self.z = tf.random_normal((batch_size, self._z_dim), dtype=tf.float32)
        fake_img = self.G(self.combine_distribution(self.z, labels) if self.conditional else self.z)
        
        if self.conditional and labels is not None:

            fake_img = self.combine_image(fake_img, labels)
            inputs = self.combine_image(inputs, labels)
            """

            fake_img = self.combine_binary_image(fake_img, labels)
            inputs = self.combine_binary_image(inputs, labels)
            """
            

        real_logit = tf.nn.sigmoid(self.D(inputs))
        fake_logit = tf.nn.sigmoid(self.D(fake_img, reuse=True))
        
        alpha = tf.random_uniform(shape=inputs.get_shape(), minval=0.,maxval=1.)
        p_inputs = self.get_perturbed_batch(inputs)
        differences = p_inputs - inputs
        interpolates = inputs + (alpha * differences)
        D_inter = self.D(interpolates, reuse=True)
        self.grad = tf.gradients(D_inter, [interpolates])[0]
        return real_logit, fake_logit

    def loss(self, real_logit, fake_logit):
        with tf.variable_scope('loss'):
            with tf.variable_scope('Discriminator_loss'):
                d_loss = -tf.reduce_mean(tf.log(real_logit + self.eps) + tf.log(1. - fake_logit + self.eps))
            with tf.variable_scope('Generator_loss'):
                g_loss = -tf.reduce_mean(tf.log(fake_logit + self.eps))
            if self._l2_reg:
                d_loss += self.D.loss()
                g_loss += self.G.loss()

            # DRAGAN Loss
            slopes = tf.sqrt(tf.reduce_sum(tf.square(self.grad), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            d_loss += self.lambd * gradient_penalty

            return d_loss, g_loss


class VGAN(DCGAN):
    """
    Variational Discriminator Bottleneck GAN
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)