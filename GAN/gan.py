import os,sys
sys.path.append('./network')
sys.path.append('./utility')
import tensorflow as tf
import math
from cnn import CNN
from optimizer import *
from generator import Generator
from discriminator import Discriminator


class GAN(CNN):
    def __init__(self,
                 z_dim=100,
                 name='GAN',
                 opt=Adam,
                 lr=0.001,
                 trainable=False,
                 interval=2):
        super().__init__(name=name, opt=opt, lr=lr, trainable=trainable)
        gen_model, dis_model = self.build()
        self.generator = Generator(model=gen_model, opt=opt, trainable=trainable)
        self.discriminator = Discriminator(model=dis_model, opt=opt, trainable=trainable)
        self.gen_train_interval = interval
        self._z_dim = z_dim
        if self._trainable:
            self.D_optimizer = eval(opt)(learning_rate=lr)
            self.G_optimizer = eval(opt)(learning_rate=lr)

    def conv_out_size_same(self, size, stride):
        return int(math.ceil(float(size) / float(stride)))

    def build(self):
        s_h = 28
        s_h2 = self.conv_out_size_same(s_h, 2)
        s_h4 = self.conv_out_size_same(s_h2, 2)
        s_h8 = self.conv_out_size_same(s_h4, 2)
        s_h16 = self.conv_out_size_same(s_h8, 2)

        gen_model = [['fc', 512*s_h16*s_h16, None],
                     ['reshape', [-1, s_h16, s_h16, 512]],
                     ['BN', 1],
                     ['ReLU'],
                     ['deconv', s_h8, 256, 2, None],
                     ['BN', 2],
                     ['ReLU'],
                     ['deconv', s_h4, 128, 2, None],
                     ['BN', 3],
                     ['ReLU'],
                     ['deconv', s_h2, 64, 2, None],
                     ['BN', 4],
                     ['ReLU'],
                     ['deconv', s_h, 1, 2, None],
                     ['tanh']]


        dis_model = [['conv', 5, 64, 2, tf.nn.leaky_relu],
                     ['conv', 5, 128, 2, None],
                     ['BN', 1],
                     ['Leaky_ReLU'],
                     ['conv', 5, 256, 2, None],
                     ['BN', 2],
                     ['Leaky_ReLU'],
                     ['conv', 5, 512, 2, None],
                     ['BN', 3],
                     ['Leaky_ReLU'],
                     ['fc', 1, None]
                     ]

        return gen_model, dis_model

    def inference(self, inputs, batch_size):
        with tf.variable_scope(self.name):
            z = tf.random_normal((batch_size, self._z_dim), dtype=tf.float32)
            self.z = tf.reshape(z, [self.batch_size, 1, 1, self._z_dim])
            self.G = self.generator.inference(self.z)
            
            self.D, self.D_logits = self.discriminator.inference(inputs)
            self.D_, self.D_logits_ = self.discriminator.inference(self.G, reuse=True)

            return self.D, self.D_logits, self.D_, self.D_logits_, self.G

    def predict(self):
        return self.generator.inference(self.z)

    def loss(self, D, D_logits, D_, D_logits_):
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.zeros_like(D_)))
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.ones_like(D_)))
        d_loss = d_loss_real + d_loss_fake

        return d_loss, g_loss

    def optimize(self, dis_loss, gen_loss, global_steps=None):
        return self.G_optimizer.optimize(loss=gen_loss, global_step=global_steps), self.D_optimizer.optimize(loss=dis_loss, global_step=global_steps)
        """
        def gen_train(loss, global_steps):
            return self.G_optimizer.optimize(loss=loss, global_step=global_steps)
        judge = tf.cast(global_steps % self.gen_train_interval == 0, tf.bool)
        gen_train_op = tf.cond(judge, lambda: gen_train(gen_loss, global_steps), lambda: tf.no_op())
        with tf.control_dependencies([gen_train_op]):
            return gen_train_op + self.D_optimizer.optimize(loss=dis_loss, global_step=global_steps)
        """
        
    