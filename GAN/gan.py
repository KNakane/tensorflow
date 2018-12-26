import os,sys
sys.path.append('./network')
sys.path.append('./utility')
import tensorflow as tf
from model import DNN
from optimizer import *
from generator import Generator
from discriminator import Discriminator


class GAN(DNN):
    def __init__(self, 
                 name='GAN',
                 opt=Adam,
                 lr=0.001,
                 interval=5,
                 trainable=False):
        super().__init__(name=name, opt=opt, lr=lr, trainable=trainable)
        gen_model, dis_model = self.build()
        self.generator = Generator(model=gen_model, trainable=trainable)
        self.discriminator = Discriminator(model=dis_model, trainable=trainable)
        self.gen_train_interval = interval

    def build(self):
        gen_model = [['conv', 5, 32, 1],
                     ['max_pool', 2, 2],
                     ['dropout', 1024, tf.nn.relu, 0.5],
                     ['fc', outdim, None]]

        dis_model = [['conv', 5, 32, 1],
                     ['max_pool', 2, 2],
                     ['dropout', 1024, tf.nn.relu, 0.5],
                     ['fc', outdim, None]]

        return gen_model, dis_model

    def inference(self, inputs, batch_size):
        with tf.variable_scope(self.name):
            z = tf.random_normal((batch_size, self._z_dim), dtype=tf.float32)
            gens = self.generator.inference(z)
            
            dis_true = self.discriminator(inputs=inputs)
            dis_fake = self.discriminator(gens)
            return dis_true, dis_fake

    def loss(self, true, fake):
        dis_loss = tf.reduce_mean(tf.nn.relu(1 - true))
        dis_loss += tf.reduce_mean(tf.nn.relu(1 + fake))

        gen_loss = -tf.reduce_mean(fake)

        return dis_loss, gen_loss

    def optimize(self, dis_loss, gen_loss, global_step=None):
        def gen_train(gen_loss):
            return self.optimizer.optimize(loss=gen_loss, global_step=global_step)

        gen_train_op = tf.cond(global_step % self.gen_train_interval == 0, gen_train, lambda: tf.no_op())
        with tf.control_dependencies([gen_train_op]):
            return self.optimizer.optimize(loss=dis_loss, global_step=global_step)