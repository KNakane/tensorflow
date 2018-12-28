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
                 z_dim=100,
                 name='GAN',
                 opt=Adam,
                 lr=0.001,
                 trainable=False,
                 interval=5):
        super().__init__(name=name, opt=opt, lr=lr, trainable=trainable)
        gen_model, dis_model = self.build()
        self.generator = Generator(model=gen_model, opt=opt, trainable=trainable)
        self.discriminator = Discriminator(model=dis_model, opt=opt, trainable=trainable)
        self.gen_train_interval = interval
        self._z_dim = z_dim

    def build(self):
        gen_model = [['fc', 7*7*512, None],
                     ['BN', 1],
                     ['ReLU'],
                     ['reshape', [-1, 7, 7, 512]],
                     ['deconv', 4, 256, 2, tf.nn.relu],
                     ['BN', 2],
                     ['deconv', 4, 128, 2, tf.nn.relu],
                     ['BN', 3],
                     ['deconv', 4, 1, 1, None]]

        dis_model = [['conv', 4, 64, 2, tf.nn.relu],
                     ['conv', 4, 128, 2, tf.nn.relu],
                     ['BN', 1],
                     ['conv', 4, 256, 2, tf.nn.relu],
                     ['BN', 2],
                     ['conv', 4, 512, 2, tf.nn.relu],
                     ['BN', 3],
                     ['fc', 2, None]]

        return gen_model, dis_model

    def inference(self, inputs, batch_size):
        with tf.variable_scope(self.name):
            self.z = tf.random_normal((batch_size, self._z_dim), dtype=tf.float32)
            gens = self.generator.inference(self.z)
            
            dis_true = self.discriminator.inference(inputs)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                dis_fake = self.discriminator.inference(gens)
            return dis_true, dis_fake, gens

    def predict(self):
        return self.generator.inference(self.z)

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