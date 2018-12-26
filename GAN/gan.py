import os,sys
sys.path.append('./network')
import tensorflow as tf
from model import DNN
from generator import Generator
from discriminator import Discriminator

class GAN(DNN):
    def __init__(self, ):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def inference(self, inputs, labels, batch_size):
        z = tf.random_normal((batch_size, self._z_dim), dtype=tf.float32)
        gens = self.generator.inference(z, labels=labels)
        
        dis_true = self.discriminator(inputs, labels=labels)
        dis_fake = self.discriminator(gens, labels=labels)

        dis_loss = tf.reduce_mean(tf.nn.relu(1 - disc_true))
        dis_loss += tf.reduce_mean(tf.nn.relu(1 + disc_fake))

        gen_loss = -tf.reduce_mean(dis_fake)

    def loss(self):
        return
