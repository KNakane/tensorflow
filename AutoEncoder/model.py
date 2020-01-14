import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from utility.optimizer import *

class Encoder(Model):
    def __init__(self, 
                 model=None,
                 name='Encoder',
                 out_dim=10,
                 l2_reg=False,
                 l2_reg_scale=0.0001
                 ):
        super().__init__()
        self.model_name = name
        self.out_dim = out_dim
        self.l2_regularizer = l2_reg_scale if l2_reg else None
        self._build()

    def _build(self):
        self.flat = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.fc2 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.fc3 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.out = tf.keras.layers.Dense(self.out_dim, activation='relu', kernel_regularizer=self.l2_regularizer)
        return

    def __call__(self, x, trainable=True):
        x = self.flat(x, training=trainable)
        x = self.fc1(x, training=trainable)
        x = self.fc2(x, training=trainable)
        x = self.fc3(x, training=trainable)
        x = self.out(x, training=trainable)
        return x


class Decoder(Model):
    def __init__(self, 
                 model=None,
                 name='Encoder',
                 size=28,
                 channel=1,
                 l2_reg=False,
                 l2_reg_scale=0.0001
                 ):
        super().__init__()
        self.model_name = name
        self.size = size
        self.channel = channel
        self.l2_regularizer = l2_reg_scale if l2_reg else None
        self._build()

    def _build(self):
        self.fc1 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.fc2 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.fc3 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.fc4 = tf.keras.layers.Dense(self.size**2 * self.channel, activation='sigmoid', kernel_regularizer=self.l2_regularizer)
        self.out = tf.keras.layers.Reshape((self.size,self.size,self.channel))
        return

    def __call__(self, x, trainable=True):
        x = self.fc1(x, training=trainable)
        x = self.fc2(x, training=trainable)
        x = self.fc3(x, training=trainable)
        x = self.fc4(x, training=trainable)
        x = self.out(x, training=trainable)
        return x


class AutoEncoder(Model):
    def __init__(self, 
                 denoise=False,
                 name='AutoEncoder',
                 size=28,
                 channel=1,
                 out_dim=10,
                 class_dim=10,
                 opt="Adam",   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001
                 ):
        super().__init__()
        self.model_name = name
        self.encode = Encoder(out_dim=out_dim, l2_reg=l2_reg, l2_reg_scale=l2_reg_scale)
        self.decode = Decoder(size=size, channel=channel, l2_reg=l2_reg, l2_reg_scale=l2_reg_scale)
        self.denoise = denoise
        self.class_dim = class_dim
        self.size = size
        self.channel =  channel
        self.optimizer = eval(opt)(learning_rate=lr, decay_step=None, decay_rate=0.95)
        self.l2_regularizer = l2_reg_scale if l2_reg else None

    def noise(self, outputs):
        outputs += tf.random_normal(tf.shape(outputs))
        return tf.clip_by_value(outputs, 1e-8, 1 - 1e-8)
    
    def inference(self, outputs, trainable=True):
        self.inputs = outputs
        if self.denoise:
            outputs = self.noise(outputs)
        outputs = self.encode(outputs, trainable)
        outputs = self.decode(outputs, trainable)
        return outputs

    def test_inference(self, outputs, trainable=False):
        return self.inference(outputs, trainable=trainable)

    def predict(self, outputs, trainable=False):
        return self.test_inference(outputs, trainable)

    def loss(self, logits, anser):
        loss = tf.reduce_mean(tf.square(logits - anser))
        return loss
    
    def optimize(self, loss, tape=None):
        assert tape is not None, 'please set tape in opmize'
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.method.apply_gradients(zip(grads, self.trainable_variables))
        return

    def accuracy(self, logits, answer, eps=1e-10):
        marginal_likelihood = tf.reduce_mean(answer * tf.math.log(logits + eps) + (1 - answer) * tf.math.log(1 - logits + eps),
                                            [1, 2])
        neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)
        return neg_loglikelihood


class VAE(AutoEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def inference(self, outputs, trainable=True):
        self.inputs = outputs
        if self.denoise:
            outputs = self.noise(outputs)
        outputs = self.encode(outputs, trainable)
        self.mu, self.var = tf.split(outputs, num_or_size_splits=2, axis=1)
        compose_img = self.re_parameterization(self.mu, self.var)
        outputs = tf.clip_by_value(self.decode(compose_img, trainable), 1e-8, 1 - 1e-8)
        return outputs

    def test_inference(self, outputs, trainable=False):
        return self.inference(outputs, trainable)

    def predict(self, outputs, trainable=False):
        x = np.linspace(0, 1, 20)
        y = np.flip(np.linspace(0, 1, 20))
        z = []
        for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                z.append(np.array([xi, yi]))
        z = np.stack(z)
        outputs = tf.clip_by_value(self.decode(tf.convert_to_tensor(z, dtype=tf.float32), trainable), 1e-8, 1 - 1e-8)
        return outputs

    def loss(self, logits, answer):
        epsilon = 1e-10
        if len(logits.shape) > 2:
            logits = tf.reshape(logits, [logits.shape[0], -1])
        if len(answer.shape) > 2:
            answer = tf.reshape(answer, [answer.shape[0], -1])
        with tf.name_scope('reconstruct_loss'):
                reconstruct_loss = -tf.reduce_sum(answer * tf.math.log(epsilon + logits) + (1 - answer) * tf.math.log(epsilon + 1 - logits), axis=1)
        with tf.name_scope('KL_divergence'):
            KL_divergence = 0.5 * tf.reduce_sum(tf.square(self.mu) + tf.exp(self.var)**2 - 2 * self.mu -1, axis=1)
        return tf.reduce_mean(reconstruct_loss + KL_divergence)

    def re_parameterization(self, mu, var):
        """
        Reparametarization trick
        parameters
        ---
        mu, var : numpy array or tensor
            mu is average, var is variance
        """
        with tf.name_scope('re_parameterization'):
            eps = tf.random.normal(tf.shape(var), dtype=tf.float32)
            return mu + tf.exp(0.5*var) * eps        

    def gaussian(self, batch_size, n_dim, mean=0, var=1):
        z = tf.random.normal(shape=(batch_size, n_dim), mean=mean, stddev=var)
        return z

class CVAE(VAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def inference(self, outputs, labels=None, trainable=True):
        self.inputs = outputs
        if self.denoise:
            outputs = self.noise(outputs)
        outputs = self.combine_image(outputs, labels)
        outputs = self.encode(outputs, trainable)
        self.mu, self.var = tf.split(outputs, num_or_size_splits=2, axis=1)
        compose_img = self.re_parameterization(self.mu, self.var)
        compose_img = self.combine_distribution(compose_img, labels)
        outputs = tf.clip_by_value(self.decode(compose_img, trainable), 1e-8, 1 - 1e-8)
        return outputs

    def test_inference(self, outputs, labels=None, trainable=False):
        return self.inference(outputs, labels, trainable)

    def predict(self, outputs, trainable=False):
        x = np.linspace(0, 1, 20, dtype=np.float32)
        y = np.flip(np.linspace(0, 1, 20, dtype=np.float32))
        z = []
        for _, xi in enumerate(x):
            for _, yi in enumerate(y):
                z.append(np.array([xi, yi]))
        z = np.stack(z)

        indices = np.array([x % self.class_dim for x in range(z.shape[0])], dtype=np.int32)
        labels = np.identity(self.class_dim, dtype=np.float32)[indices]
        #labels = tf.one_hot(indices, depth=self.class_dim, dtype=tf.float32)

        compose_img = self.combine_distribution(z, labels)

        outputs = tf.clip_by_value(self.decode(compose_img, trainable), 1e-8, 1 - 1e-8)
        return outputs

    def combine_distribution(self, z, labels=None):
        """
        latent vector Z と label情報をConcatする
        parameters
        ----------
        z : 一様分布から生成した乱数
        label : labelデータ
        returns
        ----------
        image : labelをconcatしたデータ
        """
        assert labels is not None
        return tf.concat([z, labels], axis=1)

    def combine_image(self, image, labels=None):
        """
        Generatorで生成した画像とlabelをConcatする
        
        parameters
        ----------
        image : Generatorで生成した画像
        label : labelデータ
        returns
        ----------
        image : labelをconcatしたデータ
        """
        assert labels is not None
        labels = tf.reshape(labels, [-1, 1, 1, self.class_dim])
        label_image = tf.ones((labels.shape[0], self.size, self.size, self.class_dim))
        label_image = tf.multiply(labels, label_image)
        return tf.concat([image, label_image], axis=3)