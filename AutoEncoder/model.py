import os, sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from AutoEncoder.encoder_decoder import Encoder, Decoder, Conv_Encoder, Conv_Decoder, Discriminator, Conv_Discriminator
from utility.optimizer import *


class AutoEncoder(Model):
    def __init__(self, 
                 denoise=False,
                 conv=False,
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
        Encoder_model = Conv_Encoder if conv else Encoder
        Decoder_model = Conv_Decoder if conv else Decoder
        if name == 'VAE':
            decoder_input = out_dim / 2
        elif name == 'CVAE':
            decoder_input = (out_dim / 2 + class_dim)
        elif name == 'AAE':
            decoder_input = (out_dim + class_dim)
        else:
            decoder_input = out_dim
        self.encode = Encoder_model(input_shape=(size, size, channel + (class_dim * True if name in ['CVAE'] else False)),
                                    out_dim=out_dim,
                                    l2_reg=l2_reg,
                                    l2_reg_scale=l2_reg_scale)
        self.decode = Decoder_model(input_shape=(decoder_input,),
                                    size=size,
                                    channel=channel,
                                    l2_reg=l2_reg,
                                    l2_reg_scale=l2_reg_scale)
        self.denoise = denoise
        self.class_dim = class_dim
        self.size = size
        self.channel =  channel
        self.optimizer = eval(opt)(learning_rate=lr, decay_step=None, decay_rate=0.95)
        self.l2_regularizer = l2_reg_scale if l2_reg else None

    def noise(self, outputs):
        outputs += tf.random.normal(tf.shape(outputs))
        return tf.clip_by_value(outputs, 1e-8, 1 - 1e-8)
    
    def inference(self, outputs, trainable=True):
        if self.denoise:
            outputs = self.noise(outputs)
        outputs = self.encode(outputs, trainable)
        outputs = self.decode(outputs, trainable)
        return outputs

    def test_inference(self, outputs, trainable=False):
        return self.inference(outputs, trainable=trainable)

    def predict(self, outputs, trainable=False):
        return self.test_inference(outputs, trainable)

    def loss(self, logits, answer):
        loss = tf.reduce_mean(tf.square(logits - answer))
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
        if self.denoise:
            outputs = self.noise(outputs)
        outputs = self.encode(outputs, trainable)
        self.mu, self.var = tf.split(outputs, num_or_size_splits=2, axis=1)
        compose_img = self.re_parameterization(self.mu, self.var)
        outputs = tf.clip_by_value(self.decode(compose_img, trainable), 1e-8, 1 - 1e-8)
        return [outputs, compose_img]

    def test_inference(self, outputs, trainable=False):
        compose_img, self.mu, self.var = self.gaussian(outputs.shape[0],2)
        outputs = tf.clip_by_value(self.decode(compose_img, trainable), 1e-8, 1 - 1e-8)
        return [outputs, compose_img]

    def predict(self, outputs, trainable=False):
        x = np.linspace(-1, 1, 20, dtype=np.float32)
        y = np.flip(np.linspace(-1, 1, 20, dtype=np.float32))
        z = []
        for xi in x:
            for yi in y:
                z.append(np.array([xi, yi]))
        z = np.stack(z)
        outputs = tf.clip_by_value(self.decode(z, trainable), 1e-8, 1 - 1e-8)
        return outputs

    def loss(self, logits, answer):
        logits = logits[0]
        if len(logits.shape) > 2:
            logits = tf.reshape(logits, [logits.shape[0], -1])
        if len(answer.shape) > 2:
            answer = tf.reshape(answer, [answer.shape[0], -1])
        with tf.name_scope('reconstruct_loss'):
            reconstruct_loss = -tf.reduce_sum(answer * tf.math.log(tf.clip_by_value(logits,1e-20,1e+20)) + (1 - answer) * tf.math.log(tf.clip_by_value(1 - logits,1e-20,1e+20)), axis=1)
        with tf.name_scope('KL_divergence'):
            KL_divergence = 0.5 * tf.reduce_sum(tf.square(self.mu) + tf.exp(self.var)**2 - 2 * self.var - 1, axis=1)
        return tf.reduce_mean(reconstruct_loss + KL_divergence)

    def accuracy(self, logits, answer, eps=1e-10):
        logits = logits[0]
        marginal_likelihood = tf.reduce_mean(answer * tf.math.log(logits + eps) + (1 - answer) * tf.math.log(1 - logits + eps),
                                            [1, 2])
        neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)
        return neg_loglikelihood

    def re_parameterization(self, mu, var):
        """
        Reparametarization trick
        parameters
        ---
        mu, var : numpy array or tensor
            mu is average, var is variance
        """
        with tf.name_scope('re_parameterization'):
            eps = tf.random.normal(shape=tf.shape(mu), mean=mu, stddev=var, dtype=tf.float32)
            return tf.cast(mu + tf.exp(0.5*var) * eps, dtype=tf.float32)    

    def gaussian(self, batch_size, n_dim, mean=0, var=1):
        z = tf.random.normal(shape=(batch_size, n_dim), mean=mean, stddev=var, dtype=tf.float32)
        mean = np.ones([batch_size, n_dim], dtype=np.float32) * mean
        var = np.ones([batch_size, n_dim], dtype=np.float32) * var
        return z, mean, var

class CVAE(VAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def inference(self, outputs, labels=None, trainable=True):
        if self.denoise:
            outputs = self.noise(outputs)
        outputs = self.combine_image(outputs, labels)
        outputs = self.encode(outputs, trainable)
        self.mu, self.var = tf.split(outputs, num_or_size_splits=2, axis=1)
        compose_img = self.re_parameterization(self.mu, self.var)
        _compose_img = self.combine_distribution(compose_img, labels)
        outputs = tf.clip_by_value(self.decode(_compose_img, trainable), 1e-8, 1 - 1e-8)
        return [outputs, compose_img]

    def test_inference(self, outputs, labels=None, trainable=False):
        return self.inference(outputs, labels, trainable)

    def predict(self, outputs, trainable=False):
        x = np.linspace(-1, 1, 20, dtype=np.float32)
        y = np.flip(np.linspace(-1, 1, 20, dtype=np.float32))
        z = []
        for xi in x:
            for yi in y:
                z.append(np.array([xi, yi]))
        z = np.stack(z)

        indices = np.array([x % self.class_dim for x in range(z.shape[0])], dtype=np.int32)
        labels = np.identity(self.class_dim, dtype=np.float32)[indices]

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

class AAE(CVAE):
    """
    Adversarial autoencoders
    # https://github.com/Mind-the-Pineapple/adversarial-autoencoder/blob/master/supervised_aae_deterministic.py
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__lambda = 0.01
        self._z_dim = kwargs['out_dim']
        Discriminator_model = Conv_Discriminator if kwargs['conv'] else Discriminator
        self.discriminator = Discriminator_model(input_shape=(self._z_dim,),
                                                 l2_reg=True if self.l2_regularizer is not None else False,
                                                 l2_reg_scale=self.l2_regularizer)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.d_optimizer = eval(kwargs['opt'])(learning_rate=kwargs['lr']/5, decay_step=None, decay_rate=0.95)
        self.g_optimizer = eval(kwargs['opt'])(learning_rate=kwargs['lr'], decay_step=None, decay_rate=0.95)

    def inference(self, outputs, labels=None, trainable=True):
        batch_size = tf.shape(outputs)[0]
        # AutoEncoder part
        if self.denoise:
            outputs = self.noise(outputs)
        encode_space = self.encode(outputs, trainable)
        compose_img = self.combine_distribution(encode_space, labels)
        outputs = self.decode(compose_img, trainable)

        # Discriminator part
        ## Fake
        fake_logit = self.discriminator(encode_space, trainable=trainable)

        ## Real
        latent_z = tf.random.normal([batch_size, self._z_dim], mean=0.0, stddev=1.0)
        real_logit = self.discriminator(latent_z, trainable=trainable)

        return [outputs, encode_space, real_logit, fake_logit]

    def test_inference(self, outputs, labels=None, trainable=False):
        encode_space = self.encode(outputs, trainable)
        compose_img = self.combine_distribution(encode_space, labels)
        outputs = self.decode(compose_img, trainable)
        return [outputs, encode_space, None, None]

    def predict(self, outputs, trainable=False):
        x = np.linspace(-1, 1, 20, dtype=np.float32)
        y = np.flip(np.linspace(-1, 1, 20, dtype=np.float32))
        z = []
        for xi in x:
            for yi in y:
                z.append(np.array([xi, yi]))
        z = np.stack(z)

        indices = np.array([x % self.class_dim for x in range(z.shape[0])], dtype=np.int32)
        labels = np.identity(self.class_dim, dtype=np.float32)[indices]

        compose_img = self.combine_distribution(z, labels)
        outputs = self.decode(compose_img, trainable)
        return outputs

    def loss(self, logits, answer):
        logits, real_logit, fake_logit = logits[0], logits[1], logits[2]
        # Auto Encoder part
        loss = tf.reduce_mean(tf.math.squared_difference(logits, answer))

        # Discriminator part
        if real_logit is not None and fake_logit is not None:
            real_loss = self.cross_entropy(tf.ones_like(real_logit), real_logit)
            fake_loss = self.cross_entropy(tf.zeros_like(fake_logit), fake_logit)
            d_loss = real_loss + fake_loss

            g_loss = self.cross_entropy(tf.ones_like(fake_logit), fake_logit)
        else:
            d_loss = None
            g_loss = None
        return [loss, d_loss, g_loss]

    def optimize(self, loss, tape=None):
        loss, d_loss, g_loss = loss[0], loss[1], loss[2]
        assert tape is not None, 'please set tape in opmize'
        # Optimize AutoEncoder
        grads = tape.gradient(loss, self.encode.trainable_variables + self.decode.trainable_variables)
        self.optimizer.method.apply_gradients(zip(grads, self.encode.trainable_variables + self.decode.trainable_variables))

        # Optimize Discriminator
        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.method.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        # Optimize Generator
        grads = tape.gradient(g_loss, self.encode.trainable_variables)
        for _ in tf.range(2):
            self.g_optimizer.method.apply_gradients(zip(grads, self.encode.trainable_variables))
        return

    def accuracy(self, logits, answer, eps=1e-10):
        logits = logits[0]
        marginal_likelihood = tf.reduce_mean(answer * tf.math.log(logits + eps) + (1 - answer) * tf.math.log(1 - logits + eps),
                                            [1, 2])
        neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)
        return neg_loglikelihood


class WAE(AutoEncoder):
    """
    # Based on https://github.com/sedelmeyer/wasserstein-auto-encoder/blob/master/Wasserstein-auto-encoder_tutorial.ipynb
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__lambda = 0.01

    def loss(self, logits, answer):
        loss = tf.reduce_mean(tf.square(logits - answer))
        return loss


def gaussian_mixture(batch_size, n_dim=2, n_labels=10, x_var=0.5, y_var=0.1, label_indices=None):
    # https://github.com/hwalsuklee/tensorflow-mnist-AAE/blob/ddca54fcc765ef821fa379623b5e5685ef853525/prior_factory.py
    if n_dim != 2:
        raise Exception("n_dim must be 2.")

    def sample(x, y, label, n_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * np.cos(r) - y * np.sin(r)
        new_y = x * np.sin(r) + y * np.cos(r)
        new_x += shift * np.cos(r)
        new_y += shift * np.sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x = np.random.normal(0, x_var, (batch_size, (int)(n_dim/2)))
    y = np.random.normal(0, y_var, (batch_size, (int)(n_dim/2)))
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    if label_indices is None:
        label_indices = np.array([x % n_labels for x in range(batch_size)], dtype=np.int32)
    for batch in range(batch_size):
        for zi in range((int)(n_dim/2)):
            z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
            
    return z

def swiss_roll(batch_size, n_dim=2, n_labels=10, label_indices=None):
    # https://github.com/hwalsuklee/tensorflow-mnist-AAE/blob/ddca54fcc765ef821fa379623b5e5685ef853525/prior_factory.py
    if n_dim != 2:
        raise Exception("n_dim must be 2.")

    def sample(label, n_labels):
        uni = np.random.uniform(0.0, 1.0) / float(n_labels) + float(label) / float(n_labels)
        r = np.sqrt(uni) * 3.0
        rad = np.pi * 4.0 * np.sqrt(uni)
        x = r * np.cos(rad)
        y = r * np.sin(rad)
        return np.array([x, y]).reshape((2,))

    z = np.zeros((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range((int)(n_dim/2)):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(np.random.randint(0, n_labels), n_labels)
    return z