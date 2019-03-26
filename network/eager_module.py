import numpy as np
import tensorflow as tf


class EagerModule(tf.keras.Model):
    def __init__(self, l2_reg=False, l2_reg_scale=0.0001, is_noise=False, trainable=False):
        super().__init__()
        self._l2_reg = l2_reg
        if self._l2_reg:
            self._l2_reg_scale = l2_reg_scale
        self.is_noise = is_noise
        self._trainable = trainable

    def conv(self, args):
        assert len(args) == 4, '[conv] Not enough Argument -> [kernel, filter, strides, activation]'
        regularizer = tf.keras.regularizers.l2(self._l2_reg_scale) if self._l2_reg else None
        return tf.keras.layers.Conv2D(filters=args[1],
                                      kernel_size=[args[0], args[0]],
                                      strides=[args[2], args[2]],
                                      padding='same',
                                      activation=args[3],
                                      kernel_regularizer=regularizer)

    def max_pool(self, args):
        assert len(args) == 2, '[max_pool] Not enough Argument -> [pool_size, strides]'
        return tf.keras.layers.MaxPool2D(pool_size=[args[0],args[0]],
                                         strides=[args[1], args[1]],
                                         padding='same')
    
    def avg_pool(self,args):
        assert len(args) == 2, '[avg_pool] Not enough Argument -> [pool_size, strides]'
        return tf.keras.layers.AveragePooling2D(pool_size=[args[0],args[0]],
                                                strides=[args[1], args[1]],
                                                padding='same')

    def ReLU(self, args):
        return tf.keras.layers.ReLU()

    def Leaky_ReLU(self, args):
        return tf.keras.layers.LeakyReLU()

    def flat(self, args):
        return tf.keras.layers.Flatten()

    def fc(self, args): # args = [units, activation=tf.nn.relu]
        assert len(args) == 2, '[FC] Not enough Argument -> [units, activation]'
        regularizer = tf.keras.regularizers.l2(self._l2_reg_scale) if self._l2_reg else None
        if self.is_noise:  # noisy net
            x = tf.keras.layers.Dense(units=args[0], activation=args[1], kernel_initializer=self.noisy_weight, bias_initializer=self.noisy_bias, kernel_regularizer=regularizer, use_bias=True)
        else:
            x = tf.keras.layers.Dense(units=args[0], activation=args[1], kernel_regularizer=regularizer, use_bias=True)
        return x

    def dropout(self, args):
        assert len(args) == 1, '[Dropout] Not enough Argument -> [rate]'
        return tf.keras.layers.Dropout (rate=args[2])

    def noisy_weight(self, shape, dtype=None, partition_info=None):
        fan_in, fan_out = shape[0], shape[1]
        # based on https://github.com/wenh123/NoisyNet-DQN/blob/master/tf_util.py
        def f(x):
            return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))
        mu_init = tf.random_uniform(shape=[fan_in, fan_out], minval=-1*1/np.power(fan_in, 0.5),     
                                                            maxval=1*1/np.power(fan_in, 0.5))
        sigma_init = tf.constant(0.4/np.power(fan_in, 0.5),dtype=tf.float32, shape=[fan_in, fan_out])
        # Sample noise from gaussian
        p = tf.random_normal([fan_in, 1])
        q = tf.random_normal([1, fan_out])
        f_p = f(p); f_q = f(q)
        w_epsilon = f_p*f_q

        # w = w_mu + w_sigma*w_epsilon
        w = mu_init + tf.multiply(sigma_init, w_epsilon)
        return w

    def noisy_bias(self, shape, dtype=None, partition_info=None):
        fan_out = shape[0]
        def f(x):
            return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))
        mu_init = tf.random_uniform(shape=[fan_out], minval=-1*1/np.power(fan_out, 0.5),     
                                                            maxval=1*1/np.power(fan_out, 0.5))
        sigma_init = tf.constant(0.4/np.power(fan_out, 0.5),dtype=tf.float32, shape=[fan_out])
        q = tf.random_normal([1, fan_out])
        f_q = f(q)
        b_epsilon = tf.squeeze(f_q)
        b = mu_init + tf.multiply(sigma_init, b_epsilon)

        return b