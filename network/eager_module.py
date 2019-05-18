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
            return NoisyDense(units=args[0], activation=args[1], kernel_regularizer=regularizer, use_bias=True, trainable=self._trainable)
        else:
            return tf.keras.layers.Dense(units=args[0], activation=args[1], kernel_regularizer=regularizer, use_bias=True)

    def dropout(self, args):
        assert len(args) == 1, '[Dropout] Not enough Argument -> [rate]'
        return tf.keras.layers.Dropout (rate=args[2])

class NoisyDense(tf.keras.layers.Layer):
    # Based on https://github.com/OctThe16th/Noisy-A3C-Keras/blob/master/NoisyDense.py
    def __init__(self, 
                 units,
                 sigma_init=0.02,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=False,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.units = int(units)
        self.sigma_init = sigma_init
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.trainable = trainable

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=[self.input_dim, self.units],
                                      initializer=tf.initializers.orthogonal(dtype=tf.float32),
                                      name='kernel',
                                      dtype=tf.float32,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=self.trainable)
        
        self.sigma_kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=tf.keras.initializers.Constant(value=self.sigma_init),
                                      name='sigma_kernel',
                                      trainable=self.trainable
                                      )
        
        self.epsilon_kernel = tf.keras.backend.zeros(shape=(self.input_dim, self.units), name='epsilon_kernel')
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=self.trainable)
            
            self.sigma_bias = self.add_weight(shape=(self.units,),
                                        initializer=tf.keras.initializers.Constant(value=self.sigma_init),
                                        name='sigma_bias',
                                        trainable=self.trainable)
            
            self.epsilon_bias = tf.keras.backend.zeros(shape=(self.units,), name='epsilon_bias')

        else:
            self.bias = None
            self.epsilon_bias = None

        if self.trainable:
            self.sample_noise()
        else:
            self.remove_noise()

        super().build(input_shape)

    def call(self, input):
        perturbation = self.sigma_kernel * self.epsilon_kernel
        perturbed_kernel = self.kernel + perturbation
        output = tf.keras.backend.dot(input, perturbed_kernel)
        if self.use_bias:
            bias_perturbation = self.sigma_bias * self.epsilon_bias
            perturbed_bias = self.bias + bias_perturbation
            output = tf.keras.backend.bias_add(output, perturbed_bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def sample_noise(self):
        self.epsilon_kernel.assign(np.random.normal(0, 1, (self.input_dim, self.units)))
        if self.use_bias:
            self.epsilon_bias.assign(np.random.normal(0, 1, (self.units,)))

    def remove_noise(self):
        self.epsilon_kernel.assign(np.zeros(shape=(self.input_dim, self.units)))
        if self.use_bias:
            self.epsilon_bias.assign(self.epsilon_bias, np.zeros(shape=self.units,))