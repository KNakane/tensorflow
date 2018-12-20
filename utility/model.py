# -*- coding: utf-8 -*-
import tensorflow as tf

class CNN_module():
    def __init__(self,
                 name="CNNMLP",
                 trainable=True,
                 reuse=None,
                 custom_getter=None,
                 initializer=None,
                 scale=None):
        self._name = name
        self._trainable = trainable
        self._reuse = reuse
        self._custom_getter = custom_getter
        self._initializer = initializer
        self._scale = scale

    def spectral_norm(self, name, w, iteration=1):
        # forked from "https://github.com/taki0112/Spectral_Normalization-Tensorflow"
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])
        with tf.variable_scope(name, reuse=False):
            u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
        u_hat = u
        v_hat = None

        def l2_norm(v, eps=1e-12):
            return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

        for i in range(iteration):
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = l2_norm(v_)
            u_ = tf.matmul(v_hat, w)
            u_hat = l2_norm(u_)
        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
        w_norm = w / sigma
        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_norm, w_shape)
        return w_norm

    def fully_connected(self, 
                        inputs,
                        units,
                        name="FullyConnected",
                        activation=tf.nn.relu,
                        enable_spectral_norm=False,
                        initializer=None,
                        bias_initializer=tf.constant_initializer([0]),
                        trainable=True):
        with tf.variable_scope(name):
            W = tf.get_variable("W", shape=[inputs.shape[-1], units], initializer=initializer, trainable=trainable)
            b = tf.get_variable("b", shape=[units], initializer=bias_initializer, trainable=trainable)
            out = tf.matmul(inputs, self.spectral_norm("sn", W) if enable_spectral_norm else W) + b
            if activation is not None:
                return activation(out)
            else:
                return out

    def conv(self, x, index, filter):
        return tf.layers.conv2d(inputs=x,
                                filters=filter[1],
                                kernel_size=[filter[0], filter[0]],
                                strides=[filter[2], filter[2]],
                                padding='same',
                                activation=None,
                                trainable=self._trainable,
                                name='conv{}'.format(index),
                                reuse=self._reuse)

    def max_pool(self, x, index, filter):
        return tf.layers.max_pooling2d(inputs=x,
                                       pool_size=[filter[0], filter[0]],
                                       strides=[filter[1], filter[1]],
                                       padding='same',
                                       name='pool{}'.format(index))
    
    def avg_pool(self, x, index, filter):
        return tf.layers.average_pooling2d(inputs=x,
                                           pool_size=[filter[0], filter[0]],
                                           strides=[filter[1], filter[1]],
                                           padding='same',
                                           name='pool{}'.format(index))

    def ResNet_block(self, x, index, filter):
        with tf.variable_scope('Residual_{}'.format(index)):
            input = x
            bottleneck = filter[3] if filter[3] is not None else False
            if bottleneck:
                x = self.BN(x=x, index='1_Res{}'.format(index))
                x = self.ReLU(x=x)
                x = self.conv(x=x, index='1_Res{}'.format(index), filter=[1, filter[1], 1])
                x = self.BN(x=x, index='1_Res{}'.format(index))
                x = self.ReLU(x=x)
                x = self.conv(x=x, index='2_Res{}'.format(index), filter=filter)
                x = self.BN(x=x, index='2_Res{}'.format(index))
                x = self.ReLU(x=x)
                x = self.conv(x=x, index='2_Res{}'.format(index), filter=[1, filter[1], 1])
            else:
                x = self.BN(x=x, index='1_Res{}'.format(index))
                x = self.ReLU(x=x)
                x = self.conv(x=x, index='1_Res{}'.format(index), filter=[filter[0], filter[1], 1])
                x = self.BN(x=x, index='2_Res{}'.format(index))
                x = self.ReLU(x=x)
                x = self.conv(x=x, index='2_Res{}'.format(index), filter=filter)

            if input.shape[1] != x.shape[1] or input.shape[3] != x.shape[3]:
                input = self.conv(x=input, index='_inputRes{}'.format(index), filter=[1, filter[1], filter[2]])
            return x + input

    def denseblock(self, input, index, filter, loop=3):
        bottleneck = filter[3] if filter[3] is not None else False
        for i in range(loop):
            if bottleneck:
                x = self.BN(x=input, index='{}_dense{}'.format(2*i+1,index))
                x = self.ReLU(x=x)
                x = self.conv(x=x, index='{}_dense{}'.format(2*i+1,index), filter=[1, filter[1], 1])
                x = self.BN(x=x, index='{}_dense{}'.format(2*(i+1),index))
                x = self.ReLU(x=x)
                x = self.conv(x=x, index='{}_dense{}'.format(2*(i+1),index), filter=[filter[0], filter[1], 1])
            else:
                x = self.BN(x=input, index='{}_dense{}'.format(i,index))
                x = self.ReLU(x=x)
                x = self.conv(x=x, index='{}_dense{}'.format(i,index), filter=[filter[0], filter[1], 1])

            input = tf.concat([x, input], 3)
        return input

    def transition(self, x, index, filter):
        x = self.BN(x=x, index='1_tran{}'.format(index))
        x = self.conv(x=x, index='1_tran{}'.format(index), filter=[1, filter[1], 1])
        x = self.avg_pool(x=x, index='1_tran{}'.format(index),filter=[2, 2])
        return x

    def DenseNet(self, x, index, filter):
        with tf.variable_scope('DenseBlock_{}'.format(index)):
            x = self.denseblock(x, index, filter)
            x = self.transition(x, index, filter)
            return x

    def global_average_pooling(self, x):
        for _ in range(2):
            x = tf.reduce_mean(x, axis=1)
        return x

    def BN(self, x, index):
        return tf.layers.batch_normalization(inputs=x,
                                             trainable=self._trainable,
                                             name='BN{}'.format(index),
                                             reuse=self._reuse)

    def ReLU(self, x):
        with tf.variable_scope('ReLU'):
            return tf.nn.relu(x)

    def fc(self, x, units, index, activation=tf.nn.relu, enable_spectral_norm=False):
        if len(x.shape) > 2:
            x = tf.layers.flatten(x, name='flatten')
        x = self.fully_connected(
            inputs=x, activation=activation, units=units, name='fc{}'.format(index),
            initializer=self._initializer, trainable=self._trainable, enable_spectral_norm=enable_spectral_norm)
        return x

    def dropout_fc(self, x, units, index, activation, rate=0.5):
        with tf.variable_scope('dropout_fc{}'.format(index)):
            x = self.fc(x=x, units=units, index=index, activation=activation)
            return tf.layers.dropout(inputs=x, rate=rate, training=self._trainable)


class CNN(CNN_module):
    def __init__(self,
                 model,
                 inputs,
                 select=None,
                 loop=0,
                 convs=[(32, 4, 4), (32, 4, 4)],  # (filters, kernel_size, strides)
                 pool=[(2, 1), (2, 1)],  # (pool_size, strides)
                 units=[32, 32],
                 output_dim=1,
                 activation=tf.nn.relu,
                 output_activation=tf.identity,
                 name="CNNMLP",
                 trainable=True,
                 reuse=None,
                 custom_getter=None,
                 initializer=None,
                 scale=None):

        assert model is not None, 'Please select CNN model'
        super().__init__(name=name, trainable=trainable, reuse=reuse, custom_getter=custom_getter, initializer=initializer, scale=scale)
        self._model = model
        self._inputs = inputs
        self._select = select
        self._loop = loop
        self._convs = convs
        self._pool = pool
        self._units = units
        self._output_dim = output_dim
        self._activation = activation
        self._output_activation = output_activation

    def __call__(self, inputs):
        return eval('self.' + self._model)(inputs)

    def variableCNN(self, x):
        with tf.variable_scope(self._name):
            for i in range(len(self._select)):
                if self._select[i][0] == 'conv':
                    x = self.conv(x=x, index=i, filter=self._select[i][1:])
                elif self._select[i][0] == 'ResNet':
                    x = self.ResNet_block(x=x, index=i, filter=self._select[i][1:])
                elif self._select[i][0] == 'DenseNet':
                    x = self.DenseNet(x=x, index=i, filter=self._select[i][1:])
                elif self._select[i][0] == 'max_pool':
                    x = self.max_pool(x=x, index=i, filter=self._select[i][1:])
                elif self._select[i][0] == 'BN':
                    x = self.BN(x=x, index=i)
                elif self._select[i][0] == 'GAP':
                    with tf.name_scope('GAP'):
                        x = self.global_average_pooling(x=x)
                elif self._select[i][0] == 'ReLU':
                    x = self.ReLU(x=x)
                elif self._select[i][0] == 'fc':
                    x = self.fc(x=x, index=i, units=self._select[i][1], activation=self._activation)
                elif self._select[i][0] == 'dropout':
                    x = self.dropout_fc(x=x, index=i, units=self._select[i][1], activation=self._activation, rate=self._select[i][2])
                else:
                    raise Exception('Error! Incorrect select function -> {}'.format(self._select[i][0]))
            logits = self.fc(x=x, index=i+1, units=self._output_dim, activation=None)

        return logits


    def sample_CNN(self):
        x = self._inputs
        with tf.variable_scope(self._name):
            # network weights
            W_conv1 = self.weight_variable([8,8,4,32])
            b_conv1 = self.bias_variable([32])

            W_conv2 = self.weight_variable([4,4,32,64])
            b_conv2 = self.bias_variable([64])

            W_conv3 = self.weight_variable([3,3,64,64])
            b_conv3 = self.bias_variable([64])

            W_fc1 = self.weight_variable([3136,512])
            b_fc1 = self.bias_variable([512])

            W_fc2 = self.weight_variable([512,self._output_dim])
            b_fc2 = self.bias_variable([self._output_dim])

            # input layer

            #stateInput = tf.placeholder("float",[None,84,84,4])

            # hidden layers
            h_conv1 = tf.nn.relu(self.conv2d(x,W_conv1,4) + b_conv1)
            #h_pool1 = self.max_pool_2x2(h_conv1)

            h_conv2 = tf.nn.relu(self.conv2d(h_conv1,W_conv2,2) + b_conv2)

            h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3,1) + b_conv3)
            h_conv3_flat = tf.reshape(h_conv3,[-1,3136])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)

            # Q Value layer
            logits = tf.matmul(h_fc1,W_fc2) + b_fc2

        return logits

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)
        
    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)
        
    def conv2d(self,x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")
        
    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")