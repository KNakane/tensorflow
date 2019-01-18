import tensorflow as tf

class Module(object):
    def __init__(self, l2_reg=False, l2_reg_scale=0.0001, trainable=False):
        self._l2_reg = l2_reg
        if self._l2_reg:
            self._l2_reg_scale = l2_reg_scale
        self._trainable = trainable

    def Residual(self, x, args):
        input = x
        assert len(args) == 5, '[Residual] Not enough Argument -> [kernel, filter, strides, bottleneck, No]'
        with tf.variable_scope('Residual_{}'.format(args[4])):
            if args[3]:
                x = self.BN(x=x)
                x = self.ReLU(x=x, args=None)
                x = self.conv(x=x, args=[1, args[1], 1, None])
                x = self.BN(x=x)
                x = self.ReLU(x=x, args=None)
                x = self.conv(x=x, args=[args[0], args[1], args[2], None])
                x = self.BN(x=x)
                x = self.ReLU(x=x, args=None)
                x = self.conv(x=x, args=[1, args[1],1, None])
            else:
                x = self.BN(x=x)
                x = self.ReLU(x=x, args=None)
                x = self.conv(x=x, args=[args[0], args[1], 1, None])
                x = self.BN(x=x)
                x = self.ReLU(x=x, args=None)
                x = self.conv(x=x, args=[args[0], args[1], args[2], None])

            if input.shape[1] != x.shape[1] or input.shape[3] != x.shape[3]:
                input = self.conv(x=input, args=[1, args[1], args[2], None])
            
            return x + input

    def conv(self, x, args):
        assert len(args) == 4, '[conv] Not enough Argument -> [kernel, filter, strides, activation]'
        regularizer = tf.contrib.layers.l2_regularizer(scale=self._l2_reg_scale) if self._l2_reg else None
        return tf.layers.conv2d(inputs=x,
                                filters=args[1],
                                kernel_size=[args[0], args[0]],
                                strides=[args[2], args[2]],
                                padding='same',
                                activation=args[3],
                                kernel_regularizer=regularizer,
                                trainable=self._trainable)
    
    def deconv(self, x, args):
        assert len(args) == 4, '[deconv] Not enough Argument -> [kernel, filter, strides, activation]'
        if len(x.shape) < 4:
            size = tf.sqrt(tf.cast(x.shape[1], tf.float32))
            x = tf.reshape(x, [-1, size, size, 1])
        regularizer = tf.contrib.layers.l2_regularizer(scale=self._l2_reg_scale) if self._l2_reg else None
        return tf.layers.conv2d_transpose(inputs=x,
                                          filters=args[1],
                                          kernel_size=[args[0], args[0]],
                                          strides=[args[2], args[2]],
                                          padding='same',
                                          activation=args[3],
                                          kernel_regularizer=regularizer,
                                          trainable=self._trainable)

    def reshape(self, x, args):
        return tf.reshape(tensor=x, shape=args[0])

    def max_pool(self, x, args):
        assert len(args) == 2, '[max_pool] Not enough Argument -> [pool_size, strides]'
        return tf.layers.max_pooling2d(inputs=x,
                                       pool_size=[args[0], args[0]],
                                       strides=[args[1], args[1]],
                                       padding='same')
    
    def avg_pool(self, x, args):
        assert len(args) == 2, '[avg_pool] Not enough Argument -> [pool_size, strides]'
        return tf.layers.average_pooling2d(inputs=x,
                                           pool_size=[args[0], args[0]],
                                           strides=[args[1], args[1]],
                                           padding='same')

    def gap(self, x, args): #global_average_pooling
        assert len(args) == 1, '[gap] Not enough Argument -> [output_dim]'
        with tf.variable_scope('GAP'):
            x = self.conv(x,[1, args[0], 1, None])
            for _ in range(2):
                x = tf.reduce_mean(x, axis=1)
            return x

    def BN(self, x):
        return tf.layers.batch_normalization(inputs=x, trainable=self._trainable)
    
    def ReLU(self, x, args):
        with tf.variable_scope('ReLU'):
            return tf.nn.relu(x)

    def Leaky_ReLU(self, x, args):
        with tf.variable_scope('Leaky_ReLU'):
            return tf.nn.leaky_relu(x)

    def tanh(self, x, args):
        with tf.variable_scope('tanh'):
            return tf.nn.tanh(x)

    def sigmoid(self, x, args):
        with tf.variable_scope('Sigmoid'):
            return tf.nn.sigmoid(x)

    def fc(self, x, args): # args = [units, activation=tf.nn.relu]
        assert len(args) == 2, '[FC] Not enough Argument -> [units, activation]'
        if len(x.shape) > 2:
            x = tf.layers.flatten(x, name='flatten')
        regularizer = tf.contrib.layers.l2_regularizer(scale=self._l2_reg_scale) if self._l2_reg else None
        x = tf.layers.dense(inputs=x, units=args[0], activation=args[1], kernel_regularizer=regularizer, use_bias=True)
        return x

    def dropout(self, x, args):
        assert len(args) == 3, '[Dropout] Not enough Argument -> [units, activation, rate]'
        x = self.fc(x=x, args=args[:2])
        return tf.layers.dropout(inputs=x, rate=args[2], training=self._trainable)