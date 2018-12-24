import tensorflow as tf

class Module(object):
    def __init__(self, trainable=False):
        self._trainable = trainable

    def Residual(self, x, args):
        input = x
        assert len(args) == 4, '[Residual] Not enough Argument -> [kernel, filter, strides, bottleneck]'
        with tf.variable_scope('Residual'):
            if args[3]:
                x = self.BN(x=x, args=None)
                x = self.ReLU(x=x, args=None)
                x = self.conv(x=x, args=[1, args[1],1])
                x = self.BN(x=x, args=None)
                x = self.ReLU(x=x, args=None)
                x = self.conv(x=x, args=args[:3])
                x = self.BN(x=x, args=None)
                x = self.ReLU(x=x, args=None)
                x = self.conv(x=x, args=[1, args[1],1])
            else:
                x = self.BN(x=x, args=None)
                x = self.ReLU(x=x, args=None)
                x = self.conv(x=x, args=[args[0], args[1],1])
                x = self.BN(x=x, args=None)
                x = self.ReLU(x=x, args=None)
                x = self.conv(x=x, args=args[:3])

            if input.shape[1] != x.shape[1] or input.shape[3] != x.shape[3]:
                input = self.conv(x=input, args=[1, args[1], args[2]])
            
            return x + input

    def conv(self, x, args):
        assert len(args) == 3, '[conv] Not enough Argument -> [kernel, filter, strides]'
        return tf.layers.conv2d(inputs=x,
                                filters=args[1],
                                kernel_size=[args[0], args[0]],
                                strides=[args[2], args[2]],
                                padding='same',
                                activation=None,
                                trainable=self._trainable)

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
        with tf.variable_scope('GAP'):
            for _ in range(2):
                x = tf.reduce_mean(x, axis=1)
            return x

    def BN(self, x, args): #Batch Normalization
        with tf.variable_scope('BatchNorm'):
            return tf.layers.batch_normalization(inputs=x,
                                                 trainable=self._trainable)
    
    def ReLU(self, x, args):
        with tf.variable_scope('ReLU'):
            return tf.nn.relu(x)

    def fc(self, x, args): # args = [units, activation=tf.nn.relu]
        assert len(args) == 2, '[FC] Not enough Argument -> [units, activation]'
        if len(x.shape) > 2:
            x = tf.layers.flatten(x, name='flatten')
        x = tf.layers.dense(inputs=x, units=args[0], activation=args[1], use_bias=True)
        return x

    def dropout(self, x, args):
        assert len(args) == 3, '[Dropout] Not enough Argument -> [units, activation, rate]'
        x = self.fc(x=x, args=args[:2])
        return tf.layers.dropout(inputs=x, rate=args[2], training=self._trainable)