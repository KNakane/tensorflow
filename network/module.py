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
                x = self.BN(x=x, args=None)
                x = self.ReLU(x=x, args=None)
                x = self.conv(x=x, args=[1, args[1], 1, None])
                x = self.BN(x=x, args=None)
                x = self.ReLU(x=x, args=None)
                x = self.conv(x=x, args=[args[0], args[1], args[2], None])
                x = self.BN(x=x, args=None)
                x = self.ReLU(x=x, args=None)
                x = self.conv(x=x, args=[1, args[1],1, None])
            else:
                x = self.BN(x=x, args=None)
                x = self.ReLU(x=x, args=None)
                x = self.conv(x=x, args=[args[0], args[1], 1, None])
                x = self.BN(x=x, args=None)
                x = self.ReLU(x=x, args=None)
                x = self.conv(x=x, args=[args[0], args[1], args[2], None])

            if input.shape[1] != x.shape[1] or input.shape[3] != x.shape[3]:
                input = self.conv(x=input, args=[1, args[1], args[2], None])
            
            return x + input

    def conv(self, x, args, name=None):
        """
        convolutionを行う
        parameters
        ----------
        x : tensor
            input image 4D
        args : list
            [kernel, filter, strides, activation]
        name : str
            layer name
        
        returns
        ----------
        feature map : tensor
            畳み込みした特徴マップ

        Else
        ----------
        padding = same
        """
        assert len(args) == 4, '[conv] Not enough Argument -> [kernel, filter, strides, activation]'
        regularizer = tf.contrib.layers.l2_regularizer(scale=self._l2_reg_scale) if self._l2_reg else None
        return tf.layers.conv2d(inputs=x,
                                filters=args[1],
                                kernel_size=[args[0], args[0]],
                                strides=[args[2], args[2]],
                                padding='same',
                                activation=args[3],
                                kernel_regularizer=regularizer,
                                trainable=self._trainable,
                                name=name)
    
    def deconv(self, x, args, name=None):
        """
        de-convolutionを行う
        parameters
        ----------
        x : tensor
            input image 4D
        args : list
            [kernel, filter, strides, activation]
        name : str
            layer name
        
        returns
        ----------
        feature map : tensor
            畳み込みした特徴マップ

        Else
        ----------
        padding = same
        """
        assert len(args) == 4, '[deconv] Not enough Argument -> [kernel, filter, strides, activation]'
        assert len(x.shape) == 4
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
        """
        Reshapeを行う
        parameters
        ----------
        x : tensor
            input image 4D
        args : list
            [reshapeしたいサイズ]
            ex) [-1, 28, 28, 1]
        
        returns
        ----------
        reshape : tensor
            reshapeしたtensor
        """
        return tf.reshape(tensor=x, shape=args[0])

    def max_pool(self, x, args):
        """
        Max poolingを行う
        parameters
        ----------
        x : tensor
            input image 4D
        args : list
            [pool_size, strides, padding]
        
        returns
        ----------
        feature map : tensor
            Poolingした特徴マップ
        """
        assert len(args) == 3, '[max_pool] Not enough Argument -> [pool_size, strides, padding]'
        return tf.layers.max_pooling2d(inputs=x,
                                       pool_size=[args[0], args[0]],
                                       strides=[args[1], args[1]],
                                       padding=args[2])
    
    def avg_pool(self, x, args):
        """
        Average poolingを行う
        parameters
        ----------
        x : tensor
            input image 4D
        args : list
            [pool_size, strides, padding]
        
        returns
        ----------
        feature map : tensor
            Poolingした特徴マップ
        """
        assert len(args) == 3, '[avg_pool] Not enough Argument -> [pool_size, strides, padding]'
        return tf.layers.average_pooling2d(inputs=x,
                                           pool_size=[args[0], args[0]],
                                           strides=[args[1], args[1]],
                                           padding=args[2])

    def gap(self, x, args): #global_average_pooling
        """
        Global Average poolingを行う
        parameters
        ----------
        x : tensor
            input image 4D
        args : list
            [output_dimension]
        
        returns
        ----------
        x : tensor
            GAPしたtensor
        """
        assert len(args) == 1, '[gap] Not enough Argument -> [output_dim]'
        with tf.variable_scope('GAP'):
            x = self.conv(x,[1, args[0], 1, None])
            for _ in range(2):
                x = tf.reduce_mean(x, axis=1)
            return x

    def BN(self, x, args):
        """
        Batch Normalizationを行う
        parameters
        ----------
        x : tensor
            input image 4D
        args : list
            [None]で良い
        
        returns
        ----------
        x : tensor
            BNしたtensor
        tf.get_collection(tf.GraphKeys.UPDATE_OPS)が必須
        """
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
        """
        Fully connect
        parameters
        ----------
        x : tensor
            input image 4D
        args : list
            [units, activation]
        
        returns
        ----------
        feature map : tensor
            全結合の特徴ベクトル
        """
        assert len(args) == 2, '[FC] Not enough Argument -> [units, activation]'
        if len(x.shape) > 2:
            x = tf.layers.flatten(x, name='flatten')
        regularizer = tf.contrib.layers.l2_regularizer(scale=self._l2_reg_scale) if self._l2_reg else None
        x = tf.layers.dense(inputs=x, units=args[0], activation=args[1], kernel_regularizer=regularizer, use_bias=True)
        return x

    def dropout(self, x, args):
        """
        Fully connect + Dropout
        parameters
        ----------
        x : tensor
            input image 4D
        args : list
            [units, activation, rate]
        
        returns
        ----------
        feature map : tensor
            全結合の特徴ベクトル
        """
        assert len(args) == 3, '[Dropout] Not enough Argument -> [units, activation, rate]'
        x = self.fc(x=x, args=args[:2])
        return tf.layers.dropout(inputs=x, rate=args[2], training=self._trainable)