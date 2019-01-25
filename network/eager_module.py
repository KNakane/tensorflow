import tensorflow as tf

class EagerModule(tf.keras.Model):
    def __init__(self, l2_reg=False, l2_reg_scale=0.0001, trainable=False):
        super().__init__()
        self._l2_reg = l2_reg
        if self._l2_reg:
            self._l2_reg_scale = l2_reg_scale
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
        x = tf.keras.layers.Dense(units=args[0], activation=args[1], kernel_regularizer=regularizer, use_bias=True)
        return x

    def dropout(self, args):
        assert len(args) == 1, '[Dropout] Not enough Argument -> [rate]'
        return tf.keras.layers.Dropout (rate=args[2])

    def noisy_dense(self, args): # 強化学習用
        assert len(args) == 2, '[noisy_dense] Not enough Argument -> [units, activation]'
        regularizer = tf.keras.regularizers.l2(self._l2_reg_scale) if self._l2_reg else None        
        x = tf.keras.layers.Dense(units=args[0], activation=args[1], kernel_regularizer=regularizer, use_bias=True)
        return x