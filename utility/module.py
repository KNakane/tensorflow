import tensorflow as tf

class Module(object):
    def __init__(self, trainable=False):
        self._trainable = trainable
    
    def fc(self, x, args): # args = [units, activation=tf.nn.relu]
        assert len(args) == 2, '[FC] Not enough Argument'
        if len(x.shape) > 2:
            x = tf.layers.flatten(x, name='flatten')
        x = tf.layers.dense(inputs=x, units=args[0], activation=args[1], use_bias=True)
        return x

    def dropout_fc(self, x, args):
        assert len(args) == 3, '[Dropout] Not enough Argument'
        x = self.fc(x=x, args=args)
        return tf.layers.dropout(inputs=x, rate=args[2], training=self._trainable)