import tensorflow as tf

class Module(object):
    def __init__(self, trainable=False):
        self.trainable = trainable
    
    def fc(self, args): # args = [units, activation=tf.nn.relu]
        assert len(args) == 2, '[FC] Not enough Argument'
        x = tf.layers.Dense(units=args[0], activation=args[1], use_bias=True)
        return x