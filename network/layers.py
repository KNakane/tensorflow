import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.keras.layers import Wrapper

class SpectralNormalization(Wrapper):
    def __init__(self, layer, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.w = self.layer.kernel
            self.u = tf.Variable(
            tf.random.normal((tuple([1, self.layer.kernel.shape.as_list()[-1]])), dtype=tf.float32), 
            aggregation=tf.VariableAggregation.MEAN, trainable=False)
        super(SpectralNormalization, self).build()

    def call(self, inputs, training=False):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        w_shape = self.w.shape.as_list()
        w_reshaped = K.reshape(self.w, [-1, w_shape[-1]])
        _u, _v = power_iteration(w_reshaped, self.u)
        sigma = K.dot(_v, w_reshaped)
        sigma = K.dot(sigma, K.transpose(_u))
        w_bar = w_reshaped / sigma
        if training == False:
            w_bar = K.reshape(w_bar, w_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                 w_bar = K.reshape(w_bar, w_shape) 
        output = self.layer(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())