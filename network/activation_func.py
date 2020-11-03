"""Tensorflow-Keras Implementation of Mish"""
# Fork https://github.com/digantamisra98/Mish
#Mish: Self Regularized Non-Monotonic Activation Function

## Import Necessary Modules
import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects
from tensorflow.python.ops import math_ops

class Mish(Activation):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

get_custom_objects().update({'Mish': Mish(mish)})


class GELU(Activation):
    """Applies the Gaussian error linear unit (GELU) activation function.
    Gaussian error linear unit (GELU) computes
    `x * P(X <= x)`, where `P(X) ~ N(0, 1)`.
    The (GELU) nonlinearity weights inputs by their value, rather than gates
    inputs by their sign as in ReLU.
    For example:
    >>> x = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=tf.float32)
    >>> y = tf.keras.activations.gelu(x)
    >>> y.numpy()
    array([-0.00404951, -0.15865529,  0.        ,  0.8413447 ,  2.9959507 ],
        dtype=float32)
    >>> y = tf.keras.activations.gelu(x, approximate=True)
    >>> y.numpy()
    array([-0.00363752, -0.15880796,  0.        ,  0.841192  ,  2.9963627 ],
        dtype=float32)
    Arguments:
        x: Input tensor.
        approximate: A `bool`, whether to enable approximation.
    Returns:
        The gaussian error linear activation:
        `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`
        if `approximate` is `True` or
        `x * P(X <= x) = 0.5 * x * (1 + erf(x / sqrt(2)))`,
        where `P(X) ~ N(0, 1)`,
        if `approximate` is `False`.
    Reference:
        - [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)
    """
    def __init__(self, activation, **kwargs):
        super(GELU, self).__init__(activation, **kwargs)
        self.__name__ = 'gelu'

def gelu(x, approximate=False):
    if approximate:
        coeff = math_ops.cast(0.044715, x.dtype)
        return 0.5 * x * (
            1.0 + math_ops.tanh(0.7978845608028654 *
                                (x + coeff * math_ops.pow(x, 3))))
    else:
        return 0.5 * x * (1.0 + math_ops.erf(
            x / math_ops.cast(1.4142135623730951, x.dtype)))

get_custom_objects().update({'gelu': GELU(gelu)})