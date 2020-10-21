import tensorflow as tf
from CNN.model import MyModel
from optimizer.optimizer import *

class DenseNet(MyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.growth_k = 12
        self.nb_blocks = 2
        self.dense_blocks = []
        self.transition_blocks = []

    def _build(self):
        self.conv = tf.keras.layers.Conv2D(self.growth_k * 2, (7,7), (2,2), 'same', kernel_regularizer=self.l2_regularizer)
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='valid')
        for i in range(self.nb_blocks):
            self.dense_blocks.append(DenseBlock(n_layers=4, growth_k=self.growth_k, bottle_neck=True, name='dense_'+str(i)))
            self.transition_blocks.append(TransitionLayer(self.growth_k))
        self.dense = DenseBlock(n_layers=31, growth_k=self.growth_k, bottle_neck=True, name='dense_final')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.out = tf.keras.layers.Dense(self.out_dim, activation='softmax')
        return

    @tf.function
    def __call__(self, x, trainable=True):
        with tf.name_scope(self.name):
            x = self.conv(x, training=trainable)
            x = self.maxpool(x, training=trainable)
            for i in tf.range(self.nb_blocks):
                x = self.dense_blocks[i](x, training=trainable)
                x = self.transition_blocks[i](x, training=trainable)
            x = self.dense(x, training=trainable)
            x = self.bn(x, training=trainable)
            x = self.relu(x, training=trainable)
            x = self.gap(x, training=trainable)
            x = self.out(x, training=trainable)
            return x
    


class DenseBlock(tf.keras.Model):
    def __init__(self,
                 n_layers,
                 growth_k,
                 bottle_neck,
                 name):
        super().__init__()
        self.__n_layers = n_layers
        self.__growth_k = growth_k
        self.__bottle_neck = bottle_neck
        self.__name = name
        self.dense_layers = []
        self.layers_concat = []

    def _build(self):
        for _ in range(self.__n_layers):
            self.dense_layers.append(DenseLayer(self.__growth_k, self.__bottle_neck))
        return      
            
    @tf.function
    def __call__(self, x, trainable=True):
        with tf.name_scope(self.name):
            self.layers_concat.append(x)
            inputs = x.copy()
            for layers in self.dense_layers:
                inputs = layers(inputs, training=trainable)

                self.layers_concat.append(inputs)
                x = tf.concat(self.layers_concat, axis=3)
            return x


class DenseLayer(tf.keras.Model):
    def __init__(self,
                 growth_k,
                 bottle_neck):
        super().__init__()
        self.growth_k = growth_k
        self.__bottle_neck = bottle_neck

    def _build(self):
        if self.__bottle_neck:
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.relu1 = tf.keras.layers.ReLU()
            self.conv1 = tf.keras.layers.Conv2D(filters=self.growth_k*4, kernel_size=(1,1), strides=(1, 1), padding='same')
            self.dropout1 = tf.keras.layers.Dropout(rate=0.2)
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.relu2 = tf.keras.layers.ReLU()
            self.conv2 = tf.keras.layers.Conv2D(filters=self.growth_k, kernel_size=(3,3), strides=(1, 1), padding='same')
            self.dropout2 = tf.keras.layers.Dropout(rate=0.2)
        else:
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.relu1 = tf.keras.layers.ReLU()
            self.conv1 = tf.keras.layers.Conv2D(filters=self.growth_k*4, kernel_size=(3,3), strides=(1, 1), padding='same')
            self.dropout1 = tf.keras.layers.Dropout(rate=0.2)
        return

    @tf.function
    def __call__(self, x, trainable=True):
        with tf.name_scope(self.name):
            x = self.bn1(x, training=trainable)
            x = self.relu1(x, training=trainable)
            x = self.conv1(x, training=trainable)
            x = self.dropout1(x, training=trainable)
            if self.__bottle_neck:
                x = self.bn2(x, training=trainable)
                x = self.relu2(x, training=trainable)
                x = self.conv2(x, training=trainable)
                x = self.dropout2(x, training=trainable)

            return x

class TransitionLayer(tf.keras.Model):
    def __init__(self, filters):
        super().__init__()
        self.__filters = filters

    def _build(self):
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(filters=self.__filters, kernel_size=(1,1), strides=(1, 1), padding='same')
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))
        return

    @tf.function
    def __call__(self, x, trainable=True):
        x = self.bn(x, training=trainable)
        x = self.relu(x, training=trainable)
        x = self.conv(x, training=trainable)
        x = self.dropout(x, training=trainable)
        x = self.avg_pool(x, training=trainable)
        return x