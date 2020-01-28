import tensorflow as tf
from tensorflow.keras.models import Model

class Encoder(Model):
    def __init__(self, 
                 model=None,
                 name='Encoder',
                 out_dim=10,
                 l2_reg=False,
                 l2_reg_scale=0.0001
                 ):
        super().__init__()
        self.model_name = name
        self.out_dim = out_dim
        self.l2_regularizer = l2_reg_scale if l2_reg else None
        self._build()

    def _build(self):
        self.flat = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.fc2 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.fc3 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.out = tf.keras.layers.Dense(self.out_dim, activation='relu', kernel_regularizer=self.l2_regularizer)
        return

    def __call__(self, x, trainable=True):
        x = self.flat(x, training=trainable)
        x = self.fc1(x, training=trainable)
        x = self.fc2(x, training=trainable)
        x = self.fc3(x, training=trainable)
        x = self.out(x, training=trainable)
        return x


class Conv_Encoder(Encoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self):
        self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')
        self.max_pool1 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same')
        self.max_pool2 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same')
        self.max_pool3 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        return

    def __call__(self, x, trainable=True):
        x = self.conv1(x, training=trainable)
        x = self.max_pool1(x, training=trainable)
        x = self.conv2(x, training=trainable)
        x = self.max_pool2(x, training=trainable)
        x = self.conv3(x, training=trainable)
        x = self.max_pool3(x, training=trainable)
        return x



class Decoder(Model):
    def __init__(self, 
                 model=None,
                 name='Encoder',
                 size=28,
                 channel=1,
                 l2_reg=False,
                 l2_reg_scale=0.0001
                 ):
        super().__init__()
        self.model_name = name
        self.size = size
        self.channel = channel
        self.l2_regularizer = l2_reg_scale if l2_reg else None
        self._build()

    def _build(self):
        self.fc1 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.fc2 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.fc3 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.fc4 = tf.keras.layers.Dense(self.size**2 * self.channel, activation='sigmoid', kernel_regularizer=self.l2_regularizer)
        self.out = tf.keras.layers.Reshape((self.size,self.size,self.channel))
        return

    def __call__(self, x, trainable=True):
        x = self.fc1(x, training=trainable)
        x = self.fc2(x, training=trainable)
        x = self.fc3(x, training=trainable)
        x = self.fc4(x, training=trainable)
        x = self.out(x, training=trainable)
        return x


class Conv_Decoder(Decoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self):
        self.conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same')
        self.upsample1 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same')
        self.upsample2 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.conv3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='valid')
        self.upsample3 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.conv4 = tf.keras.layers.Conv2D(filters=self.channel, kernel_size=(3,3), activation='sigmoid', padding='same')
        return

    def __call__(self, x, trainable=True):
        x = self.conv1(x, training=trainable)
        x = self.upsample1(x, training=trainable)
        x = self.conv2(x, training=trainable)
        x = self.upsample2(x, training=trainable)
        x = self.conv3(x, training=trainable)
        x = self.upsample3(x, training=trainable)
        x = self.conv4(x, training=trainable)
        return x