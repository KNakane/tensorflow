import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from optimizer.optimizer import *
from network.activation_func import gelu


# https://github.com/emla2805/vision-transformer/blob/master/model.py

class VisionTransformer(Model):
    def __init__(self, 
                 model=None,
                 name='Model',
                 input_shape=None,
                 out_dim=10,
                 opt="Adam",   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 patch_size=4,
                 d_model=64,
                 num_layers=4,
                 num_heads=4,
                 mlp_dim=128,
                 dropout=0.1
                 ):
        super().__init__()
        self.model_name = name
        self.out_dim = out_dim
        self.optimizer = eval(opt)(learning_rate=lr, decay_step=None, decay_rate=0.95)
        self.l2_regularizer = tf.keras.regularizers.l2(l2_reg_scale) if l2_reg else None

        self.d_model = d_model
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers

        num_patches = (input_shape[0] // patch_size) ** 2
        self.patch_dim = input_shape[2] * patch_size ** 2

        self.pos_emb = self.add_weight(
            "pos_emb", shape=(1, num_patches + 1, d_model)
        )
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, d_model))

        self._build()
        self.loss_function = tf.losses.CategoricalCrossentropy()
        self.accuracy_function = tf.keras.metrics.CategoricalAccuracy()
        with tf.device("/cpu:0"):
            self(x=tf.constant(tf.zeros(shape=(1,)+input_shape,
                                             dtype=tf.float32)))

    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def _build(self):
        self.embedded_patche = tf.keras.layers.Dense(self.d_model)

        self.enc_layers = [
            TransformerBlock(self.d_model, self.num_heads, self.mlp_dim, self.dropout)
            for _ in range(self.num_layers)
        ]

        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense1 = tf.keras.layers.Dense(self.mlp_dim, activation=gelu)
        self.dropout1 = tf.keras.layers.Dropout(self.dropout)
        self.dense2 = tf.keras.layers.Dense(self.out_dim, activation='softmax')


    def __call__(self, x, trainable=True):
        batch_size = tf.shape(x)[0]
        patches = self.extract_patches(x)
        x = self.embedded_patche(patches)

        class_emb = tf.broadcast_to(
            self.class_emb, [batch_size, 1, self.d_model]
        )

        x = tf.concat([class_emb, x], axis=1)
        x = x + self.pos_emb

        for layer in self.enc_layers:
            x = layer(x, trainable)

        x = self.ln1(x[:, 0])
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)

        return x

    def loss(self, logits, answer):
        return self.loss_function(y_true=answer, y_pred=logits)

    def optimize(self, loss, tape=None):
        assert tape is not None, 'please set tape in opmize'
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.method.apply_gradients(zip(grads, self.trainable_variables))
        return

    def accuracy(self, logits, answer):
        self.accuracy_function(y_true=answer, y_pred=logits)
        return self.accuracy_function.result()


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = tf.keras.layers.Dense(embed_dim)
        self.key_dense = tf.keras.layers.Dense(embed_dim)
        self.value_dense = tf.keras.layers.Dense(embed_dim)
        self.combine_heads = tf.keras.layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(mlp_dim, activation=gelu),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(embed_dim),
                tf.keras.layers.Dropout(dropout),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.att(inputs_norm)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = attn_output + inputs

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        mlp_output = self.dropout2(mlp_output, training=training)
        return mlp_output + out1