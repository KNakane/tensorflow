# -*- coding: utf-8 -*-
import sys
import numpy as np
import tensorflow as tf
sys.path.append('./utility')
from eager_module import EagerModule
from optimizer import *

class BasedEagerNN(EagerModule):
    def __init__(self, 
                 model=None,
                 name='NN',
                 out_dim=10,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 trainable=False,
                 is_categorical=False,
                 is_noise=False
                 ):
        super().__init__(l2_reg=l2_reg,l2_reg_scale=l2_reg_scale, trainable=trainable,is_noise=is_noise)

        self.out_dim = out_dim
        self.model = model
        self._layers = []
        self.is_categorical = is_categorical
        self.N_atoms = 51 if is_categorical else None
        if self._trainable:
            self.optimizer = eval(opt)(learning_rate=lr, decay_step=None)
        self._build()

    def _build(self):
        raise Exception('please build network')

    @tf.contrib.eager.defun
    def inference(self, x):
        raise Exception('please build network')

    def loss(self, logits, labels):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        return loss

    def optimize(self, loss, global_step, tape=None):
        assert tape is not None, 'please set tape in opmize'
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.method.apply_gradients(zip(grads, self.trainable_variables),global_step)
    

class EagerNN(BasedEagerNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self):
        for l in range(len(self.model)):
            my_layer = eval('self.' + self.model[l][0])(self.model[l][1:])
            self._layers.append(my_layer)

    #@tf.contrib.eager.defun
    def inference(self, x, softmax=True):
        x = tf.convert_to_tensor(x)
        x = tf.cast(x, tf.float32)
        for my_layer in self._layers:
            try:
                x = my_layer(x, training=self._trainable)
            except:
                x = my_layer(x)
                
        if self.is_categorical:
            x = tf.reshape(x, (-1, self.out_dim, self.N_atoms))
            if softmax:
                return tf.clip_by_value(tf.keras.activations.softmax(x, axis=2), 1e-8, 1.0-1e-8)
            else:
                return x
        else:
            return x


class Dueling_Net(BasedEagerNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self):
        for l in range(len(self.model)):
            my_layer = eval('self.' + self.model[l][0])(self.model[l][1:])
            self._layers.append(my_layer)

    @tf.contrib.eager.defun
    def inference(self, x):
        x = tf.convert_to_tensor(x)
        x = tf.cast(x, tf.float32)
        for i, my_layer in enumerate(self._layers):
            try:
                x = my_layer(x, training=self._trainable)
            except:
                x = my_layer(x)
                    
        if self.is_categorical:
            # Dueling part
            x = tf.reshape(x, (-1, self.out_dim + 1, self.N_atoms))
            V, A = tf.reshape(x[:,0], (-1, 1, self.N_atoms)), tf.reshape(x[:, 1:], [-1, self.out_dim, self.N_atoms])
            x = V + A - tf.expand_dims(tf.reduce_mean(A, axis=1), axis=1)
            return tf.clip_by_value(tf.keras.activations.softmax(x, axis=-1), 1e-8, 1.0-1e-8)
        else:
            # Dueling part
            V, A = tf.expand_dims(x[:,0], -1), x[:,1:]
            return V + A - tf.expand_dims(tf.reduce_mean(A, axis=-1), axis=-1)
            

class ActorNet(BasedEagerNN):
    def __init__(self, *args, **kwargs):
        self.max_action = kwargs.pop('max_action')
        super().__init__(*args, **kwargs)

    def _build(self):
        with tf.variable_scope(self.name):
            for l in range(len(self.model)):
                my_layer = eval('self.' + self.model[l][0])(self.model[l][1:])
                self._layers.append(my_layer)
            
    @tf.contrib.eager.defun
    def inference(self, x):
        x = tf.convert_to_tensor(x)
        x = tf.cast(x, tf.float32)
        for _, my_layer in enumerate(self._layers):
            try:
                x = my_layer(x, training=self._trainable)
            except:
                x = my_layer(x)

        if self.max_action is not None: # DDPG
            return tf.multiply(self.max_action, x)
        else:
            return x


class CriticNet(BasedEagerNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self):
        for l in range(len(self.model)):
            # categorical DQN
            if l == len(self.model) - 1 and self.is_categorical:
                self.model[l][1] = self.out_dim * self.N_atoms   # 
            my_layer = eval('self.' + self.model[l][0])(self.model[l][1:])
            self._layers.append(my_layer)

    @tf.contrib.eager.defun
    def inference(self, inputs):
        if len(inputs) == 2:
            x, u = inputs
            x = tf.concat([x, u], axis=1)
        else:
            x = inputs
            
        for i, my_layer in enumerate(self._layers):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            try:
                x = my_layer(x, training=self._trainable)
            except:
                x = my_layer(x)
        return x

class A3CNet(BasedEagerNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _build(self):
        for l in range(len(self.model)):
            my_layer = eval('self.' + self.model[l][0])(self.model[l][1:])
            self._layers.append(my_layer)

    @tf.contrib.eager.defun
    def inference(self, x):
        x = tf.convert_to_tensor(x)
        x = tf.cast(x, tf.float32)
        for i, my_layer in enumerate(self._layers):
            try:
                x = my_layer(x, training=self._trainable)
            except:
                x = my_layer(x)
        action = x[:, 1:]
        V = x[:,0]
        return action, V

    def optimize(self, grads):
        self.optimizer.method.apply_gradients(zip(grads, self.trainable_variables))

    def get_grads(self, loss, global_step, tape=None):
        assert tape is not None, 'please set tape in opmize'
        grads = tape.gradient(loss, self.trainable_variables)
        return grads
    

class A2CNet(BasedEagerNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _build(self):
        for l in range(len(self.model)):
            if l == len(self.model) - 1:
                # 状態価値V用に1unit追加
                self.model[l][1] = self.out_dim + 1
            my_layer = eval('self.' + self.model[l][0])(self.model[l][1:])
            self._layers.append(my_layer)

    @tf.contrib.eager.defun
    def inference(self, x):
        x = tf.convert_to_tensor(x)
        x = tf.cast(x, tf.float32)
        for i, my_layer in enumerate(self._layers):
            try:
                x = my_layer(x, training=self._trainable)
            except:
                x = my_layer(x)
                
        action = x[:, 1:]
        V = tf.reshape(x[:,0], (x.shape[0], 1))
        V = tf.tile(V, [1, self.out_dim])
        return action, V