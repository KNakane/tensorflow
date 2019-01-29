# -*- coding: utf-8 -*-
import sys
import numpy as np
import tensorflow as tf
sys.path.append('./utility')
from eager_module import EagerModule
from optimizer import *

class EagerCNN(EagerModule):
    def __init__(self, 
                 model=None,
                 name='CNN',
                 out_dim=10,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 trainable=False,
                 is_noisy=False,
                 is_categorical=False
                 ):
        super().__init__(l2_reg=l2_reg,l2_reg_scale=l2_reg_scale, trainable=trainable)
        self.out_dim = out_dim
        self.model = model
        self._layers = []
        self.is_noisy = is_noisy
        self.is_categorical = is_categorical
        self.N_atoms = 51 if is_categorical else None
        if self._trainable:
            self.optimizer = eval(opt)(learning_rate=lr)
        self._build()

    def _build(self):
        for l in range(len(self.model)):
            # categorical DQN
            if l == len(self.model) - 1 and self.is_categorical:
                self.model[l][1] = self.out_dim * self.N_atoms   # 
            if self.is_noisy and self.model[l][0]=='fc':
                my_layer = self.noisy_dense(self.model[l][1:]) #noisy_net
            else:
                my_layer = eval('self.' + self.model[l][0])(self.model[l][1:])
            self._layers.append(my_layer)

    def inference(self, x):
        for my_layer in self._layers:
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            try:
                x = my_layer(x, training=self._trainable)
            except:
                x = my_layer(x)
                
        if self.is_categorical:
            x = tf.reshape(x, (x.shape[0], self.out_dim + 1, self.N_atoms))
            return tf.keras.activations.softmax(x, axis=2)
        else:
            return x

    def loss(self, logits, labels):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        return loss

    def optimize(self, loss, global_step, tape=None):
        assert tape is not None, 'please set tape in opmize'
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.method.apply_gradients(zip(grads, self.trainable_variables),global_step)


class Dueling_Net(EagerCNN):
    def __init__(self, 
                 model=None,
                 name='Dueling_Net',
                 out_dim=10,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 trainable=False,
                 is_noisy=False,
                 is_categorical=False
                 ):
        super().__init__(model=model,name=name,out_dim=out_dim,opt=opt,lr=lr,l2_reg=l2_reg,l2_reg_scale=l2_reg_scale,trainable=trainable,is_noisy=is_noisy,is_categorical=is_categorical)

    def _build(self):
        for l in range(len(self.model)):
            if l == len(self.model) - 1:
                # 状態価値V用に1unit追加するが、categoricalの場合も考慮
                self.model[l][1] = (self.out_dim + 1) * self.N_atoms if self.is_categorical else self.out_dim + 1
            if self.is_noisy and self.model[l][0]=='fc':
                my_layer = self.noisy_dense(self.model[l][1:]) #noisy_net
            else:
                my_layer = eval('self.' + self.model[l][0])(self.model[l][1:])
            self._layers.append(my_layer)

    def inference(self, x):
        for i, my_layer in enumerate(self._layers):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            try:
                x = my_layer(x, training=self._trainable)
            except:
                x = my_layer(x)
                    
        if self.is_categorical:
            # Dueling part
            x = tf.reshape(x, (x.shape[0], self.out_dim + 1, self.N_atoms))
            V = tf.reshape(x[:,0], (x.shape[0], 1, self.N_atoms))
            V = tf.tile(V, [1, self.out_dim, 1])
            x = x[:, 1:] + V - tf.tile(tf.reshape(np.average(x[:,1:], axis=1), (x.shape[0], 1, self.N_atoms)), [1, self.out_dim, 1])
            return tf.keras.activations.softmax(x, axis=2)
        else:
            # Dueling part
            V = tf.reshape(x[:,0], (x.shape[0], 1))
            V = tf.tile(V, [1, self.out_dim])
            return x[:, 1:] + V - tf.tile(tf.reshape(np.average(x[:,1:], axis=1), (x.shape[0], 1)), [1, self.out_dim])

class ActorNet(EagerCNN):
    def __init__(self, 
                 model=None,
                 name='ActorNet',
                 out_dim=10,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 trainable=False,
                 max_action=None
                 ):
        super().__init__(model=model,name=name,out_dim=out_dim,opt=opt,lr=lr,l2_reg=l2_reg,l2_reg_scale=l2_reg_scale,trainable=trainable)
        self.max_action = max_action

    def _build(self):
        with tf.variable_scope(self.name):
            for l in range(len(self.model)):
                if l == len(self.model) - 1:
                    self.model[l][2] = tf.nn.tanh
                my_layer = eval('self.' + self.model[l][0])(self.model[l][1:])
                self._layers.append(my_layer)
            

    def inference(self, x):
        for i, my_layer in enumerate(self._layers):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            try:
                x = my_layer(x, training=self._trainable)
            except:
                x = my_layer(x)
        return self.max_action * x

    def optimize(self, loss, global_step, tape=None):
        assert tape is not None, 'please set tape in opmize'
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.method.apply_gradients(zip(grads, self.trainable_variables),global_step)

class CriticNet(EagerCNN):
    def __init__(self, 
                 model=None,
                 name='CriticNet',
                 out_dim=10,
                 opt=Adam,   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 trainable=False,
                 is_noisy=False,
                 is_categorical=False
                 ):
        super().__init__(model=model,name=name,out_dim=out_dim,opt=opt,lr=lr,l2_reg=l2_reg,l2_reg_scale=l2_reg_scale,trainable=trainable,is_noisy=is_noisy,is_categorical=is_categorical)


    def _build(self):
        for l in range(len(self.model)):
            # categorical DQN
            if l == len(self.model) - 1 and self.is_categorical:
                self.model[l][1] = self.out_dim * self.N_atoms   # 
            if self.is_noisy and self.model[l][0]=='fc':
                my_layer = self.noisy_dense(self.model[l][1:]) #noisy_net
            else:
                my_layer = eval('self.' + self.model[l][0])(self.model[l][1:])
            self._layers.append(my_layer)

    def inference(self, inputs):
        x, u = inputs
        for i, my_layer in enumerate(self._layers):
            if i > 0:
                x = tf.concat([x, u], axis=1)
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            try:
                x = my_layer(x, training=self._trainable)
            except:
                x = my_layer(x)
        return x

    def optimize(self, loss, global_step, tape=None):
        assert tape is not None, 'please set tape in opmize'
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.method.apply_gradients(zip(grads, self.trainable_variables),global_step)
