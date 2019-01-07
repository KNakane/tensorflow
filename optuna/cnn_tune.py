#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append('./utility')
sys.path.append('./network')
sys.path.append('./dataset')
import tensorflow as tf
import optuna
from cnn import CNN
from lenet import LeNet
from load import Load
from trainer import Train
from search import Optuna
from functools import partial

def objective(args, trial):
    tf.reset_default_graph()
    param = {
        'opt' : trial.suggest_categorical('opt', ['SGD','Momentum','Adadelta','Adagrad','Adam','RMSProp']),
        'lr' : trial.suggest_loguniform('lr', 8e-5, 8e-2),
    }

    # prepare training
    ## load dataset
    data = Load(FLAGS.data)

    ## setting models
    #model_set = set_model(data.output_dim)
    #model = eval(FLAGS.network)(model=model_set, name=FLAGS.network, out_dim=data.output_dim, lr=FLAGS.lr, opt=FLAGS.opt, trainable=True)
    model = LeNet(model=None, name='LeNet', out_dim=data.output_dim, lr=param['lr'], opt=param['opt'], trainable=True)

    #training
    trainer = Train(FLAGS=FLAGS, message=None, data=data, model=model, name='tuning')
    _, _, test_accuracy = trainer.train()
    return -test_accuracy
    

def main(argv):
    op = Optuna('example-study')
    op.search(objective, objective, FLAGS.n_trials)
    return

if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('network', 'CNN', 'Choice the training data name -> [CNN,LeNet]')
    flags.DEFINE_string('data', 'mnist', 'Choice the training data name -> ["mnist","cifar10","cifar100","kuzushiji"]')
    flags.DEFINE_integer('n_epoch', '3000', 'Input max epoch')
    flags.DEFINE_integer('n_trials', '1000', 'Input trial epoch')
    flags.DEFINE_integer('batch_size', '32', 'Input batch size')
    flags.DEFINE_integer('checkpoints_to_keep', 5,'checkpoint keep count')
    flags.DEFINE_integer('keep_checkpoint_every_n_hours', 1, 'checkpoint create ')
    flags.DEFINE_integer('save_checkpoint_steps', 1000,'save checkpoint step')
    tf.app.run()