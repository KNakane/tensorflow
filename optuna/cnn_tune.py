#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append('./utility')
sys.path.append('./network')
sys.path.append('./dataset')
import optuna
import tensorflow as tf
from model import DNN
from lenet import LeNet
from load import Load
from utils import Utils
from functools import partial

def objective(args, trial):
    param = {
        'opt' : trial.suggest_categorical('opt', ['SGD','Momentum','Adadelta','Adagrad','Adam','RMSProp']),
        'lr' : trial.suggest_loguniform('lr', 1e-2, 1e-5),
    }
    max_steps = FLAGS.n_epoch
    batch_size = FLAGS.batch_size

    # load dataset
    data = Load(FLAGS.data)
    dataset = data.load(data.x_train, data.y_train, batch_size=batch_size, is_training=True)
    iterator = dataset.make_initializable_iterator()
    inputs, labels = iterator.get_next()
    inputs = tf.reshape(inputs, (-1, data.size, data.size, data.channel)) / 255.0

    # build train operation
    global_step = tf.train.get_or_create_global_step()

    #training
    #model_set = set_model(data.output_dim)
    model = LeNet(model=None, name='LeNet', out_dim=data.output_dim, lr=param['lr'], opt=param['opt'], trainable=True)
    logits = model.inference(inputs)
    logits  = tf.identity(logits, name="output_logits")
    train_loss = model.loss(logits, labels)

    opt_op = model.optimize(train_loss, global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group([opt_op] + update_ops)

    #test
    test_inputs, test_labels = data.load_test(data.x_test, data.y_test)
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        logits = model.inference(test_inputs)
    logits  = tf.identity(logits, name="test_logits")
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(test_labels, 1))
    test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_fn(scaffold, session):
        session.run(iterator.initializer,feed_dict={data.features_placeholder: data.x_train,
                                                    data.labels_placeholder: data.y_train})

    # create saver
    scaffold = tf.train.Scaffold(init_fn=init_fn)

    # create hooks
    hooks = []
    hooks.append(tf.train.NanTensorHook(train_loss))
    if max_steps:
        hooks.append(tf.train.StopAtStepHook(last_step=max_steps))

    # training
    session = tf.train.MonitoredTrainingSession(
        hooks=hooks,
        scaffold=scaffold)
        
    with session:
        while not session.should_stop():
            _, accuracy = session.run([train_op, test_accuracy])
    
    return -accuracy

def main(argv):
    study_name = 'example-study'
    study = optuna.create_study(study_name=study_name, storage='sqlite:///results/example.db')
    f = partial(objective, FLAGS)
    study.optimize(f, n_trials=1000)
    return

if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('network', 'DNN', 'Choice the training data name -> [DNN,LeNet]')
    flags.DEFINE_string('data', 'mnist', 'Choice the training data name -> ["mnist","cifar10","cifar100","kuzushiji"]')
    flags.DEFINE_integer('n_epoch', '5000', 'Input max epoch')
    flags.DEFINE_integer('batch_size', '32', 'Input batch size')
    flags.DEFINE_integer('checkpoints_to_keep', 5,'checkpoint keep count')
    flags.DEFINE_integer('keep_checkpoint_every_n_hours', 1, 'checkpoint create ')
    flags.DEFINE_integer('save_checkpoint_steps', 1000,'save checkpoint step')
    tf.app.run()