import os,sys
sys.path.append('./utility')
sys.path.append('./dataset')
sys.path.append('./network')
import tensorflow as tf
import numpy as np
import time
from eager_load import Load
from eager_nn import EagerNN
from eager_model import LeNet
from eager_trainer import EagerTrain
from collections import OrderedDict

# Eager Mode
tf.enable_eager_execution()

def set_model(outdim):
    model_set = [['conv', 3, 32, 1, tf.nn.relu],
                 ['conv', 3, 32, 1, tf.nn.relu],
                 ['flat'],
                 ['fc', 50, tf.nn.relu],
                 ['fc', outdim, None]]
    return model_set

def main(argv):
    message = OrderedDict({
        "Network": FLAGS.network,
        "data": FLAGS.data,
        "epoch":FLAGS.n_epoch,
        "batch_size": FLAGS.batch_size,
        "Optimizer":FLAGS.opt,
        "learning_rate":FLAGS.lr,
        "l2_norm": FLAGS.l2_norm,
        "Augmentation": FLAGS.aug})
    
    data = Load(FLAGS.data)
    model_set = set_model(data.output_dim)

    model = eval(FLAGS.network)(model=model_set, name=FLAGS.network, out_dim=data.output_dim, lr=FLAGS.lr, opt=FLAGS.opt, l2_reg=FLAGS.l2_norm, trainable=True)

    #training
    trainer = EagerTrain(FLAGS, message, data, model, FLAGS.network)
    trainer.train()

    return

if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('network', 'EagerNN', 'Choice the training data name -> [EagerNN,LeNet]')
    flags.DEFINE_string('data', 'mnist', 'Choice the training data name -> ["mnist","cifar10","cifar100","kuzushiji"]')
    flags.DEFINE_integer('n_epoch', '1000', 'Input max epoch')
    flags.DEFINE_integer('batch_size', '32', 'Input batch size')
    flags.DEFINE_float('lr', '0.001', 'Input learning rate')
    flags.DEFINE_string('opt', 'SGD', 'Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]')
    flags.DEFINE_string('aug','None','Choice the Augmentation -> ["shift","mirror","rotate","shift_rotate","cutout","random_erace"]')
    flags.DEFINE_bool('l2_norm', 'False', 'Input learning rate')
    flags.DEFINE_string('init_model', 'None', 'Choice the checkpoint directpry(ex. ./results/181225_193106/model)')
    tf.app.run()
    