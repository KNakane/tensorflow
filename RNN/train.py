import os, sys
sys.path.append('./utility')
sys.path.append('./network')
sys.path.append('./dataset')
import tensorflow as tf
from rnn_load import RNN_Load
from lstm import LSTM
from trainer import Train
from collections import OrderedDict

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

    # prepare training
    ## load dataset
    data = RNN_Load(FLAGS.data)
    print(data.x_train.shape)
    sys.exit()
    ## setting models
    model_set = None #set_model(data.output_dim)
    model = eval(FLAGS.network)(model=model_set, name=FLAGS.network, out_dim=data.output_dim, lr=FLAGS.lr, opt=FLAGS.opt, l2_reg=FLAGS.l2_norm, trainable=True)

    #training
    trainer = Train(FLAGS, message, data, model, FLAGS.network)
    trainer.train()


if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('network', 'LSTM', 'Choice the training data name -> [LSTM]')
    flags.DEFINE_string('data', 'sample', 'Choice the training data name -> [sample, imdb]')
    flags.DEFINE_integer('n_epoch', '1000', 'Input max epoch')
    flags.DEFINE_integer('batch_size', '32', 'Input batch size')
    flags.DEFINE_float('lr', '0.001', 'Input learning rate')
    flags.DEFINE_string('opt', 'SGD', 'Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]')
    flags.DEFINE_string('aug','None','Choice the Augmentation -> ["shift","mirror","rotate","shift_rotate","cutout","random_erace"]')
    flags.DEFINE_bool('l2_norm', 'False', 'Input learning rate')
    flags.DEFINE_string('init_model', 'None', 'Choice the checkpoint directpry(ex. ./results/181225_193106/model)')
    flags.DEFINE_integer('checkpoints_to_keep', 5,'checkpoint keep count')
    flags.DEFINE_integer('keep_checkpoint_every_n_hours', 1, 'checkpoint create ')
    flags.DEFINE_integer('save_checkpoint_steps', 100,'save checkpoint step')
    tf.app.run()