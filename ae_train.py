import os, sys
from absl import app
from absl import flags
import tensorflow as tf
from AutoEncoder.model import AutoEncoder
from dataset.load import Load
from trainer.trainer import AE_Trainer
from collections import OrderedDict

def main(args):

    message = OrderedDict({
        "Network": FLAGS.network,
        "data": FLAGS.data,
        "epoch":FLAGS.n_epoch,
        "batch_size": FLAGS.batch_size,
        "Denoising":FLAGS.denoise,
        "Optimizer":FLAGS.opt,
        "learning_rate":FLAGS.lr,
        "l2_norm": FLAGS.l2_norm,
        "Augmentation": FLAGS.aug})

    data = Load(FLAGS.data)
    model = eval(FLAGS.network)(name=FLAGS.network, out_dim=data.output_dim, lr=FLAGS.lr, opt=FLAGS.opt, l2_reg=FLAGS.l2_norm)

    #training
    trainer = AE_Trainer(FLAGS, message, data, model, FLAGS.network)
    trainer.train()


    return

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('network', 'LeNet', 'Choice the training data name -> [AutoEncoder,VAE]')
    flags.DEFINE_string('data', 'mnist', 'Choice the training data name -> ["mnist","cifar10","cifar100","kuzushiji"]')
    flags.DEFINE_integer('n_epoch', '1000', 'Input max epoch')
    flags.DEFINE_integer('batch_size', '32', 'Input batch size')
    flags.DEFINE_bool('denoise', 'False', 'True : Denoising AE, False : standard AE')
    flags.DEFINE_float('lr', '0.001', 'Input learning rate')
    flags.DEFINE_string('opt', 'SGD', 'Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]')
    flags.DEFINE_string('aug','None','Choice the Augmentation -> ["shift","mirror","rotate","shift_rotate","cutout","random_erace"]')
    flags.DEFINE_bool('l2_norm', 'False', 'Input learning rate')
    flags.DEFINE_string('init_model', 'None', 'Choice the checkpoint directpry(ex. ./results/181225_193106/model)')
    flags.DEFINE_integer('checkpoints_to_keep', 5,'checkpoint keep count')
    flags.DEFINE_integer('keep_checkpoint_every_n_hours', 1, 'checkpoint create ')
    flags.DEFINE_integer('save_checkpoint_steps', 100,'save checkpoint step')
    app.run(main)