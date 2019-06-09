import os,sys
sys.path.append('./CNN')
sys.path.append('./dataset')
import tensorflow as tf
from gan import GAN, DCGAN, WGAN, WGAN_GP, LSGAN, ACGAN, DRAGAN
from utils import Utils
from load import Load
from collections import OrderedDict


def main(args):
    message = OrderedDict({
        "Network": FLAGS.network,
        "Conditional": FLAGS.conditional,
        "data": FLAGS.data,
        "z_dim": FLAGS.z_dim,
        "epoch":FLAGS.n_epoch,
        "batch_size": FLAGS.batch_size,
        "Optimizer":FLAGS.opt,
        "learning_rate":FLAGS.lr,
        "n_disc_update":FLAGS.n_disc_update,
        "l2_norm":FLAGS.l2_norm,
        "Augmentation": FLAGS.aug})

    

if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('network', 'GAN', 'Choice the training data name -> [GAN, DCGAN, WGAN, WGAN_GP, LSGAN, ACGAN, DRAGAN]')
    flags.DEFINE_string('data', 'mnist', 'Choice the training data name -> ["mnist","cifar10","cifar100", "kuzushiji"]')
    flags.DEFINE_integer('n_epoch', '50000', 'Input max epoch')
    flags.DEFINE_integer('batch_size', '32', 'Input batch size')
    flags.DEFINE_float('lr', '2e-4', 'Input learning rate')
    flags.DEFINE_string('opt', 'SGD', 'Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]')
    flags.DEFINE_string('aug','None','Choice the Augmentation -> ["shift","mirror","rotate","shift_rotate","cutout","random_erace"]')
    flags.DEFINE_bool('l2_norm', 'False', 'Input learning rate')
    flags.DEFINE_integer('z_dim', '100', 'Latent z dimension')
    flags.DEFINE_bool('conditional', 'False', 'Conditional true or false')
    flags.DEFINE_integer('n_disc_update', '2', 'Input max epoch')
    flags.DEFINE_string('init_model', 'None', 'Choice the checkpoint directpry(ex. ./results/181225_193106/model)')
    flags.DEFINE_integer('checkpoints_to_keep', 5,'checkpoint keep count')
    flags.DEFINE_integer('keep_checkpoint_every_n_hours', 1, 'checkpoint create ')
    flags.DEFINE_integer('save_checkpoint_steps', 1000,'save checkpoint step')
    tf.app.run()