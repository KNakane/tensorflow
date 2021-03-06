import sys
sys.path.append('./trainer')
sys.path.append('./utility')
sys.path.append('./network')
sys.path.append('./dataset')
import tensorflow as tf
from ae import AutoEncoder, VAE, CVAE
from load import Load
from utils import Utils
from trainer import AETrainer
from collections import OrderedDict
from hooks import SavedModelBuilderHook, MyLoggerHook, AEHook


def set_model(outdim, size=28, channel=1):
    """
    encode = [['conv', 5, 16, 2, tf.nn.leaky_relu],
              ['conv', 5, 32, 2, tf.nn.leaky_relu],
              ['fc', 40, None]]

    decode = [['fc', 7*7*32, tf.nn.relu],
              ['reshape', [-1, 7, 7, 32]],
              ['deconv',  5, 16, 2, tf.nn.relu],
              ['deconv',  5, outdim, 2, None]]
    """
    encode = [['fc', 256, tf.nn.elu],
              ['fc', 128, tf.nn.elu],
              ['fc', 64, tf.nn.elu],
              ['fc', outdim, None]]

    decode = [['fc', 64, tf.nn.elu],
              ['fc', 128, tf.nn.elu],
              ['fc', 256, tf.nn.elu],
              ['fc', size*size*channel, tf.nn.sigmoid],
              ['reshape', [-1, size, size, channel]]]
    return encode, decode

def main(argv):
    message = OrderedDict({
        "Network": FLAGS.network,
        "data": FLAGS.data,
        "epoch":FLAGS.n_epoch,
        "batch_size": FLAGS.batch_size,
        "Optimizer":FLAGS.opt,
        "learning_rate":FLAGS.lr,
        "Denoising":FLAGS.denoise,
        "l2_norm": FLAGS.l2_norm,
        "Augmentation": FLAGS.aug})

    ## load dataset
    data = Load(FLAGS.data)
    
    ## setting models
    encode, decode = set_model(outdim=40, size=data.size, channel=data.channel)
    model = eval(FLAGS.network)(encode=encode, decode=decode, denoise=FLAGS.denoise, size=data.size, channel=data.channel, name=FLAGS.network, out_dim=data.output_dim, lr=FLAGS.lr, opt=FLAGS.opt, trainable=True)

    #training
    trainer = AETrainer(FLAGS, message, data, model, FLAGS.network)
    trainer.train()
    return

if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('network', 'AutoEncoder', 'Choice the training data name -> [AutoEncoder,VAE]')
    flags.DEFINE_string('data', 'mnist', 'Choice the training data name -> ["mnist","cifar10","cifar100","kuzushiji"]')
    flags.DEFINE_integer('n_epoch', '1000', 'Input max epoch')
    flags.DEFINE_integer('batch_size', '32', 'Input batch size')
    flags.DEFINE_float('lr', '0.001', 'Input learning rate')
    flags.DEFINE_string('opt', 'SGD', 'Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]')
    flags.DEFINE_string('aug','None','Choice the Augmentation -> ["shift","mirror","rotate","shift_rotate","cutout","random_erace"]')
    flags.DEFINE_bool('l2_norm', 'False', 'Input learning rate')
    flags.DEFINE_bool('denoise', 'False', 'True : Denoising AE, False : standard AE')
    flags.DEFINE_string('init_model', 'None', 'Choice the checkpoint directpry(ex. ./results/181225_193106/model)')
    flags.DEFINE_integer('checkpoints_to_keep', 5,'checkpoint keep count')
    flags.DEFINE_integer('keep_checkpoint_every_n_hours', 1, 'checkpoint create ')
    flags.DEFINE_integer('save_checkpoint_steps', 1000,'save checkpoint step')
    tf.app.run()