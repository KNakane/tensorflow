import sys
sys.path.append('./utility')
sys.path.append('./network')
sys.path.append('./dataset')
import tensorflow as tf
from ae import AutoEncoder, VAE
from AE_load import AE_Load
from trainer import Train
from collections import OrderedDict


def set_model(outdim):
    encode = [['conv', 5, 32, 1, tf.nn.relu],
              ['max_pool', 2, 2],
              ['conv', 5, 64, 1, tf.nn.relu],
              ['max_pool', 2, 2],
              ['fc', 20, None]]

    decode = [['fc', 49, tf.nn.relu],
              ['deconv',  5, 32, 2, tf.nn.relu],
              ['deconv',  5, outdim, 2, None]]
    return encode, decode

def main(argv):
    message = OrderedDict({
        "Network": FLAGS.network,
        "data": FLAGS.data,
        "epoch":FLAGS.n_epoch,
        "batch_size": FLAGS.batch_size,
        "Optimizer":FLAGS.opt,
        "learning_rate":FLAGS.lr})

    # prepare training
    ## load dataset
    data = AE_Load(FLAGS.data)
    ## setting models
    encode, decode = set_model(data.channel)
    model = eval(FLAGS.network)(encode=encode, decode=decode, name=FLAGS.network, out_dim=data.output_dim, lr=FLAGS.lr, opt=FLAGS.opt, trainable=True)

    #training
    trainer = Train(FLAGS, message, data, model, FLAGS.network)
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
    flags.DEFINE_integer('checkpoints_to_keep', 5,'checkpoint keep count')
    flags.DEFINE_integer('keep_checkpoint_every_n_hours', 1, 'checkpoint create ')
    flags.DEFINE_integer('save_checkpoint_steps', 1000,'save checkpoint step')
    tf.app.run()