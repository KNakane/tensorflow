import sys
sys.path.append('./utility')
sys.path.append('./network')
sys.path.append('./dataset')
import tensorflow as tf
from cnn import CNN
from lenet import LeNet
from load import Load
from trainer import Train
from collections import OrderedDict

def set_model(outdim):
    model_set = [['Residual', 3, 32, 2, False, 1],
                 ['Residual', 3, 32, 2, False, 2],
                 ['Residual', 3, 32, 2, False, 3],
                 ['fc', 100, tf.nn.relu],
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
        "Augmentation": FLAGS.aug})

    # prepare training
    ## load dataset
    data = Load(FLAGS.data)
    ## setting models
    model_set = set_model(data.output_dim)
    model = eval(FLAGS.network)(model=model_set, name=FLAGS.network, out_dim=data.output_dim, lr=FLAGS.lr, opt=FLAGS.opt, l2_reg=FLAGS.l2_norm, trainable=True)

    #training
    trainer = Train(FLAGS, message, data, model, FLAGS.network)
    trainer.train()


if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('network', 'CNN', 'Choice the training data name -> [CNN,LeNet]')
    flags.DEFINE_string('data', 'mnist', 'Choice the training data name -> ["mnist","cifar10","cifar100","kuzushiji"]')
    flags.DEFINE_integer('n_epoch', '1000', 'Input max epoch')
    flags.DEFINE_integer('batch_size', '32', 'Input batch size')
    flags.DEFINE_float('lr', '0.001', 'Input learning rate')
    flags.DEFINE_string('opt', 'SGD', 'Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]')
    flags.DEFINE_string('aug','None','Choice the Augmentation -> ["shift","mirror","rotate","shift_rotate","cutout"]')
    flags.DEFINE_bool('l2_norm', 'False', 'Input learning rate')
    flags.DEFINE_integer('checkpoints_to_keep', 5,'checkpoint keep count')
    flags.DEFINE_integer('keep_checkpoint_every_n_hours', 1, 'checkpoint create ')
    flags.DEFINE_integer('save_checkpoint_steps', 100,'save checkpoint step')
    tf.app.run()