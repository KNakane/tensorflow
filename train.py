import tensorflow as tf
from collections import OrderedDict
from network.cnn import CNN
from network.lenet import LeNet, VGG
from network.resnet import ResNet, ResNeXt, SENet, sSENet, scSENet
from network.dense_net import DenseNet
from dataset.load import Load
from trainer.trainer import Train


def set_model(outdim):
    model_set = [['Residual', 3, 16, 2, False, 1],
                 ['Residual', 3, 32, 2, False, 2],
                 ['Residual', 3, 64, 2, False, 3],
                 ['gap', outdim]]
    return model_set

def image_recognition(args):
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
    data = Load(FLAGS.data)
    ## setting models
    model_set = set_model(data.output_dim)
    model = eval(FLAGS.network)(model=model_set, name=FLAGS.network, out_dim=data.output_dim, lr=FLAGS.lr, opt=FLAGS.opt, l2_reg=FLAGS.l2_norm, trainable=True)

    #training
    trainer = Train(FLAGS, message, data, model, FLAGS.network)
    trainer.train()
    return

def construction_image(args):
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

    return

def GAN_fn(args):
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

    return

def main(args):
    if FLAGS.network == 'CNN' or FLAGS.network == 'LeNet' or FLAGS.network == 'VGG' or FLAGS.network == 'ResNet' or FLAGS.network == 'ResNeXt' or FLAGS.network == 'SENet' or FLAGS.network == 'DenseNet' or FLAGS.network == 'sSENet':
        image_recognition(args)
    elif FLAGS.network == 'AutoEncoder' or FLAGS.network == 'VAE':
        construction_image(args)
    elif FLAGS.network == 'GAN' or FLAGS.network == 'DCGAN':
        GAN_fn(args)
    else:
        raise NotImplementedError()
    return


if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('network', 'CNN', 'Choice the training data name -> [CNN, LeNet, VGG, ResNet, ResNeXt, SENet, DenseNet, sSENet, AutoEncoder, VAE]')
    flags.DEFINE_string('data', 'mnist', 'Choice the training data name -> ["mnist","cifar10","cifar100","kuzushiji"]')
    flags.DEFINE_integer('n_epoch', 1000, 'Input max epoch')
    flags.DEFINE_integer('batch_size', 32, 'Input batch size')
    flags.DEFINE_float('lr', 0.001, 'Input learning rate')
    flags.DEFINE_string('opt', 'SGD', 'Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]')
    flags.DEFINE_string('aug',None,'Choice the Augmentation -> ["shift","mirror","rotate","shift_rotate","cutout","random_erace"]')
    flags.DEFINE_bool('l2_norm', False, 'Input learning rate')
    flags.DEFINE_bool('denoise', False, 'True : Denoising AE, False : standard AE')
    flags.DEFINE_string('init_model', None, 'Choice the checkpoint directpry(ex. ./results/181225_193106/model)')
    flags.DEFINE_integer('checkpoints_to_keep', 5,'checkpoint keep count')
    flags.DEFINE_integer('keep_checkpoint_every_n_hours', 1, 'checkpoint create ')
    flags.DEFINE_integer('save_checkpoint_steps', 100,'save checkpoint step')
    tf.app.run()