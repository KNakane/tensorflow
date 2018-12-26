import os,sys
import tensorflow as tf
from data_load import GAN_Load

def main(args):
    # Setting
    batch_size = FLAGS.batch_size
    
    # Load Dataset
    data = GAN_Load(FLAGS.data)
    d_dataset = data.load(data.x_train, data.correct_label, batch_size=batch_size, is_training=True) #discriminator用
    g_dataset = data.load(data.z, data.fake_label, batch_size=batch_size, is_training=True)
    d_iterator = d_dataset.make_initializable_iterator()
    inputs, labels = iterator.get_next()
    inputs = tf.reshape(inputs, (-1, data.size, data.size, data.channel)) / 255.0

    return

if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('data', 'mnist', 'Choice the training data name -> ["mnist","cifar10","cifar100"]')
    flags.DEFINE_integer('n_epoch', '1000', 'Input max epoch')
    flags.DEFINE_integer('batch_size', '32', 'Input batch size')
    flags.DEFINE_float('lr', '0.001', 'Input learning rate')
    flags.DEFINE_string('opt', 'SGD', 'Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]')
    flags.DEFINE_integer('checkpoints_to_keep', 5,'checkpoint keep count')
    flags.DEFINE_integer('keep_checkpoint_every_n_hours', 1, 'checkpoint create ')
    flags.DEFINE_integer('save_checkpoint_steps', 1000,'save checkpoint step')
    tf.app.run()