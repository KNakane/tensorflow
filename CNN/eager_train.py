import os,sys
sys.path.append('./dataset')
sys.path.append('./network')
import tensorflow as tf
import numpy as np
from load import Load
from eager_cnn import EagerCNN
# Eager Mode
tf.enable_eager_execution()

def set_model(outdim):
    model_set = [['flat'],
                 ['fc', 500,tf.nn.relu],
                 ['fc', 50, tf.nn.relu],
                 ['fc', outdim, tf.nn.softmax]]
    return model_set

def main(argv):
    ## load dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .batch(FLAGS.batch_size)
    .shuffle(10000))

    train_dataset = (
        train_dataset.map(lambda x, y: 
                        (tf.div_no_nan(tf.cast(x, tf.float32), 255.0), 
                        tf.reshape(tf.one_hot(y, 10), (-1, 10))))
    )
    model_set = set_model(10)
    model = EagerCNN(model=model_set, name=FLAGS.network, out_dim=10, lr=FLAGS.lr, opt=FLAGS.opt, l2_reg=FLAGS.l2_norm, trainable=True)
    for j in range(FLAGS.n_epoch):
        running_loss = 0
        for i, (x_, y_) in enumerate(train_dataset):
            x = tf.transpose(x_, perm=[0, 2, 3, 1])
            with tf.GradientTape() as tape:
                y_pre = model.inference(x)
                loss = model.loss(y_, y_pre)
            model.optimize(loss,tape)
            running_loss += loss
        print("-----epoch {} -----".format(j + 1))
        print("loss: ", running_loss.numpy()/(i + 1))
    return

if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('network', 'CNN', 'Choice the training data name -> [CNN,LeNet]')
    flags.DEFINE_string('data', 'mnist', 'Choice the training data name -> ["mnist","cifar10","cifar100","kuzushiji"]')
    flags.DEFINE_integer('n_epoch', '1000', 'Input max epoch')
    flags.DEFINE_integer('batch_size', '32', 'Input batch size')
    flags.DEFINE_float('lr', '0.001', 'Input learning rate')
    flags.DEFINE_string('opt', 'SGD', 'Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]')
    flags.DEFINE_string('aug','None','Choice the Augmentation -> ["shift","mirror","rotate","shift_rotate","cutout","random_erace"]')
    flags.DEFINE_bool('l2_norm', 'False', 'Input learning rate')
    flags.DEFINE_integer('checkpoints_to_keep', 5,'checkpoint keep count')
    flags.DEFINE_integer('keep_checkpoint_every_n_hours', 1, 'checkpoint create ')
    flags.DEFINE_integer('save_checkpoint_steps', 100,'save checkpoint step')
    tf.app.run()
    