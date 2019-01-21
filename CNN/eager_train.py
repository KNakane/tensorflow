import os,sys
sys.path.append('./dataset')
sys.path.append('./network')
import tensorflow as tf
import numpy as np
from eager_load import Load
from eager_cnn import EagerCNN
# Eager Mode
tf.enable_eager_execution()

def set_model(outdim):
    model_set = [['conv', 3, 32, 1, tf.nn.relu],
                 ['conv', 3, 32, 1, tf.nn.relu],
                 ['flat'],
                 ['fc', 50, tf.nn.relu],
                 ['fc', outdim, tf.nn.softmax]]
    return model_set

def main(argv):
    ## load dataset
    global_steps=tf.train.get_or_create_global_step()
    data = Load(FLAGS.data)
    dataset = data.load(data.x_train, data.y_train, batch_size=FLAGS.batch_size, is_training=True, augmentation=FLAGS.aug)

    model_set = set_model(10)
    model = EagerCNN(model=model_set, name=FLAGS.network, out_dim=10, lr=FLAGS.lr, opt=FLAGS.opt, l2_reg=FLAGS.l2_norm, trainable=True)
    for j in range(FLAGS.n_epoch):
        running_loss = 0
        for (batch, (images, labels)) in enumerate(dataset):
            with tf.GradientTape() as tape:
                y_pre = model.inference(images)
                loss = model.loss(labels, y_pre)
            model.optimize(loss, global_steps, tape)
            running_loss += loss
        print("-----epoch {} -----".format(j + 1))
        print("loss: ", running_loss.numpy()/(batch + 1))
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
    