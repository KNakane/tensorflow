import os,sys
import random
import tensorflow as tf
sys.path.append('./utility')
sys.path.append('./network')
from unet import UNet
from utils import Utils
from data_load import Load
from util import loader as ld
from util import model
from util import repoter as rp

def load_dataset(train_rate):
    loader = ld.Loader(dir_original="data_set/VOCdevkit/VOC2012/JPEGImages",
                       dir_segmented="data_set/VOCdevkit/VOC2012/SegmentationClass")
    return loader.load_train_test(train_rate=train_rate, shuffle=False)

def main(argv):
    print("---Start Learning------")
    print("epoch : {}".format(FLAGS.n_epoch))
    print("batch_size : {}".format(FLAGS.batch_size))
    print("Optimizer : {}".format(FLAGS.opt))
    print("learning rate : {}".format(FLAGS.lr))
    print("-----------------------")

    checkpoints_to_keep = FLAGS.checkpoints_to_keep
    keep_checkpoint_every_n_hours = FLAGS.keep_checkpoint_every_n_hours
    max_steps = FLAGS.n_epoch
    save_checkpoint_steps = FLAGS.save_checkpoint_steps
    batch_size = FLAGS.batch_size

    # Load Dataset
    data = Load()
    x_train, y_train = data.get_data()
    dataset = data.load(x_train, y_train, batch_size)
    iterator = dataset.make_one_shot_iterator()
    inputs, labels = iterator.get_next()

    """
    # load dataset
    train, test = load_dataset(train_rate=FLAGS.trainrate)
    valid = train.perm(0, 30)
    test = test.perm(0, 150)
    """

    # build train operation
    global_step = tf.train.get_or_create_global_step()

    # Create a model
    model = UNet(model=None, name='U-Net', lr=FLAGS.lr, opt=FLAGS.opt, trainable=True)
    logits = model.inference(inputs)
    loss = model.loss(logits, labels)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                                  logits=logits))
    opt_op = model.optimize(loss, global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group([opt_op] + update_ops)
    
    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 3), tf.argmax(labels, 3))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # logging for tensorboard
    util = Utils()
    util.conf_log()
    tf.summary.scalar('global_step', global_step)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    # create saver
    scaffold = tf.train.Scaffold(
        saver=tf.train.Saver(
            max_to_keep=checkpoints_to_keep,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours))

    # create hooks
    hooks = []
    tf.logging.set_verbosity(tf.logging.INFO)
    metrics = {
        "global_step": global_step,
        "loss": loss,
        "accuracy": accuracy}
    hooks.append(tf.train.LoggingTensorHook(metrics, every_n_iter=100))
    hooks.append(tf.train.NanTensorHook(loss))
    if max_steps:
        hooks.append(tf.train.StopAtStepHook(last_step=max_steps))

    # training
    session = tf.train.MonitoredTrainingSession(
        checkpoint_dir=util.model_path,
        hooks=hooks,
        scaffold=scaffold,
        save_checkpoint_steps=save_checkpoint_steps)

    run_options = tf.RunOptions(output_partition_graphs=True)
    run_metadata = tf.RunMetadata()
        
    with session:
        while not session.should_stop():
            session.run([train_op],run_metadata=run_metadata, options=run_options)
    return

if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('n_epoch', '1000', 'Input max epoch')
    flags.DEFINE_integer('batch_size', '32', 'Input batch size')
    flags.DEFINE_float('lr', '0.001', 'Input learning rate')
    flags.DEFINE_string('opt', 'SGD', 'Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]')
    flags.DEFINE_float('trainrate','0.85', 'Training rate')
    flags.DEFINE_float('l2reg', '0.0001','L2 regularization')
    flags.DEFINE_integer('checkpoints_to_keep', 5,'checkpoint keep count')
    flags.DEFINE_integer('keep_checkpoint_every_n_hours', 1, 'checkpoint create ')
    flags.DEFINE_integer('save_checkpoint_steps', 1000,'save checkpoint step')
    tf.app.run()