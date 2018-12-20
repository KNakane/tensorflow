import random
import tensorflow as tf

from util import loader as ld
from util import model
from util import repoter as rp


def load_dataset(train_rate):
    origin_dir = '/home/rl/Desktop/tensorflow/models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012'
    loader = ld.Loader(dir_original=origin_dir + "/JPEGImages",
                       dir_segmented=origin_dir + "/SegmentationClass")
    return loader.load_train_test(train_rate=train_rate, shuffle=False)

def get_flag():
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_boolean('g', True , 'Using GPUs')
    flags.DEFINE_integer('e', 250, 'Number of epochs')
    flags.DEFINE_integer('b', 32, 'Batch size')
    flags.DEFINE_float('t', 0.85, 'Training rate')
    flags.DEFINE_boolean('a', True, 'Data Augmentation')
    flags.DEFINE_float('r', 0.0001, 'L2 regularization')
    flags.DEFINE_integer('checkpoints_to_keep', 5,'checkpoint keep count')
    flags.DEFINE_integer('keep_checkpoint_every_n_hours', 1, 'checkpoint create ')
    flags.DEFINE_integer('max_steps', 10000, 'max trainig step')
    flags.DEFINE_integer('save_checkpoint_steps', 1000, 'save checkpoint step')

    return FLAGS

def main(flags):
    train, test = load_dataset(train_rate=flags.t)

    # build train operation
    global_step = tf.train.get_or_create_global_step()
    model = model.UNet(l2_reg=flags.r).model
    logits = model.outputs
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=model.teacher,
                                                                        logits=logits))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
    
    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(model_unet.outputs, 3), tf.argmax(model_unet.teacher, 3))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # logging for tensorboard
    tf.summary.scalar('global_step', global_step)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    #tf.summary.image('image', inputs)

    gpu_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7), device_count={'GPU': 1},
                                log_device_placement=False, allow_soft_placement=True)

    # create saver
    scaffold = tf.train.Scaffold(
        saver=tf.train.Saver(
            max_to_keep=flags.checkpoints_to_keep,
            keep_checkpoint_every_n_hours=flags.keep_checkpoint_every_n_hours))
    
    # create hooks
    hooks = []
    tf.logging.set_verbosity(tf.logging.INFO)
    metrics = {
        "global_step": global_step,
        "loss": loss,
        "accuracy": accuracy}
    hooks.append(tf.train.LoggingTensorHook(metrics, every_n_iter=100))
    hooks.append(tf.train.NanTensorHook(loss))
    if flags.max_steps:
        hooks.append(tf.train.StopAtStepHook(last_step=flags.max_steps))

    # training
    session = tf.train.MonitoredTrainingSession(
        checkpoint_dir=model_path,
        hooks=hooks,
        scaffold=scaffold,
        save_checkpoint_steps=flags.save_checkpoint_steps,
        config=gpu_config)

    with session:
        while not session.should_stop():
            session.run([train_op])

    return


if __name__ == '__main__':
    #parser = get_parser().parse_args()
    #train(parser)
    flags = get_flag()
    main(flags)
