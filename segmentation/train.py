import os,sys
import random
import tensorflow as tf
sys.path.append('./utility')
sys.path.append('./network')
sys.path.append('./dataset')
from unet import UNet
from utils import Utils
from segment_load import Load
from collections import OrderedDict
from hooks import SavedModelBuilderHook, MyLoggerHook

def main(argv):
    message = OrderedDict({
        "epoch":FLAGS.n_epoch,
        "batch_size": FLAGS.batch_size,
        "Optimizer":FLAGS.opt,
        "learning_rate":FLAGS.lr})

    checkpoints_to_keep = FLAGS.checkpoints_to_keep
    keep_checkpoint_every_n_hours = FLAGS.keep_checkpoint_every_n_hours
    max_steps = FLAGS.n_epoch
    save_checkpoint_steps = FLAGS.save_checkpoint_steps
    batch_size = FLAGS.batch_size

    # Load Dataset
    data = Load()
    x_train, y_train = data.get_data()
    dataset = data.load(x_train, y_train, batch_size, buffer_size=1000, is_training=True)
    iterator = dataset.make_initializable_iterator()
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
    model = UNet(model=None, name='U-Net', lr=FLAGS.lr, opt=FLAGS.opt, trainable=True, output_dim=len(data.category))
    logits = model.inference(inputs)
    loss = model.loss(logits, labels)
    
    opt_op = model.optimize(loss, global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group([opt_op] + update_ops)
    
    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 3), tf.argmax(labels, 3))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    mIoU, _ = tf.metrics.mean_iou(logits, labels, len(data.category))

    # logging for tensorboard
    util = Utils(prefix='segment')
    util.conf_log()
    tf.summary.scalar('global_step', global_step)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('mIoU', mIoU)
    tf.summary.image('image', inputs)

    def init_fn(scaffold, session):
        session.run(iterator.initializer,feed_dict={data.features_placeholder: x_train,
                                                    data.labels_placeholder: y_train})

    # create saver
    scaffold = tf.train.Scaffold(
        init_fn=init_fn,
        saver=tf.train.Saver(
            max_to_keep=checkpoints_to_keep,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours))

    # create hooks
    hooks = []
    tf.logging.set_verbosity(tf.logging.INFO)
    metrics = {
        "global_step": global_step,
        "loss": loss,
        "accuracy": accuracy,
        "mIoU": mIoU}
    hooks.append(MyLoggerHook(message, util.log_dir, metrics, every_n_iter=50))
    hooks.append(tf.train.NanTensorHook(loss))
    if max_steps:
        hooks.append(tf.train.StopAtStepHook(last_step=max_steps))
        
    # training
    session = tf.train.MonitoredTrainingSession(
        checkpoint_dir=util.model_path,
        hooks=hooks,
        scaffold=scaffold,
        save_checkpoint_steps=save_checkpoint_steps,
        summary_dir=util.tf_board)
        
    with session:
        while not session.should_stop():
            session.run([train_op, loss, mIoU, global_step])
    return

if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('n_epoch', '1000', 'Input max epoch')
    flags.DEFINE_integer('batch_size', '32', 'Input batch size')
    flags.DEFINE_float('lr', '0.0001', 'Input learning rate')
    flags.DEFINE_string('opt', 'SGD', 'Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]')
    flags.DEFINE_float('trainrate','0.85', 'Training rate')
    flags.DEFINE_float('l2reg', '0.0001','L2 regularization')
    flags.DEFINE_integer('checkpoints_to_keep', 5,'checkpoint keep count')
    flags.DEFINE_integer('keep_checkpoint_every_n_hours', 1, 'checkpoint create ')
    flags.DEFINE_integer('save_checkpoint_steps', 100,'save checkpoint step')
    tf.app.run()
