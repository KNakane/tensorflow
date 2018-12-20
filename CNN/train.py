import sys
sys.path.append('./utility')
sys.path.append('./network')
import tensorflow as tf
from model import DNN
from lenet import LeNet
from losses import classification_loss, add_to_watch_list
from data_load import Load
from utils import Utils

def set_model(outdim):
    model_set = [['conv', 5, 32, 1],
                 ['max_pool', 2, 2],
                 ['conv', 5, 64, 1],
                 ['max_pool', 2, 2],
                 ['dropout', 1024, tf.nn.relu, 0.5],
                 ['fc', outdim, None]]
    return model_set

def main(argv):
    print("---Start Learning------")
    print("data : {}".format(FLAGS.data))
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

    # load dataset
    data = Load(FLAGS.data)
    dataset = data.load(data.x_train, data.y_train, batch_size=batch_size, is_training=True)
    iterator = dataset.make_one_shot_iterator()
    inputs, labels = iterator.get_next()
    inputs = tf.reshape(inputs, (-1, data.size, data.size, data.channel)) / 255.0

    # build train operation
    global_step = tf.train.get_or_create_global_step()

    model_set = set_model(data.output_dim)
    model = LeNet(model=model_set, name='sample', lr=FLAGS.lr, opt=FLAGS.opt, trainable=True)
    logits = model.inference(inputs)
    logits  = tf.identity(logits, name="output_logits")
    loss = model.loss(logits, labels)
    #loss = classification_loss(logits, labels)
    #add_to_watch_list("loss/classification", loss)
    
    opt_op = model.optimize(loss, global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group([opt_op] + update_ops)
    predict = model.predict(logits)
    correct_prediction = tf.equal(tf.argmax(logits), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # logging for tensorboard
    util = Utils()
    util.conf_log()
    tf.summary.scalar('global_step', global_step)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('image', inputs)

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
        save_checkpoint_steps=save_checkpoint_steps,
        summary_dir=util.tf_board)
        
    with session:
        while not session.should_stop():
            session.run([train_op, labels, predict])


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