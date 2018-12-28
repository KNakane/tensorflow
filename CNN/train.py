import sys
sys.path.append('./utility')
sys.path.append('./network')
sys.path.append('./dataset')
import tensorflow as tf
from model import DNN
from lenet import LeNet
from losses import classification_loss, add_to_watch_list
from load import Load
from kuzushiji_load import kuzushiji_Load
from utils import Utils

def set_model(outdim):
    model_set = [['conv', 5, 32, 1, tf.nn.relu],
                 ['max_pool', 2, 2],
                 ['conv', 5, 64, 1, tf.nn.relu],
                 ['max_pool', 2, 2],
                 ['dropout', 1024, tf.nn.relu, 0.5],
                 ['fc', outdim, None]]
    return model_set

def main(argv):
    print("---Start Learning------")
    print("Network : {}".format(FLAGS.network))
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
    iterator = dataset.make_initializable_iterator()
    inputs, labels = iterator.get_next()
    inputs = tf.reshape(inputs, (-1, data.size, data.size, data.channel)) / 255.0

    # build train operation
    global_step = tf.train.get_or_create_global_step()

    #training
    model_set = set_model(data.output_dim)
    model = eval(FLAGS.network)(model=model_set, name=FLAGS.network, lr=FLAGS.lr, opt=FLAGS.opt, trainable=True)
    logits = model.inference(inputs)
    logits  = tf.identity(logits, name="output_logits")
    train_loss = model.loss(logits, labels)

    opt_op = model.optimize(train_loss, global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group([opt_op] + update_ops)
    predict = model.predict(logits)
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels, 1))
    train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #test
    test_inputs, test_labels = data.load_test(data.x_test, data.y_test)
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        logits = model.inference(test_inputs)
    logits  = tf.identity(logits, name="test_logits")
    test_loss = model.loss(logits, test_labels)
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(test_labels, 1))
    test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # logging for tensorboard
    util = Utils(prefix='CNN')
    util.conf_log()

    tf.summary.scalar('test/loss', test_loss)
    tf.summary.scalar('test/accuracy', test_accuracy)
    tf.summary.image('test/image', test_inputs)

    tf.summary.scalar('train/loss', train_loss)
    tf.summary.scalar('train/accuracy', train_accuracy)
    tf.summary.image('train/image', inputs)

    def init_fn(scaffold, session):
        session.run(iterator.initializer,feed_dict={data.features_placeholder: data.x_train,
                                                    data.labels_placeholder: data.y_train})

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
        "test loss": test_loss,
        "test accuracy":test_accuracy,
        "global_step": global_step,
        "train loss": train_loss,
        "train accuracy":train_accuracy}
    hooks.append(tf.train.LoggingTensorHook(metrics, every_n_iter=100))
    hooks.append(tf.train.NanTensorHook(train_loss))
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
    flags.DEFINE_string('network', 'DNN', 'Choice the training data name -> [DNN,LeNet]')
    flags.DEFINE_string('data', 'mnist', 'Choice the training data name -> ["mnist","cifar10","cifar100","kuzushiji"]')
    flags.DEFINE_integer('n_epoch', '1000', 'Input max epoch')
    flags.DEFINE_integer('batch_size', '32', 'Input batch size')
    flags.DEFINE_float('lr', '0.001', 'Input learning rate')
    flags.DEFINE_string('opt', 'SGD', 'Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]')
    flags.DEFINE_integer('checkpoints_to_keep', 5,'checkpoint keep count')
    flags.DEFINE_integer('keep_checkpoint_every_n_hours', 1, 'checkpoint create ')
    flags.DEFINE_integer('save_checkpoint_steps', 1000,'save checkpoint step')
    tf.app.run()