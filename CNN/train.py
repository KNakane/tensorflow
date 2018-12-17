import tensorflow as tf
from model import MnistClassifier
from data_load import Load
from utils import Utils

def main(argv):
    print("---Start Learning------")
    print("data : {}".format(FLAGS.data))
    print("epoch : {}".format(FLAGS.n_epoch))
    print("batch_size : {}".format(FLAGS.batch_size))
    print("learning rate : {}".format(FLAGS.lr))
    print("Optimizer : {}".format(FLAGS.opt))
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

    # build train operation
    global_step = tf.train.get_or_create_global_step()
    model = MnistClassifier(hidden_size=512, classes=10, opt=FLAGS.opt, lr=FLAGS.lr)
    logits = model(inputs)
    loss = model.loss(logits, labels)
    train_op = model.optimize(loss)
    predict = model.predict(logits)
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.control_dependencies([train_op]):
        train_op = tf.assign_add(global_step, 1)

    # logging for tensorboard
    util = Utils()
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
        save_checkpoint_steps=save_checkpoint_steps)

    with session:
        while not session.should_stop():
            session.run([train_op, labels, predict])


if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('data', 'mnist', 'Choice the training data name -> ["mnist","cifar10","cifar100"]')
    flags.DEFINE_integer('n_epoch', '1000', 'Input max epoch')
    flags.DEFINE_integer('batch_size', '32', 'Input batch size')
    flags.DEFINE_float('lr', '0.1', 'Input learning rate')
    flags.DEFINE_string('opt','SGD','Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]')
    flags.DEFINE_integer('checkpoints_to_keep', 5,'checkpoint keep count')
    flags.DEFINE_integer('keep_checkpoint_every_n_hours', 1, 'checkpoint create ')
    flags.DEFINE_integer('save_checkpoint_steps', 1000,'save checkpoint step')
    tf.app.run()