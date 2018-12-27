import os,sys
sys.path.append('./CNN')
import tensorflow as tf
from gan import GAN
from utils import Utils
from data_load import Load


def main(args):
    print("---Start Learning------")
    print("Network : {}".format(FLAGS.network))
    print("data : {}".format(FLAGS.data))
    print("epoch : {}".format(FLAGS.n_epoch))
    print("batch_size : {}".format(FLAGS.batch_size))
    print("Optimizer : {}".format(FLAGS.opt))
    print("learning rate : {}".format(FLAGS.lr))
    print("-----------------------")

    # Setting
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

    model = GAN(name='GAN', lr=FLAGS.lr, opt=FLAGS.opt, trainable=True)
    dis_true, dis_fake, fake_image = model.inference(inputs, batch_size)
    #logits  = tf.identity(dis_true, dis_fake, name="output_logits")
    dis_loss, gen_loss = model.loss(dis_true, dis_fake)
    
    opt_op = model.optimize(dis_loss, gen_loss, global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group([opt_op] + update_ops)
    predict = model.predict()
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # logging for tensorboard
    util = Utils(prefix='GAN')
    util.conf_log()
    tf.summary.scalar('global_step', global_step)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('image', inputs)
    tf.summary.image('image', fake_image)

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
        "global_step": global_step,
        "loss": loss,
        "accuracy": accuracy}
    hooks.append(tf.train.LoggingTensorHook(metrics, every_n_iter=50))
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

    return

if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('network', 'GAN', 'Choice the training data name -> [GAN]')
    flags.DEFINE_string('data', 'mnist', 'Choice the training data name -> ["mnist","cifar10","cifar100"]')
    flags.DEFINE_integer('n_epoch', '1000', 'Input max epoch')
    flags.DEFINE_integer('batch_size', '32', 'Input batch size')
    flags.DEFINE_float('lr', '0.001', 'Input learning rate')
    flags.DEFINE_string('opt', 'SGD', 'Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]')
    flags.DEFINE_integer('checkpoints_to_keep', 5,'checkpoint keep count')
    flags.DEFINE_integer('keep_checkpoint_every_n_hours', 1, 'checkpoint create ')
    flags.DEFINE_integer('save_checkpoint_steps', 1000,'save checkpoint step')
    tf.app.run()