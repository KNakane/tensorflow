import os,sys
sys.path.append('./CNN')
sys.path.append('./dataset')
import tensorflow as tf
from gan import GAN, DCGAN, WGAN, WGAN_GP, LSGAN
from utils import Utils
from load import Load
from collections import OrderedDict
from hooks import SavedModelBuilderHook, MyLoggerHook, GanHook


def main(args):
    message = OrderedDict({
        "Network": FLAGS.network,
        "Conditional": FLAGS.conditional,
        "data": FLAGS.data,
        "z_dim": FLAGS.z_dim,
        "epoch":FLAGS.n_epoch,
        "batch_size": FLAGS.batch_size,
        "Optimizer":FLAGS.opt,
        "learning_rate":FLAGS.lr,
        "Augmentation": FLAGS.aug})

    # Setting
    checkpoints_to_keep = FLAGS.checkpoints_to_keep
    keep_checkpoint_every_n_hours = FLAGS.keep_checkpoint_every_n_hours
    max_steps = FLAGS.n_epoch
    save_checkpoint_steps = FLAGS.save_checkpoint_steps
    batch_size = FLAGS.batch_size
    n_disc_update = FLAGS.n_disc_update
    restore_dir = FLAGS.init_model
    global_step = tf.train.create_global_step()

    # load dataset
    data = Load(FLAGS.data)
    dataset = data.load(data.x_train, data.y_train, batch_size=batch_size, is_training=True)
    iterator = dataset.make_initializable_iterator()
    inputs, labels = iterator.get_next()
    test_inputs = tf.random.uniform([batch_size*3, FLAGS.z_dim],-1,+1)

    model = eval(FLAGS.network)(z_dim=FLAGS.z_dim,
                                size=data.size,
                                channel=data.channel,
                                lr=FLAGS.lr,
                                class_num=data.output_dim,
                                conditional=FLAGS.conditional,
                                opt=FLAGS.opt,
                                trainable=True)

    if FLAGS.network == 'CGAN':
        D_logits, D_logits_ = model.inference(inputs, batch_size, labels)
    elif FLAGS.network == 'WGAN' or FLAGS.network == 'WGAN-GP':
        D_logits, D_logits_ = model.inference(inputs, batch_size, labels)
        n_disc_update = 5
    else:
        D_logits, D_logits_ = model.inference(inputs, batch_size, labels)

    if FLAGS.network == 'ACGAN':
        dis_loss, gen_loss = model.loss(D_logits, D_logits_, labels)
    else:
        dis_loss, gen_loss = model.loss(D_logits, D_logits_)
    
    d_op, g_op = model.optimize(d_loss=dis_loss, g_loss=gen_loss, global_step=global_step)
    train_accuracy = model.evaluate(D_logits, D_logits_)
    G = model.predict(test_inputs, batch_size*3)

    # logging for tensorboard
    util = Utils(prefix=FLAGS.network)
    util.conf_log()
    tf.summary.scalar('global_step', global_step)
    tf.summary.scalar('discriminator_loss', dis_loss)
    tf.summary.scalar('generator_loss', gen_loss)
    tf.summary.histogram('generator_output', G)
    tf.summary.histogram('True_image', inputs)
    tf.summary.image('image', inputs)
    tf.summary.image('fake_image', G)

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
        "discriminator_loss": dis_loss,
        "generator_loss": gen_loss,
        "train_accuracy": train_accuracy}

    hooks.append(MyLoggerHook(message, util.log_dir, metrics, every_n_iter=100))
    hooks.append(GanHook(G, util.log_dir, every_n_iter=1000))
    hooks.append(tf.train.NanTensorHook(dis_loss))
    hooks.append(tf.train.NanTensorHook(gen_loss))

    # training
    session = tf.train.MonitoredTrainingSession(
        checkpoint_dir=util.model_path,
        hooks=hooks,
        scaffold=scaffold,
        save_summaries_steps=1,
        save_checkpoint_steps=save_checkpoint_steps,
        summary_dir=util.tf_board)

    def _train(session, d_op, g_op, n_disc_update):
        for _ in range(max_steps):
            for _ in range(n_disc_update):
                c_op_vals = session.run(d_op)

            g_op_vals = session.run(g_op)
        return c_op_vals, g_op_vals


    with session:
        if restore_dir is not None:
            ckpt = tf.train.get_checkpoint_state(restore_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(session, ckpt.model_checkpoint_path)
        _train(session, d_op, g_op, n_disc_update)
    return

if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('network', 'GAN', 'Choice the training data name -> [GAN, DCGAN, WGAN, WGAN_GP, CGAN]')
    flags.DEFINE_string('data', 'mnist', 'Choice the training data name -> ["mnist","cifar10","cifar100", "kuzushiji"]')
    flags.DEFINE_integer('n_epoch', '50000', 'Input max epoch')
    flags.DEFINE_integer('batch_size', '32', 'Input batch size')
    flags.DEFINE_float('lr', '2e-4', 'Input learning rate')
    flags.DEFINE_string('opt', 'SGD', 'Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]')
    flags.DEFINE_string('aug','None','Choice the Augmentation -> ["shift","mirror","rotate","shift_rotate","cutout","random_erace"]')
    flags.DEFINE_bool('l2_norm', 'False', 'Input learning rate')
    flags.DEFINE_integer('z_dim', '100', 'Latent z dimension')
    flags.DEFINE_bool('conditional', 'False', 'Conditional true or false')
    flags.DEFINE_integer('n_disc_update', '2', 'Input max epoch')
    flags.DEFINE_string('init_model', 'None', 'Choice the checkpoint directpry(ex. ./results/181225_193106/model)')
    flags.DEFINE_integer('checkpoints_to_keep', 5,'checkpoint keep count')
    flags.DEFINE_integer('keep_checkpoint_every_n_hours', 1, 'checkpoint create ')
    flags.DEFINE_integer('save_checkpoint_steps', 1000,'save checkpoint step')
    tf.app.run()