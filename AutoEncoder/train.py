import sys
sys.path.append('./utility')
sys.path.append('./network')
sys.path.append('./dataset')
import tensorflow as tf
from ae import AutoEncoder, VAE, CVAE
from load import Load
from utils import Utils
from collections import OrderedDict
from hooks import SavedModelBuilderHook, MyLoggerHook, AEHook


def set_model(outdim):
    """
    encode = [['conv', 5, 16, 2, tf.nn.leaky_relu],
              ['conv', 5, 32, 2, tf.nn.leaky_relu],
              ['fc', 40, None]]

    decode = [['fc', 7*7*32, tf.nn.relu],
              ['reshape', [-1, 7, 7, 32]],
              ['deconv',  5, 16, 2, tf.nn.relu],
              ['deconv',  5, outdim, 2, None]]
    """
    encode = [['fc', 500, tf.nn.elu],
              ['fc', 100, tf.nn.elu],
              ['fc', 40, None]]

    decode = [['fc', 100, tf.nn.elu],
              ['fc', 500, tf.nn.elu],
              ['fc', 784, tf.nn.sigmoid],
              ['reshape', [-1, 28, 28, 1]]]
    return encode, decode

def main(argv):
    message = OrderedDict({
        "Network": FLAGS.network,
        "data": FLAGS.data,
        "epoch":FLAGS.n_epoch,
        "batch_size": FLAGS.batch_size,
        "Optimizer":FLAGS.opt,
        "learning_rate":FLAGS.lr,
        "Denoising":FLAGS.denoise,
        "l2_norm": FLAGS.l2_norm,
        "Augmentation": FLAGS.aug})

    # prepare training
    # setting
    checkpoints_to_keep = FLAGS.checkpoints_to_keep
    keep_checkpoint_every_n_hours = FLAGS.keep_checkpoint_every_n_hours
    max_steps = FLAGS.n_epoch
    save_checkpoint_steps = FLAGS.save_checkpoint_steps
    batch_size = FLAGS.batch_size
    restore_dir = FLAGS.init_model
    global_step = tf.train.create_global_step()


    ## load dataset
    data = Load(FLAGS.data)
    dataset = data.load(data.x_train, data.y_train, batch_size=batch_size, is_training=True)
    valid = data.load(data.x_test, data.y_test, batch_size=batch_size*3, is_training=False)
    iterator = dataset.make_initializable_iterator()
    valid_iter = valid.make_initializable_iterator()
    inputs, labels = iterator.get_next()
    valid_inputs, valid_labels = valid_iter.get_next()

    ## setting models
    encode, decode = set_model(data.channel)
    model = eval(FLAGS.network)(encode=encode, decode=decode, denoise=FLAGS.denoise, size=data.size, channel=data.channel, name=FLAGS.network, out_dim=data.output_dim, lr=FLAGS.lr, opt=FLAGS.opt, trainable=True)

    #train
    train_logits = model.inference(inputs, labels) if FLAGS.network == 'CVAE' else model.inference(inputs)
    train_loss = model.loss(train_logits, inputs)
    opt_op = model.optimize(train_loss, global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group([opt_op] + update_ops)

    #test
    test_logits = model.inference(valid_inputs, valid_labels, reuse=True) if FLAGS.network == 'CVAE' else model.inference(valid_inputs, reuse=True)
    test_loss = model.loss(test_logits, valid_inputs)
    test_image = model.predict(valid_inputs, valid_labels) if FLAGS.network == 'CVAE' else model.predict(valid_inputs)

    # logging for tensorboard
    util = Utils(prefix=FLAGS.network)
    util.conf_log()
    tf.summary.scalar('train/loss', train_loss)
    tf.summary.scalar('train/Learning_rate', model.optimizer.lr)
    tf.summary.image('train/image', inputs)
    tf.summary.image('train/encode_image', train_logits)
    tf.summary.image('test/encode_image', test_logits)
    tf.summary.scalar('test/loss', test_loss)
    tf.summary.image('test/image', valid_inputs)

    def init_fn(scaffold, session):
        session.run([iterator.initializer,valid_iter.initializer],
                            feed_dict={data.features_placeholder: data.x_train,
                                       data.labels_placeholder: data.y_train,
                                       data.valid_placeholder: data.x_test,
                                       data.valid_labels_placeholder: data.y_test})

    metrics = OrderedDict({
            "global_step": global_step,
            "train loss": train_loss,
            "test loss": test_loss})

    # create saver
    scaffold = tf.train.Scaffold(
        init_fn=init_fn,
        saver=tf.train.Saver(
            max_to_keep=checkpoints_to_keep,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours))

    tf.logging.set_verbosity(tf.logging.INFO)
    hooks = []
    hooks.append(AEHook(test_image, util.log_dir, every_n_iter=100))
    hooks.append(MyLoggerHook(message, util.log_dir, metrics, every_n_iter=100))
    hooks.append(tf.train.NanTensorHook(train_loss))
    if max_steps:
        hooks.append(tf.train.StopAtStepHook(last_step=max_steps))
     
    # training
    session = tf.train.MonitoredTrainingSession(
        checkpoint_dir=util.model_path,
        hooks=hooks,
        scaffold=scaffold,
        save_summaries_steps=1,
        save_checkpoint_steps=save_checkpoint_steps,
        summary_dir=util.tf_board)

    
    with session:
            if restore_dir is not None:
                ckpt = tf.train.get_checkpoint_state(restore_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    saver.restore(session, ckpt.model_checkpoint_path)
            while not session.should_stop():
                _, test_input, test_output = session.run([train_op, valid_inputs, test_image])
            util.construct_figure(test_input, test_output)
    return

if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('network', 'AutoEncoder', 'Choice the training data name -> [AutoEncoder,VAE]')
    flags.DEFINE_string('data', 'mnist', 'Choice the training data name -> ["mnist","cifar10","cifar100","kuzushiji"]')
    flags.DEFINE_integer('n_epoch', '1000', 'Input max epoch')
    flags.DEFINE_integer('batch_size', '32', 'Input batch size')
    flags.DEFINE_float('lr', '0.001', 'Input learning rate')
    flags.DEFINE_string('opt', 'SGD', 'Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]')
    flags.DEFINE_string('aug','None','Choice the Augmentation -> ["shift","mirror","rotate","shift_rotate","cutout","random_erace"]')
    flags.DEFINE_bool('l2_norm', 'False', 'Input learning rate')
    flags.DEFINE_bool('denoise', 'False', 'True : Denoising AE, False : standard AE')
    flags.DEFINE_string('init_model', 'None', 'Choice the checkpoint directpry(ex. ./results/181225_193106/model)')
    flags.DEFINE_integer('checkpoints_to_keep', 5,'checkpoint keep count')
    flags.DEFINE_integer('keep_checkpoint_every_n_hours', 1, 'checkpoint create ')
    flags.DEFINE_integer('save_checkpoint_steps', 1000,'save checkpoint step')
    tf.app.run()