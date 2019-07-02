import os,sys
sys.path.append('./utility')
sys.path.append('./network')
sys.path.append('./dataset')
import tensorflow as tf
from load import Load
from utils import Utils
from gan import GAN, DCGAN, WGAN, WGAN_GP, LSGAN, ACGAN, DRAGAN
from train import set_model


def main(args):
    print("------------Start Evaluation-----------")
    print("CheckPoint : {}".format(FLAGS.ckpt_dir))
    print("Network : {}".format(FLAGS.network))
    print("data : {}".format(FLAGS.data))
    print("---------------------------------------")

    # load dataset
    data = Load(FLAGS.data)
    batch_size = 3000
    iteration = data.x_test.shape[0] // batch_size
    test = data.load_test(data.x_test, data.y_test, batch_size)
    test_iter = test.make_initializable_iterator()
    inputs, labels = test_iter.get_next()

    model_set = set_model(data.output_dim)
    model = eval(FLAGS.network)(z_dim=FLAGS.z_dim,
                                size=data.size,
                                channel=data.channel,
                                lr=0.0,
                                class_num=data.output_dim,
                                conditional=FLAGS.conditional,
                                opt=FLAGS.opt,
                                trainable=False)

    logits = model.inference(inputs)
    logits  = tf.identity(logits, name="output_logits")
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.train.Saver()
    with tf.Session() as sess:
        sess.run([test_iter.initializer],feed_dict={data.test_placeholder: data.x_test,
                                        data.test_labels_placeholder: data.y_test})
        utils = Utils(sess=sess)
        if utils.restore_model(FLAGS.ckpt_dir):
            avg_accuracy = 0
            for i in range(iteration):
                test_accuracy = sess.run(accuracy)
                avg_accuracy += test_accuracy
                print("accuracy_{} : {}".format(i, test_accuracy))
            print("average_accuracy : {}".format(avg_accuracy / iteration))
            return
        else:
            return

if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('network', 'GAN', 'Choice the training data name -> [GAN, DCGAN, WGAN, WGAN_GP, LSGAN, ACGAN, DRAGAN]')
    flags.DEFINE_string('ckpt_dir', './results/181225_193106/model', 'Choice the checkpoint directpry')
    flags.DEFINE_string('data', 'mnist', 'Choice the training data name -> ["mnist","cifar10","cifar100","kuzushiji"]')
    flags.DEFINE_bool('conditional', 'False', 'Conditional true or false')
    tf.app.run()