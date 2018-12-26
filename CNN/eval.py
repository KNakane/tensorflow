import os,sys
sys.path.append('./utility')
sys.path.append('./network')
import tensorflow as tf
from data_load import Load
from utils import Utils
from model import DNN
from lenet import LeNet
from train import set_model

def main(args):
    print("---------Start Evaluation--------")
    print("CheckPoint : {}".format(FLAGS.ckpt_dir))
    print("Network : {}".format(FLAGS.network))
    print("data : {}".format(FLAGS.data))
    print("---------------------------------")

    # load dataset
    data = Load(FLAGS.data)
    dataset = data.load(data.x_test, data.y_test, batch_size=1000, is_training=False)
    iterator = dataset.make_initializable_iterator()
    inputs, labels = iterator.get_next()
    inputs = tf.reshape(inputs, (-1, data.size, data.size, data.channel)) / 255.0

    model_set = set_model(data.output_dim)
    model = eval(FLAGS.network)(model=model_set, name=FLAGS.network, lr=0, opt=None, trainable=False)
    logits = model.inference(inputs)
    logits  = tf.identity(logits, name="output_logits")
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.train.Saver()
    with tf.Session() as sess:
        utils = Utils(sess=sess)
        utils.restore_model(FLAGS.ckpt_dir)
        sess.run(iterator.initializer,feed_dict={data.features_placeholder: data.x_test,
                                                 data.labels_placeholder: data.y_test})
        test_accuracy = sess.run(accuracy)
        print("accuracy : {}".format(test_accuracy))
    return

if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('network', 'LeNet', 'Choice the training data name -> [DNN,LeNet]')
    flags.DEFINE_string('ckpt_dir', './results/181225_193106/model', 'Choice the checkpoint directpry')
    flags.DEFINE_string('data', 'mnist', 'Choice the training data name -> ["mnist","cifar10","cifar100"]')
    tf.app.run()