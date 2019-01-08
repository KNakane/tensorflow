import os,sys
sys.path.append('./utility')
sys.path.append('./network')
sys.path.append('./dataset')
import tensorflow as tf
from load import Load
from utils import Utils
from cnn import CNN
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
    inputs, labels = data.load_test(data.x_test, data.y_test)

    model_set = set_model(data.output_dim)
    model = eval(FLAGS.network)(model=model_set, name=FLAGS.network, out_dim=data.output_dim, lr=0, opt=None, trainable=False)
    logits = model.inference(inputs)
    logits  = tf.identity(logits, name="output_logits")
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.train.Saver()
    with tf.Session() as sess:
        utils = Utils(sess=sess)
        if utils.restore_model(FLAGS.ckpt_dir):
            test_accuracy = sess.run(accuracy)
            print("accuracy : {}".format(test_accuracy))
            return
        else:
            return

if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('network', 'LeNet', 'Choice the training data name -> [DNN,LeNet]')
    flags.DEFINE_string('ckpt_dir', './results/181225_193106/model', 'Choice the checkpoint directpry')
    flags.DEFINE_string('data', 'mnist', 'Choice the training data name -> ["mnist","cifar10","cifar100","kuzushiji"]')
    tf.app.run()