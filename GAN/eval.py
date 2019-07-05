import os,sys
sys.path.append('./utility')
sys.path.append('./network')
sys.path.append('./dataset')
import tensorflow as tf
from load import Load
from utils import Utils
from gan import GAN, DCGAN, WGAN, WGAN_GP, LSGAN, ACGAN, DRAGAN


def tile_index(batch_size):
    index = [9, 23, 28, 32, 39, 42]
    for _ in range(49):
        index.extend([9, 23, 28, 32, 39, 42])
    return index

def main(args):
    print("------------Start Evaluation-----------")
    print("CheckPoint : {}".format(FLAGS.ckpt_dir))
    print("Network : {}".format(FLAGS.network))
    print("data : {}".format(FLAGS.data))
    print("---------------------------------------")

    # load dataset
    data = Load(FLAGS.data)
    batch_size = 100
    dataset = data.load(data.x_train, data.y_train, batch_size=batch_size, is_training=True)
    iterator = dataset.make_initializable_iterator()
    inputs, labels = iterator.get_next()
    test_inputs = tf.random.uniform([batch_size*3, FLAGS.z_dim],-1,+1)
    index = tile_index(batch_size*3)

    model = eval(FLAGS.network)(z_dim=FLAGS.z_dim,
                                size=data.size,
                                channel=data.channel,
                                lr=0.0,
                                class_num=data.output_dim,
                                conditional=FLAGS.conditional,
                                opt=None,
                                trainable=False)
    
    
    D_logits, D_logits_ = model.inference(inputs, batch_size, labels)
    G = model.predict(test_inputs, batch_size*3, index)

    tf.train.Saver()
    with tf.Session() as sess:
        utils = Utils(sess=sess)
        utils.initial()
        if utils.restore_model(FLAGS.ckpt_dir):
            image = sess.run(G)
            utils.gan_plot(image)
            return
        else:
            return

if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('z_dim', '100', 'Latent z dimension')
    flags.DEFINE_string('network', 'GAN', 'Choice the training data name -> [GAN, DCGAN, WGAN, WGAN_GP, LSGAN, ACGAN, DRAGAN]')
    flags.DEFINE_string('ckpt_dir', './results/181225_193106/model', 'Choice the checkpoint directpry')
    flags.DEFINE_string('data', 'mnist', 'Choice the training data name -> ["mnist","cifar10","cifar100","kuzushiji"]')
    flags.DEFINE_bool('conditional', 'False', 'Conditional true or false')
    tf.app.run()