import sys
sys.path.append('./utility')
sys.path.append('./network')
sys.path.append('./dataset')
import tensorflow as tf
from mdn_load import MDN_Load
from mdn import MDN
from mdn_trainer import Train
from collections import OrderedDict

def set_model(outdim):
    model_set = None
    return model_set

def main(argv):
    message = OrderedDict({
        "data": FLAGS.data,
        "epoch":FLAGS.n_epoch,
        "batch_size": FLAGS.batch_size,
        "Optimizer":FLAGS.opt,
        "learning_rate":FLAGS.lr,
        "l2_norm": FLAGS.l2_norm})

    # prepare training
    ## load dataset
    data = MDN_Load(FLAGS.data)
    ## setting models
    model_set = set_model(data.output_dim)
    model = MDN(model=model_set, name='MDN', out_dim=data.output_dim, lr=FLAGS.lr, opt=FLAGS.opt, l2_reg=FLAGS.l2_norm, trainable=True)

    #training
    trainer = Train(FLAGS, message, data, model, 'MDN')
    trainer.train()


if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('data', 'sample', 'Choice the training data name -> [sample]')
    flags.DEFINE_integer('n_epoch', '1000', 'Input max epoch')
    flags.DEFINE_integer('batch_size', '32', 'Input batch size')
    flags.DEFINE_float('lr', '0.001', 'Input learning rate')
    flags.DEFINE_string('opt', 'SGD', 'Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]')
    flags.DEFINE_bool('l2_norm', 'False', 'Input learning rate')
    flags.DEFINE_string('init_model', 'None', 'Choice the checkpoint directpry(ex. ./results/181225_193106/model)')
    flags.DEFINE_integer('checkpoints_to_keep', 5,'checkpoint keep count')
    flags.DEFINE_integer('keep_checkpoint_every_n_hours', 1, 'checkpoint create ')
    flags.DEFINE_integer('save_checkpoint_steps', 100,'save checkpoint step')
    tf.app.run()