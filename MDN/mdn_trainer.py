import os, sys
sys.path.append('./utility')
import tensorflow as tf
from utils import Utils
from collections import OrderedDict
from hooks import MyLoggerHook

class Train():
    def __init__(self,
                 FLAGS,
                 message,
                 data,
                 model,
                 name):
        self.checkpoints_to_keep = FLAGS.checkpoints_to_keep
        self.keep_checkpoint_every_n_hours = FLAGS.keep_checkpoint_every_n_hours
        self.max_steps = FLAGS.n_epoch
        self.save_checkpoint_steps = self.max_steps / 10 if FLAGS.save_checkpoint_steps is None else FLAGS.save_checkpoint_steps
        self.batch_size = FLAGS.batch_size
        self.name = name
        self.message = message
        self.data = data
        self.global_step = tf.train.get_or_create_global_step()
        self.model = model
        self.restore_dir = FLAGS.init_model
        self.util = Utils(prefix=self.name)

    def load(self):
        # Load Dataset
        dataset = self.data.load(self.data.x_train, self.data.y_train, batch_size=self.batch_size, is_training=True)
        valid = self.data.load(self.data.x_test, self.data.y_test, batch_size=self.batch_size*3, is_training=False)
        self.iterator = dataset.make_initializable_iterator()
        self.valid_iter = valid.make_initializable_iterator()
        inputs, labels = self.iterator.get_next()
        valid_inputs, valid_labels = self.valid_iter.get_next()
        return inputs, labels, valid_inputs, valid_labels

    def build_logits(self, train_data, train_ans, valid_data, valid_ans):
        # train
        self.train_logits = self.model.inference(train_data)
        self.train_loss = self.model.loss(self.train_logits, train_ans)
        #self.predict = self.model.predict(train_data)
        opt_op = self.model.optimize(self.train_loss, self.global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op = tf.group([opt_op] + update_ops)
        #self.train_accuracy = self.model.evaluate(self.train_logits, train_ans)

        # test
        self.test_logits = self.model.test_inference(valid_data, reuse=True)
        self.test_loss = self.model.loss(self.test_logits, valid_ans)
        #self.test_accuracy = self.model.evaluate(self.test_logits, valid_ans)

        return 

    def summary(self):
        # tensorboard
        tf.summary.scalar('train/loss', self.train_loss)
        #tf.summary.scalar('train/accuracy', self.train_accuracy)
        tf.summary.scalar('train/Learning_rate', self.model.optimizer.lr)
        tf.summary.scalar('test/loss', self.test_loss)
        #tf.summary.scalar('test/accuracy', self.test_accuracy)
        return

    def hook_append(self, metrics, signature_def_map=None):
        """
        hooksをまとめる関数
        """
        hooks = []
        hooks.append(tf.train.NanTensorHook(self.train_loss))
        hooks.append(MyLoggerHook(self.message, self.util.log_dir, metrics, every_n_iter=100))
        return hooks

    def train(self):
        self.util.conf_log()
        inputs, corrects, valid_inputs, valid_labels = self.load()
        self.build_logits(inputs, corrects, valid_inputs, valid_labels)
        self.summary()

        def init_fn(scaffold, session):
            session.run([self.iterator.initializer,self.valid_iter.initializer],
                        feed_dict={self.data.features_placeholder: self.data.x_train,
                                   self.data.labels_placeholder: self.data.y_train,
                                   self.data.valid_placeholder: self.data.x_test,
                                   self.data.valid_labels_placeholder: self.data.y_test})

        # create saver
        saver = tf.train.Saver(
                max_to_keep=self.checkpoints_to_keep,
                keep_checkpoint_every_n_hours=self.keep_checkpoint_every_n_hours)

        scaffold = tf.train.Scaffold(
            init_fn=init_fn,
            saver=saver)

        tf.logging.set_verbosity(tf.logging.INFO)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

        metrics = OrderedDict({
            "global_step": self.global_step,
            "train loss": self.train_loss,
            #"train accuracy":self.train_accuracy,
            "test loss": self.test_loss})
            #"test accuracy":self.test_accuracy})

        hooks = self.hook_append(metrics)

        session = tf.train.MonitoredTrainingSession(
            config=config,
            checkpoint_dir=self.util.model_path,
            hooks=hooks,
            scaffold=scaffold,
            save_summaries_steps=1,
            save_checkpoint_steps=self.save_checkpoint_steps,
            summary_dir=self.util.tf_board)

        with session:
            for _ in range(self.max_steps):
                session.run([self.train_op])
            output, test_input, targets = session.run([self.test_logits, valid_inputs, valid_labels])
        self.util.MDN_figure(test_input, targets, output)

        