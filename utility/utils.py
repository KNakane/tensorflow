# -*- coding: utf-8 -*-
import datetime
import tensorflow as tf


class Utils():
    def __init__(self, sess=None, prefix=None):
        dt_now = datetime.datetime.now()
        self.sess = sess
        self.res_dir = "results/"+dt_now.strftime("%y%m%d_%H%M%S")
        if prefix is not None:
            self.res_dir = self.res_dir + "_{}".format(prefix)
        self.log_dir = self.res_dir + "/log"
        self.tf_board = self.res_dir + "/tf_board"
        self.model_path = self.res_dir + "/model"
        self.saved_model_path = self.model_path + "/saved_model"

    def conf_log(self):
        if tf.gfile.Exists(self.res_dir):
            tf.gfile.DeleteRecursively(self.res_dir)
        tf.gfile.MakeDirs(self.res_dir)

    def initial(self):
        self.conf_log()

    def save_init(self, model):
        self.checkpoint = tf.train.Checkpoint(policy=model)
        self.saver = tf.contrib.checkpoint.CheckpointManager(self.checkpoint,
                                                             directory=self.model_path,
                                                             max_to_keep=5)

    def save_model(self, episode=None):
        if self.sess is not None:
            self.saver.save(self.sess, self.log_dir + "/model.ckpt"%episode)
        else:
            self.saver.save(checkpoint_number=tf.train.get_or_create_global_step())

    def restore_agent(self, log_dir=None):
        self.checkpoint.restore(tf.train.latest_checkpoint(log_dir))


    def restore_model(self, log_dir=None):
        assert log_dir is not None, 'Please set log_dir to restore checkpoint'
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver = tf.train.import_meta_graph("{}.meta".format(ckpt.model_checkpoint_path))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Restore : {}.meta".format(ckpt.model_checkpoint_path))
            return True
        else:
            print ('Not Restore model in "{}"'.format(log_dir))
            return False

    def saved_model(self, x, y):
        '''
        x : Placeholder input
        y : Placeholder label or correct data
        '''
        builder = tf.saved_model.builder.SavedModelBuilder(self.model_path)
        signature = tf.saved_model.predict_signature_def(inputs={'inputs':x}, outputs={'label':y})
        builder.add_meta_graph_and_variables(sess=self.sess,
                                             tags=[tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={'predict': signature})
        builder.save()
        return