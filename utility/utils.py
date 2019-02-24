# -*- coding: utf-8 -*-
import os
import numpy as np
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt


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
        return

    def initial(self):
        self.conf_log()
        if not os.path.isdir(self.log_dir):
            tf.gfile.MakeDirs(self.log_dir)
        return

    def write_configuration(self, message):
        """
        設定をテキストに出力する

        parameters
        -------
        message : dict
        """
        with open(self.log_dir + '/log.txt', 'a') as f:
            f.write("------Learning Details------\n")
            for key, info in message.items():
                f.write("%s : %s\n"%(key, info))
            f.write("----------------------------\n")
        return 


    def write_log(self, message):
        """
        学習状況をテキストに出力する

        parameters
        -------
        message : dict
        """
        stats = []
        for key, info in message.items():
            stats.append("%s = %s" % (key, info))
        info = "%s\n"%(", ".join(stats))
        with open(self.log_dir + '/log.txt', 'a') as f:
            f.write(str(info))
        return 

    def save_init(self, model):
        self.checkpoint = tf.train.Checkpoint(policy=model)
        self.saver = tf.contrib.checkpoint.CheckpointManager(self.checkpoint,
                                                             directory=self.model_path,
                                                             max_to_keep=5)
        return

    def save_model(self, episode=None):
        if self.sess is not None:
            self.saver.save(self.sess, self.log_dir + "/model.ckpt"%episode)
        else:
            self.saver.save(checkpoint_number=tf.train.get_or_create_global_step())

    def restore_agent(self, log_dir=None):
        self.checkpoint.restore(tf.train.latest_checkpoint(log_dir))
        return


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

    def construct_figure(self, x_test, decoded_imgs, n=10):
        '''
        元の画像と生成した画像10枚ずつを保存する
        parameters
        ----------
        x_test : input test image
        decoded_imgs : generate image

        returns
        -------

        '''
        plt.figure(figsize=(20, 4))
        for i in range(n):
            #  display original
            ax = plt.subplot(2, n, i + 1)
            ax.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2, n, i + 1 + n)
            ax.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            plt.tight_layout()
        plt.savefig(self.log_dir + '/construct_figure.png')

    def reconstruct_image(self, decoded_imgs):
        """
        VAEで出力した画像の図を作成する
        """
        plt.figure(figsize=(8, 8)) 
        plt.imshow(decoded_imgs, cmap="gray")
        plt.savefig(self.log_dir + '/construct_figure.png')
