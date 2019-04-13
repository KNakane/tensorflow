# -*- coding: utf-8 -*-
import os
import numpy as np
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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


    def write_log(self, message, test=False):
        """
        学習状況をテキストに出力する

        parameters
        -------
        message : dict

        test : bool
        """
        stats = []
        for key, info in message.items():
            stats.append("%s = %s" % (key, info))
        info = "%s\n"%(", ".join(stats))
        if test:
            with open(self.log_dir + '/test_log.txt', 'a') as f:
                f.write(str(info))
        else:
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

    def restore_agent(self, model, log_dir=None):
        self.checkpoint = tf.train.Checkpoint(policy=model)
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

    def gan_plot(self, samples):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        i = 0
        while(True):
            name = self.log_dir + '/{}.png'.format(str(i).zfill(3))
            if os.path.isfile(name):
                i += 1
            else:
                plt.savefig(name, bbox_inches='tight')
                break

        plt.close(fig)

        return 

    def MDN_figure(self, x, y, y_test):
        [pi, sigma, mu] = y_test
        y_test = self.generate_ensemble(pi, mu, sigma)
        probs = np.exp(-0.5 * (mu - y) ** 2 / np.square(sigma)) / np.sqrt(2 * np.pi * np.square(sigma))
        probs = np.sum(pi * probs, axis=1)
        levels_log = np.linspace(0, np.log(probs.max()), 21)
        levels = np.exp(levels_log)
        levels[0] = 0
        #x_test, y_test = np.meshgrid(x, y)
        plt.scatter(x, y, alpha=0.5, label="observation", color='b')
        plt.scatter(x, y_test, alpha=0.5, label="output", color='r')
        #plt.contourf(x_test, y_test, probs, levels, alpha=0.5)
        plt.xlim(x.min(), x.max())
        plt.ylim(y.min(), y.max())
        plt.legend()
        plt.savefig(self.log_dir + '/MDN_figure.png')

        return

    def get_pi_idx(self, x, pdf):
        N = pdf.size
        accumulate = 0
        for i in range(0, N):
            accumulate += pdf[i]
            if (accumulate >= x):
                return i
        print ('error with sampling ensemble')
        return -1

    def generate_ensemble(self,out_pi, out_mu, out_sigma, M = 1):
        NTEST = out_pi.shape[0]
        result = np.random.rand(NTEST, M) # initially random [0, 1]
        rn = np.random.randn(NTEST, M) # normal random matrix (0.0, 1.0)
        mu = 0
        std = 0
        idx = 0

        # transforms result into random ensembles
        for j in range(0, M):
            for i in range(0, NTEST):
                idx = self.get_pi_idx(result[i, j], out_pi[i])
                mu = out_mu[i, idx]
                std = out_sigma[i, idx]
                result[i, j] = mu + rn[i, j]*std
        return result

def set_output_dim(network, category, out_dim, N_atoms=51):
    """
    Networkの出力ユニット数を決める関数
    """
    if network == 'Dueling_Net':
        return (out_dim + 1) * N_atoms if category else out_dim + 1
    elif network == 'DDPG' or network == 'TD3':
        return out_dim
    else:
        return out_dim * N_atoms if category else out_dim