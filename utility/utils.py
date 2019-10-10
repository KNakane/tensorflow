# -*- coding: utf-8 -*-
import os, re
import numpy as np
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.python.client import device_lib


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
        if tf.io.gfile.exists(self.res_dir):
            tf.io.gfile.remove(self.res_dir)
        tf.io.gfile.makedirs(self.res_dir)
        return

    def initial(self):
        self.conf_log()
        if not os.path.isdir(self.log_dir):
            tf.io.gfile.makedirs(self.log_dir)
        return

    def write_configuration(self, message, _print=False):
        """
        設定をテキストに出力する

        parameters
        -------
        message : dict

        _print : True / False : terminalに表示するか
        """
        with open(self.log_dir + '/log.txt', 'a') as f:
            f.write("------Learning Details------\n")
            if _print:
                print("------Learning Details------")
            for key, info in message.items():
                f.write("%s : %s\n"%(key, info))
                if _print:
                    print("%s : %s"%(key, info))
            f.write("----------------------------\n")
            print("----------------------------")
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
        self.saver =  tf.train.CheckpointManager(self.checkpoint,
                                                 directory=self.model_path,
                                                 max_to_keep=5)
        return

    def save_model(self, global_step=None):
        if self.sess is not None:
            self.saver.save(self.sess, self.log_dir + "/model.ckpt"%global_step)
        else:
            self.saver.save(checkpoint_number=global_step)

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
            try:
                ax.imshow(x_test[i].reshape(28, 28))
            except:
                ax.imshow(x_test[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2, n, i + 1 + n)
            try:
                ax.imshow(decoded_imgs[i].reshape(28, 28))
            except:
                ax.imshow(decoded_imgs[i])
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
        """
        GANで生成した画像の図を作成する
        """
        fig = plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(6, 6)
        gs.update(wspace=0.05, hspace=0.05)

        for i in range(36):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            try:
                plt.imshow(samples[i].reshape(28, 28), cmap='Greys_r')
            except:
                plt.imshow(np.clip(samples[i],0,255))

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
        pos = np.sum(pi * probs, axis=1)
        levels_log = np.linspace(0, np.log(pos.max()), 21)
        levels = np.exp(levels_log)
        levels[0] = 0
        #x_test, y_test = np.meshgrid(x, y)
        plt.scatter(x, y, alpha=0.5, label="observation", color='b')
        #plt.fill_between(x[:,0], mu - probs, mu + probs,facecolor='r',alpha=0.3)
        plt.scatter(x, y_test, alpha=0.5, label="output", color='r')
        #plt.contourf(x_test, y_test, probs, levels, alpha=0.5)
        plt.xlim(x.min(), x.max())
        plt.ylim(y.min(), y.max())
        plt.legend()
        plt.savefig(self.log_dir + '/MDN_figure.png')

        plt.close()

        def sample_from_mixture(x, pred_weights, pred_means, pred_std, amount):
            from scipy.stats import norm as normal
            samples = np.zeros((amount, 2))
            n_mix = len(pred_weights[0])
            to_choose_from = np.arange(n_mix)
            for j,(weights, means, std_devs) in enumerate(zip(pred_weights, pred_means, pred_std)):
                index = np.random.choice(to_choose_from, p=weights)
                samples[j,1]= normal.rvs(means[index], std_devs[index], size=1)
                samples[j,0]= x[j]
                if j == amount -1:
                    break
            return samples

        a = sample_from_mixture(x, pi, mu, sigma, amount=len(x))

        fig = plt.figure(figsize=(8, 8))
        #H = plt.hist2d(a[:,0],a[:,1], bins=40)
        plt.contourf(a[:,0], a[:,1], pos, levels, alpha=0.5)
        plt.savefig(self.log_dir + '/MDN_confidence_figure.png')
        plt.close()
        
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

def set_output_dim(FLAGS, out_dim, N_atoms=51):
    """
    Networkの出力ユニット数を決める関数
    """
    if hasattr(FLAGS, 'network') and FLAGS.network == 'Dueling_Net':
        return (out_dim + 1) * N_atoms if FLAGS.category else out_dim + 1
    elif FLAGS.agent == 'DDPG' or FLAGS.agent == 'TD3':
        return out_dim
    elif FLAGS.agent == 'A3C':
        return out_dim + 1
    else:
        return out_dim * N_atoms if FLAGS.category else out_dim

def find_gpu():
    device_list = device_lib.list_local_devices()
    for device in device_list:
        if re.match('/device:GPU', device.name):
            return 0
    return -1