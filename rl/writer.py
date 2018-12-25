# -*- coding: utf-8 -*-
import sys
sys.path.append('./utility')
from utils import Utils
import datetime
import tensorflow as tf

class Writer(Utils):
    def __init__(self, sess):
        super().__init__(sess=sess)
        dt_now = datetime.datetime.now()
        self.res_dir = './results/'+dt_now.strftime("%y%m%d_%H%M%S")

        if tf.gfile.Exists(self.res_dir):
            tf.gfile.DeleteRecursively(self.res_dir)
        tf.gfile.MakeDirs(self.res_dir)

        self.log_dir = self.res_dir + '/log'
        self.tf_board = self.res_dir + '/tf_board'
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(self.tf_board, sess.graph)

    def add_list(self, dic, episode, test=False):
        if test:
            record_list = [tf.Summary(value=[tf.Summary.Value(tag="test_Record/" + key, simple_value=dic[key])]) for key in dic]
        else:
            record_list = [tf.Summary(value=[tf.Summary.Value(tag="Record/" + key, simple_value=dic[key])]) for key in dic]
        [self.add(i, episode) for i in record_list]

    def add(self, list, iter):
        self.writer.add_summary(list, iter)
        self.writer.flush()
        return