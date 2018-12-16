# -*- coding: utf-8 -*-
import datetime
import tensorflow as tf


class Utils():
    def __init__(self):
        dt_now = datetime.datetime.now()
        self.res_dir = "./results/"+dt_now.strftime("%y%m%d_%H%M%S")
        self.log_dir = self.res_dir + "/log"
        self.tf_board = self.res_dir + "/tf_board"

    def conf_log(self):
        if tf.gfile.Exists(self.res_dir):
            tf.gfile.DeleteRecursively(self.res_dir)
        tf.gfile.MakeDirs(self.res_dir)

    def initial(self):
        self.conf_log()
