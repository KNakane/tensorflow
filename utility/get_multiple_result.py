import os, sys
import re
import glob
import plotly
import datetime
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from operator import itemgetter
from itertools import chain, groupby
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MultipleEventGetter():
    def __init__(self, csv_list):
        dt_now = datetime.datetime.now()
        self.csv_list = csv_list
        self._all_csv_list = self._get_dict
        self.colorlist = ["#ff0000", "#008000", "#4169e1", "#00ffff", "#ff00ff", "#a52a2a", "#696969", "#00008b"]
        
        self.log_dir = "results/" + dt_now.strftime("%y%m%d_%H%M%S") + "_probs"
        
        self._init_log()
        

    def _init_log(self):
        """ 結果を格納するdirectoryの作成 """
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

    @property
    def _get_dict(self):
        all_csv_dict = []
        for csv_name in self.csv_list:
            split_csv_name = csv_name.split('/')
            csv_dict = dict(date=csv_name,
                        event=split_csv_name[2].split('.')[0])
            all_csv_dict.append(csv_dict)
        return all_csv_dict


    def __call__(self):
        self._all_csv_list.sort(key=lambda m: m['event'])
        for _, group in groupby(self._all_csv_list, key=lambda m: m['event']):
            all_df = {}
            for each_dict in (list(group)):
                filename = each_dict['date']
                date = filename.split('/')[1]
                event = each_dict['event']
                all_df[date] = pd.read_csv(filename, index_col=0)
            self.plot_graph(event, all_df)
            #self.plotly_graph(event, all_df)
        return

    def plot_graph(self, name, result_dict):
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1, 1, 1)
        for i, (key, df) in enumerate(result_dict.items()):
            mean = df['mean']
            variance = df['variance']
            index = range(mean.shape[0])
            plt.plot(index, mean, linestyle='solid', color=self.colorlist[i], alpha=0.8, label=key)
            plt.fill_between(index ,mean - variance, mean + variance, facecolor=self.colorlist[i], alpha=0.3)

        plt.grid(which='major',color='gray',linestyle='-')
        plt.xlabel("epoch")
        plt.ylabel(name)
        plt.legend()
        plt.savefig(self.log_dir + '/{}_prob.png'.format(name))
        plt.close()
        return

    def plotly_graph(self, name, result_dict):
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1, 1, 1)
        for i, (key, df) in enumerate(result_dict.items()):
            mean = df['mean']
            variance = df['variance']
            index = range(mean.shape[0])
            plt.plot(index, mean, linestyle='solid', color=self.colorlist[i], label=key)
            plt.fill_between(index ,mean - variance, mean + variance, facecolor=self.colorlist[i], alpha=0.3)

        plt.grid(which='major',color='gray',linestyle='-')
        plt.xlabel("epoch")
        plt.ylabel(name)
        plt.legend()
        plt.savefig(self.log_dir + '/{}_prob.png'.format(name))
        plt.close()
        return
        


def main(args):
    assert args.dir is not None
    csv_list = list(chain.from_iterable([glob.glob(res_dir+"*_prob.csv", recursive=True) for res_dir in args.dir]))
    meg = MultipleEventGetter(csv_list)
    meg()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',nargs='*', help='tensorboard event directory')
    parser.add_argument('--prob', action='store_true', help='Probability distribution graph')
    parser.add_argument('--regression', action='store_true', help='Whether Regression task or not')
    args = parser.parse_args()
    main(args)
