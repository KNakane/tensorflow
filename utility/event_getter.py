import os, sys
import re
import glob
import datetime
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import chain
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

class EventGetter():
    """ tensorboardに表示したデータを取得し、グラフを作成する """
    def __init__(self, events_list, prob):
        dt_now = datetime.datetime.now()
        self.events_list = events_list
        self._prob = prob               # 確率分布のグラフを作成するフラグ
        self.log_dir = "results/" + dt_now.strftime("%y%m%d_%H%M%S") + "_events"
        self.result_dic = {}
        self._init_log()

    def _init_log(self):
        """ 結果を格納するdirectoryの作成 """
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

    def __call__(self):
        # eventごとにデータを取得し、dictに格納する
        self.gather_result()
        # 格納したdictから各項目ごとにグラフを作成する
        num = 0
        for key in self.result_dic:
            self.make_graph(key, self.result_dic[key])
            self.make_graph_moving_avg(key, self.result_dic[key])
            if self._prob:
                self.make_graph_prob(key, self.result_dic[key], num)
                num += 1
        self.logger()
        return
    
    def logger(self):
        """ どのファイルのeventを読み込んでグラフを作成したかテキストファイルに書き出す"""
        str_ = '\n'.join(self.events_list)
        with open(self.log_dir + "/filelist.txt", 'wt') as f:
            f.write(str_)
        return

    def gather_result(self):
        """
        directoryの中身を取得、項目ごとにdictに格納
        """
        for event in self.events_list:
            # 名前の取得
            name = event.split('/')[1]
            self.get_scores(name, event)
        return
        
    def get_scores(self, name, path_file):
        """指定されたファイルからスコアの系列を取得して返す

         parameters
        ----------
        name : result directory

        path_file : event path

        returns
        ----------
        
        """
        accumulator = EventAccumulator(path_file)
        accumulator.Reload()

        tag_dict = accumulator.Tags() #'images', 'audio', 'histograms', 'scalars', 'distributions', 'tensors', 'graph', 'meta_graph', 'run_metadata'
        scalars_key = tag_dict['scalars']
        assert scalars_key is not None

        for key in scalars_key:
            key_name = key.replace('/','_')
            if not key_name in self.result_dic:
                self.result_dic.setdefault(key_name,{})
            if not name in self.result_dic[key_name]:
                self.result_dic.setdefault(name,{})
            value = accumulator.Scalars(key)
            self.result_dic[key_name][name] = np.array([tmp.value for tmp in value])
        return 

    def make_graph(self, name, values):
        """
        項目ごとにグラフを作成する

         parameters
        ----------
        name : result directory

        values : dict

        returns
        ----------
        """
        fig = plt.figure(figsize=(10,5))
        colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]
        ax = fig.add_subplot(1, 1, 1)
        i = 0
        if not len(values.keys()):
            return 
        for key in values:
            array = values[key]
            plt.plot(range(array.shape[0]), array, linestyle='solid', color=colorlist[i], label=key, alpha=0.6)
            i += 1
        if re.search('accuracy', name):
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        plt.grid(which='major',color='gray',linestyle='-')
        plt.xlabel("epoch")
        plt.ylabel(name)
        plt.legend()
        plt.savefig(self.log_dir + '/{}.png'.format(name))
        plt.close()
        return

    def make_graph_moving_avg(self, name, values, rate=3):
        """
        項目ごとに移動平均したグラフを作成する

         parameters
        ----------
        name : result directory

        values : dict

        returns
        ----------
        """
        fig = plt.figure(figsize=(10,5))
        colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]
        ax = fig.add_subplot(1, 1, 1)
        i = 0
        if not len(values.keys()):
            return 
        for key in values:
            array = np.convolve(values[key], np.ones(rate)/float(rate), 'valid')
            plt.plot(range(array.shape[0]), array, linestyle='solid', color=colorlist[i], label=key, alpha=0.6)
            i += 1
        if re.search('accuracy', name):
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        plt.grid(which='major',color='gray',linestyle='-')
        plt.xlabel("epoch")
        plt.ylabel(name)
        plt.legend()
        plt.savefig(self.log_dir + '/{}_moving_avg.png'.format(name))
        plt.close()
        return

    def make_graph_prob(self, name, values, num):
        """
        項目ごとに確率分布のグラフを作成する

         parameters
        ----------
        name : result directory

        values : dict

        returns
        ----------
        """
        fig = plt.figure(figsize=(10,5))
        colorlist = ["r", "g", "b", "c", "m", "brown", "grey", "darkblue"]
        ax = fig.add_subplot(1, 1, 1)
        key_num = len(values.keys())
        if not key_num:
            return
        key = list(values)
        all_results = np.zeros((key_num, len(values[key[0]])))
        for i in range(key_num):
            all_results[i] = values[key[i]]
        mean = np.mean(all_results, axis=0)
        std = np.std(all_results, axis=0)
        plt.plot(range(mean.shape[0]), mean, linestyle='solid', color=colorlist[num])
        
        if re.search('accuracy', name):
            plt.fill_between(range(mean.shape[0]) ,np.clip(mean - std,0,1), np.clip(mean + std,0,1),facecolor=colorlist[num],alpha=0.3)
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        else:
            plt.fill_between(range(mean.shape[0]) ,mean - std, mean + std, facecolor=colorlist[num],alpha=0.3)
        plt.grid(which='major',color='gray',linestyle='-')
        plt.xlabel("epoch")
        plt.ylabel(name)
        plt.savefig(self.log_dir + '/{}_prob.png'.format(name))
        plt.close()
        return


def main(args):
    assert args.dir is not None
    events_list = list(chain.from_iterable([glob.glob(res_dir+"tf_board/events.*", recursive=True) for res_dir in args.dir]))
    evget = EventGetter(events_list, args.prob)
    evget()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',nargs='*', help='tensorboard event directory')
    parser.add_argument('--prob', action='store_true', help='Probability distribution graph')
    args = parser.parse_args()
    main(args)