import os, sys
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
    def __init__(self, events_list):
        dt_now = datetime.datetime.now()
        self.events_list = events_list
        self.log_dir = "results/" + dt_now.strftime("%y%m%d_%H%M%S") + "_events"
        self.name_list = []
        #self._init_log()

    def _init_log(self):
        """ 結果を格納するdirectoryの作成 """
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

    def __call__(self):
        res_list = self.gather_result()
        for key in res_list[0].keys():
            for i in range(len(res_list)):
                print(res_list[i][key]) 
        #self.logger()
        return
    
    def logger(self):
        """ どのファイルのeventを読み込んでグラフを作成したかテキストファイルに書き出す"""
        str_ = '\n'.join(self.name_list)
        with open(self.log_dir + "/filelist.txt", 'wt') as f:
            f.write(str_)
        return

    def gather_result(self):
        tmp_res = []
        old_key = None
        for event in self.events_list:
            result_dic = self.get_scores(event)
            if result_dic is None:
                continue
            else:
                if old_key is None:
                    tmp_res.append(result_dic)
                    old_key = result_dic.keys()
                    continue
                elif set(list(old_key)) == set(list(result_dic.keys())):
                    tmp_res.append(result_dic)
                    old_key = result_dic.keys()
                else:
                    raise NotImplementedError()
        return tmp_res
        
    def get_scores(self, path_file):
        """指定されたファイルからスコアの系列を取得して返す

        :param path_file:
        :return:
        (steps, scores)
        steps : ステップの系列
        scores : スコアの系列
        """
        dic = {} # 結果を格納するdictionary
        accumulator = EventAccumulator(path_file)
        accumulator.Reload()

        tag_dict = accumulator.Tags() #'images', 'audio', 'histograms', 'scalars', 'distributions', 'tensors', 'graph', 'meta_graph', 'run_metadata'
        scalars_key = tag_dict['scalars']
        if not scalars_key:
            return None

        self.name_list.append(path_file.split('/')[1])
        for key in scalars_key:
            eval = accumulator.Scalars(key)
            if 'steps' not in dic:
                dic['steps'] = np.array([tmp.step for tmp in eval])
            dic[key] = np.array([tmp.value for tmp in eval]) 

        return dic

    def make_graph(self, values):
        fig = plt.figure(figsize=(10,3))
        colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]
        ax = fig.add_subplot(1, 2, 1)
        tmp = np.zeros((len(values), values[0].shape[0]))
        for i,value in enumerate(values):
            tmp[i] = value
            plt.plot(range(value.shape[0]), value, linestyle='solid', color=colorlist[i], label=i)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        plt.grid(which='major',color='gray',linestyle='-')
        plt.title("FetchReach-v1 of Each Result")
        plt.xlabel("Episode")
        plt.ylabel("Success Rate")

        ax = fig.add_subplot(1, 2, 2)
        mean = np.mean(tmp, axis=0)
        std = np.std(tmp, axis=0)
        plt.grid(which='major',color='gray',linestyle='-')
        plt.plot(range(tmp.shape[1]), mean, linestyle='solid', color='blue', label=i)
        plt.fill_between(range(tmp.shape[1]) ,mean - std,mean + std,facecolor='blue',alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        plt.title("FetchReach-v1 of Average Result")
        plt.xlabel("Episode")
        plt.ylabel("Success Rate")
        plt.savefig(self.log_dir + '/accuracy.png')
        return

def main(args):
    events_list = list(chain.from_iterable([glob.glob(res_dir+"/**/events.*", recursive=True) for res_dir in args.dir]))
    evget = EventGetter(events_list)
    evget()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',nargs='*', help='tensorboard event directory')
    args = parser.parse_args()
    main(args)