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
    def __init__(self, events_list):
        dt_now = datetime.datetime.now()
        self.events_list = events_list
        self.log_dir = "results/" + dt_now.strftime("%y%m%d_%H%M%S") + "_events"
        self.result_dic = {}
        self.name_list = []
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
        for key in self.result_dic:
            self.make_graph(key, self.result_dic[key])
        #self.logger()
        return
    
    def logger(self):
        """ どのファイルのeventを読み込んでグラフを作成したかテキストファイルに書き出す"""
        str_ = '\n'.join(self.name_list)
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
        fig = plt.figure(figsize=(10,5))
        colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]
        ax = fig.add_subplot(1, 1, 1)
        i = 0
        for key in values:
            if key is None:
                continue
            array = values[key]
            plt.plot(range(array.shape[0]), array, linestyle='solid', color=colorlist[i], label=key)
            i += 1
        if re.search('accuracy', name):
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        plt.grid(which='major',color='gray',linestyle='-')
        plt.title(name)
        plt.xlabel("epoch")
        plt.ylabel("value")
        plt.legend()
        plt.savefig(self.log_dir + '/{}.png'.format(name))
        return

def main(args):
    assert args.dir is not None
    events_list = list(chain.from_iterable([glob.glob(res_dir+"tf_board/events.*", recursive=True) for res_dir in args.dir]))
    evget = EventGetter(events_list)
    evget()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',nargs='*', help='tensorboard event directory')
    args = parser.parse_args()
    main(args)