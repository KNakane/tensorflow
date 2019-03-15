import os,sys
import six
import tempfile
import shutil
import time
import logging
import signal
import importlib
import importlib.util

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class SavedModelBuilderHook(tf.train.SessionRunHook):
    """
    SavedModelの出力する
    parameters
    ----------
    export_dir : string SavedModelを出力するdirectory
    
    signature_def_map : 保存内容
    """
    def __init__(self, export_dir, signature_def_map, tags=None):
        self.export_dir = export_dir
        self.signature_def_map = signature_def_map
        self.tags = tags if tags is not None else [tf.saved_model.tag_constants.SERVING]

    def end(self, session):
        session.graph._unsafe_unfinalize()
        builder = tf.saved_model.builder.SavedModelBuilder(self.export_dir)
        builder.add_meta_graph_and_variables(
            session,
            self.tags,
            signature_def_map=self.signature_def_map
        )
        builder.save()

class OptunaHook(tf.train.SessionRunHook):
    """
    optuna用のHook

    parameters
    -----------
    tensors : dict 表示させたい内容

    return
    ----------
    学習終了時に結果を表示する

    """
    def __init__(self, tensors, formatter=None):
        self._tensors = tensors
        self._tag_order = tensors.keys()
        
    def begin(self):
        self._iter_count = 0
        self._current_tensors = {tag: _as_graph_element(tensor)
                                 for (tag, tensor) in self._tensors.items()}
        return

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self._current_tensors)

    def end(self, session):
        values = session.run(self._current_tensors)
        stats = []
        for tag in self._tag_order:
            stats.append("%s = %s" % (tag, values[tag]))
        logging.info("%s", ", ".join(stats))

class MyLoggerHook(tf.train.SessionRunHook):
    """
    terminalとlogファイルに学習過程を出力をする
    parameters
    ----------
    message : dict 表示する内容
    
    log_dir : string logを出力するdirectory
    
    every_n_iter : int
    
    every_n_secs : int
    
    at_end : bool

    """
    def __init__(self, message, log_dir, tensors, every_n_iter=None, every_n_secs=None,
                 at_end=True, formatter=None):
        self.log_dir = log_dir
        self.message = message
        if not os.path.isdir(self.log_dir):
            tf.gfile.MakeDirs(self.log_dir)
        only_log_at_end = (
            at_end and (every_n_iter is None) and (every_n_secs is None))
        if (not only_log_at_end and
            (every_n_iter is None) == (every_n_secs is None)):
            raise ValueError(
                "either at_end and/or exactly one of every_n_iter and every_n_secs "
                "must be provided.")
        if every_n_iter is not None and every_n_iter <= 0:
            raise ValueError("invalid every_n_iter=%s." % every_n_iter)
        if not isinstance(tensors, dict):
            self._tag_order = tensors
            tensors = {item: item for item in tensors}
        else:
            self._tag_order = tensors.keys()
        self._tensors = tensors
        self._formatter = formatter
        self._timer = (
            NeverTriggerTimer() if only_log_at_end else
            tf.train.SecondOrStepTimer(every_secs=every_n_secs, every_steps=every_n_iter))
        self._log_at_end = at_end

    def begin(self):
        self._timer.reset()
        self._iter_count = 0
        self.f = open(self.log_dir + '/log.txt', 'w')
        self._current_tensors = {tag: _as_graph_element(tensor)
                                 for (tag, tensor) in self._tensors.items()}
        self._opening()

    def _opening(self):
        if len(self.message) or self.message is not None:
            print("------Learning Details------")
            self.f.write("------Learning Details------\n")
            for key, info in self.message.items():
                print("%s : %s"%(key, info))
                self.f.write("%s : %s\n"%(key, info))
            print("----------------------------")
            self.f.write("----------------------------\n")
        else:
            pass

    def before_run(self, run_context):  # pylint: disable=unused-argument
        self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
        if self._should_trigger:
            return tf.train.SessionRunArgs(self._current_tensors)
        else:
            return None

    def _log_tensors(self, tensor_values):
        original = np.get_printoptions()
        np.set_printoptions(suppress=True)
        elapsed_secs, _ = self._timer.update_last_triggered_step(self._iter_count)
        if self._formatter:
            logging.info(self._formatter(tensor_values))
        else:
            stats = []
            for tag in self._tag_order:
                stats.append("%s = %s" % (tag, tensor_values[tag]))
            if elapsed_secs is not None:
                info = "%s (%.3f sec)\n"%(", ".join(stats), elapsed_secs)
                self.f.write(str(info))
                logging.info("%s (%.3f sec)", ", ".join(stats), elapsed_secs)
            else:
                logging.info("%s", ", ".join(stats))
            np.set_printoptions(**original)

    def after_run(self, run_context, run_values):
        _ = run_context
        if self._should_trigger:
            self._log_tensors(run_values.results)

        self._iter_count += 1

    def end(self, session):
       if self._log_at_end:
           values = session.run(self._current_tensors)
           self._log_tensors(values)
           self.f.close()


class NeverTriggerTimer():
    """Timer that never triggers."""
    
    def should_trigger_for_step(self, step):
        _ = step
        return False
        
    def update_last_triggered_step(self, step):
        _ = step
        return (None, None)

    def last_triggered_step(self):
        return None


def _as_graph_element(obj):
    """Retrieves Graph element."""
    graph = ops.get_default_graph()
    if not isinstance(obj, six.string_types):
        if not hasattr(obj, "graph") or obj.graph != graph:
            raise ValueError("Passed %s should have graph attribute that is equal "
                       "to current graph %s." % (obj, graph))
        return obj
    if ":" in obj:
        element = graph.as_graph_element(obj)
    else:
        element = graph.as_graph_element(obj + ":0")
        # Check that there is no :1 (e.g. it's single output).
        try:
            graph.as_graph_element(obj + ":1")
        except (KeyError, ValueError):
            pass
        else:
            raise ValueError("Name %s is ambiguous, "
                             "as this `Operation` has multiple outputs "
                             "(at least 2)." % obj)
    return element


class GanHook(tf.train.SessionRunHook):
    """
    GANで生成した画像をepochごとに保存するHook
    ----------
    image : 生成した画像 
    
    log_dir : string logを出力するdirectory
    
    every_n_iter : int

    every_n_secs : int
    
    at_end : bool

    """
    def __init__(self, image, log_dir, every_n_iter=None, every_n_secs=None, at_end=True):
        self.log_dir = log_dir
        self.image = image
        if not os.path.isdir(self.log_dir):
            tf.gfile.MakeDirs(self.log_dir)
        only_log_at_end = (
            at_end and (every_n_iter is None) and (every_n_secs is None))
        if (not only_log_at_end and
            (every_n_iter is None) == (every_n_secs is None)):
            raise ValueError(
                "either at_end and/or exactly one of every_n_iter and every_n_secs "
                "must be provided.")
        if every_n_iter is not None and every_n_iter <= 0:
            raise ValueError("invalid every_n_iter=%s." % every_n_iter)
        self._timer = (
            NeverTriggerTimer() if only_log_at_end else
            tf.train.SecondOrStepTimer(every_secs=every_n_secs, every_steps=every_n_iter))
        self._log_at_end = at_end

    def begin(self):
        self._timer.reset()
        self._iter_count = 0
        return

    def before_run(self, run_context):  # pylint: disable=unused-argument
        self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
        if self._should_trigger:
            return tf.train.SessionRunArgs(self.image)
        else:
            return None

    def after_run(self, run_context, run_values):
        _ = run_context
        if self._should_trigger:
            self._timer.update_last_triggered_step(self._iter_count)
            self.plot_figure(run_values.results, self._iter_count)

        self._iter_count += 1

    def end(self, session):
       if self._log_at_end:
           self.plot_figure(session.run([self.image])[0], self._iter_count)

    def plot_figure(self, samples, iteration):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i in range(16):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(samples[i].reshape(28, 28), cmap='Greys_r')

        
        name = self.log_dir + '/{}.png'.format(str(iteration).zfill(3))
        plt.savefig(name, bbox_inches='tight')
        plt.close(fig)

        return 


class AEHook(GanHook):
    """
    AutoEncoderで生成した画像をepochごとに保存するHook
    ----------
    image : 生成した画像 
    
    log_dir : string logを出力するdirectory
    
    every_n_iter : int

    every_n_secs : int
    
    at_end : bool

    """
    def __init__(self, image, log_dir, every_n_iter=None, every_n_secs=None, at_end=True):
        super().__init__(image, log_dir, every_n_iter, every_n_secs, at_end)

    def begin(self):
        self._timer.reset()
        self._iter_count = 0
        return

    def before_run(self, run_context):  # pylint: disable=unused-argument
        self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
        if self._should_trigger:
            return tf.train.SessionRunArgs(self.image)
        else:
            return None

    def after_run(self, run_context, run_values):
        _ = run_context
        if self._should_trigger:
            self._timer.update_last_triggered_step(self._iter_count)
            self.plot_figure(run_values.results, self._iter_count)

        self._iter_count += 1

    def end(self, session):
       if self._log_at_end:
           self.plot_figure(session.run([self.image])[0], self._iter_count)

    def plot_figure(self, samples, iteration):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i in range(16):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(samples[i].reshape(28, 28), cmap='Greys_r')

        
        name = self.log_dir + '/{}.png'.format(str(iteration).zfill(3))
        plt.savefig(name, bbox_inches='tight')
        plt.close(fig)

        return 