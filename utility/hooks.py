import os
import tempfile
import shutil
import time
import logging
import signal
import importlib
import importlib.util

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import sylph.util.exporter as exporter
from sylph.network import losses

class ThroughputMeasureHook(tf.train.SessionRunHook):
    """Throughput計測用のHook

    sess.runの前後で時間を計測し、１秒あたりの処理データ量などを記録する

    """
    def __init__(self, batch_size, average_steps=10, warmup_steps=100, logger=None):
        """

        :param int batch_size: 1stepあたりに処理するデータ量
        :param average_steps: 何stepごとの平均をとるか
        :param logger: loggingに用いるLogger
        """
        self._batch_size = batch_size
        self._step_counter = 0
        self._warmup_steps = warmup_steps
        self._average_steps = average_steps

        self._warmup_completed = False

        if logger is None:
            logger = logging.getLogger(__name__)

        self._logger = logger

        # 全体平均用のカウンタ群
        self._sum_data_per_sec = 0
        self._count_data_per_sec = 0


    def before_run(self, run_context):
        self._step_counter += 1

        if not self._warmup_completed:
            if self._step_counter <= self._warmup_steps:
                return

            self._warmup_completed = True
            self._list_duration = []

        self._start_time = time.time()

    def after_run(self, run_context, run_values):
        if not self._warmup_completed:
            return

        end_time = time.time()
        duration_seconds = end_time - self._start_time
        self._list_duration.append(duration_seconds)

        if len(self._list_duration) == self._average_steps:
            total_data = self._batch_size * self._average_steps
            total_time = np.sum(self._list_duration)
            data_per_sec = total_data / total_time

            self._logger.info("performance {:.1f} [data/sec]".format(data_per_sec))

            self._sum_data_per_sec += data_per_sec
            self._count_data_per_sec += 1
            self._list_duration = []

    def end(self, session):
        if self._count_data_per_sec > 0:
            average = self._sum_data_per_sec / self._count_data_per_sec
            self._logger.info("total performance {:.1f} [data/sec]".format(average))


class SignalHandlerHook(tf.train.SessionRunHook):
    """SIGNALをキャッチしてMonitoredTrainingSessionを終了させるHook

    """
    def __init__(self, target_signal=signal.SIGINT):
        self._signal_received = False

        def _handler(signal_no, frame):
            self._signal_received = True

        signal.signal(target_signal, _handler)

    def before_run(self, run_context):
        if self._signal_received:
            run_context.request_stop()

    def after_run(self, run_context, run_values):
        if self._signal_received:
            run_context.request_stop()


class QuintusHook(tf.train.SessionRunHook):
    """Quintusのアップロードを行うためのHook

    このクラスをインスタンス化しない限り、quintusに依存しないようにする
    このHookはすべてのHookの最後にhooksに加えるべき

    """
    def __init__(self, exp_root, flags, path_input=None):
        """

        :param str exp_root:
        :param flags:
        :param str path_input: データディレクトリ (なくともよい)
        """
        spec = importlib.util.find_spec("quintus")
        if spec is None:
            raise ValueError("Quintus is not found.")
        _quintus = spec.loader.load_module()

        self._quintus = _quintus.Quintus(exp_root=exp_root)
        self._exp_flags = flags
        self._exp_path_input = path_input

    def begin(self):
        self._quintus.start_experiment(flags=self._exp_flags, path_input=self._exp_path_input)

    def end(self, sess):
        self._quintus.finish_experiment()


class EvaluatorHook(tf.train.SessionRunHook):
    """Validation Datasetにおける評価を行う。

    trainとSessionを共有する

    """
    def __init__(self,
                 ts_eval_loss,
                 ts_corrects,
                 fn_reset,
                 every_n_iter=100,
                 max_eval_iter=None,
                 output_dir=None,
                 summary_writer=None,
                 use_keras=False):
        """

        :param tf.Tensor ts_eval_loss: 損失を表すTensor。 Tensor<None(batch_size)>
        :param tf.Tensor ts_corrects: 各データが正解しているか表すTensor。 Tensor<None(batch_size)>
        :param fn_reset: Dataset等を初期化するための関数（各評価タイミングの先頭で呼ばれる）。引数にtf.Sessionをとる
        :param int every_n_iter: 何ステップごとに精度を計測するか
        :param int max_eval_iter: 評価時にevaluationの上限を設けるかどうか (Noneならデータセットを一周する)
        :param str output_dir: FileWriterの出力先ディレクトリ (summary_writerが指定されていれば不要)
        :param tf.summary.FileWriter summary_writer: 結果を出力するためのwriter
        :param bool use_keras: Kerasを利用しているかどうか。(feed_dict時にtraining_phaseを与えるかどうか)
        """

        if every_n_iter is None or every_n_iter <= 0:
            raise ValueError('invalid every_n_iter=%s.' % every_n_iter)

        self._every_n_iter = every_n_iter
        self._max_eval_iter = max_eval_iter
        self._ts_eval_loss = ts_eval_loss
        self._ts_corrects = ts_corrects
        self._fn_reset = fn_reset
        self._use_keras = use_keras

        self._iter_count = 0
        self._timer = tf.train.SecondOrStepTimer(every_steps=every_n_iter)

        if summary_writer is not None:
            self._summary_writer = summary_writer
        elif output_dir is not None:
            self._summary_writer = tf.summary.FileWriterCache.get(output_dir)
        else:
            self._summary_writer = None

    def _evaluate(self, sess):
        logger = logging.getLogger(__name__)

        self._fn_reset(sess)

        list_loss = []
        list_corrects = []

        global_step = tf.train.get_global_step()
        step_value = sess.run(global_step)

        # Keras考慮コード
        if self._use_keras:
            learning_phase = keras.backend.learning_phase()
            feed_dict = {learning_phase: False}
        else:
            feed_dict = None

        try:
            if self._max_eval_iter is None:
                while True:
                    cur_loss, cur_corrects = sess.run([self._ts_eval_loss, self._ts_corrects],
                                                      feed_dict=feed_dict)

                    list_loss += list(cur_loss)
                    list_corrects += list(cur_corrects)
            else:
                for idx_loop in range(self._max_eval_iter):
                    cur_loss, cur_corrects = sess.run([self._ts_eval_loss, self._ts_corrects],
                                                      feed_dict=feed_dict)

                    list_loss += list(cur_loss)
                    list_corrects += list(cur_corrects)

        except tf.errors.OutOfRangeError:
            pass

        num_data = len(list_corrects)
        num_corrects = np.sum(list_corrects)
        average_loss = np.mean(list_loss)

        acc = num_corrects/num_data

        logger.info("Evaluate {} points. ACC {:.1f}% {}/{}".format(
            num_data, 100*acc, num_corrects, num_data))

        # TODO Summaryの出力名を外部から与えられるようにする
        if self._summary_writer:
            summary = tf.Summary()
            summary.value.add(tag="loss/acc", simple_value=acc)
            summary.value.add(tag="loss/classification", simple_value=average_loss)
            self._summary_writer.add_summary(summary, step_value)

        self._timer.update_last_triggered_step(self._iter_count)

    def after_run(self, run_context, run_values):
        """Runs evaluator."""
        self._iter_count += 1
        if self._timer.should_trigger_for_step(self._iter_count):
            self._evaluate(run_context.session)

    def end(self, session):
        """Runs evaluator for final model."""
        self._evaluate(session)

        
class WriteSavedModelHook(tf.train.SessionRunHook):
    """SavedModelを最後に出力するためのHook

    sylphの仕様に強烈によっているため一般性はない
    """

    def __init__(self, dataset, network, saved_model_path, remove_existed_model=False):
        """推論用のグラフからSavedModelを構築する

        :param sylph.dataset.dataset.DatasetBase dataset: 入力にとるデータセット
        :param sylph.network.network.ClassificationNetwork network: 推論用グラフ
        :param str saved_model_path: 出力先ディレクトリ
        :param bool remove_existed_model: すでにSavedModelが存在していた場合削除するかどうか
        :return:
        """
        self._dataset = dataset
        self._network = network
        self._saved_model_path = saved_model_path

        if os.path.exists(saved_model_path):
            if remove_existed_model:
                shutil.rmtree(saved_model_path)
            else:
                raise ValueError("SavedModel {} exists".format(saved_model_path))


    def end(self, session):
        exporter.write_saved_model(self._dataset, self._network, session, self._saved_model_path)


class WriteSavedModelHookForKeras(tf.train.SessionRunHook):
    """SavedModelを最後に出力するためのHook (Keras由来のモデル用)

    sylphの仕様に強烈によっているため一般性はない
    """

    def __init__(self, dataset, pre_build_model, saved_model_path, checkpoint_path,
                 out_checkpointable=None, remove_existed_model=False):
        """推論用のグラフからSavedModelを構築する (Keras用)

        :param sylph.dataset.dataset.DatasetBase dataset: 入力にとるデータセット (前処理用)
        :param tf.keras.Model pre_build_model: Kerasモデル
        :param str saved_model_path: 出力先ディレクトリ
        :param str check_model_path: parameterディレクトリ
        :parma str out_checkpointable: Checkpointableを出力する場合のprefix
        :param bool remove_existed_model: すでにSavedModelが存在していた場合削除するかどうか
        :return:
        """
        self._dataset = dataset
        self._pre_build_model = pre_build_model
        self._saved_model_path = saved_model_path
        self._checkpoint_path = checkpoint_path
        self._out_checkpointable = out_checkpointable

        if os.path.exists(saved_model_path):
            if remove_existed_model:
                shutil.rmtree(saved_model_path)
            else:
                raise ValueError("SavedModel {} exists".format(saved_model_path))


    def end(self, session):
        exporter.write_saved_model_for_keras(self._dataset, self._pre_build_model,
                                             self._saved_model_path, self._checkpoint_path, self._out_checkpointable)



class RuntimeGraphSaverHook(tf.train.SessionRunHook):
    """実行時のグラフ(device mappingが正しい)を保存する

    初回SessionRun時に実行するようになっている (init系に組み込むべき？)
    一応作ったが、ProfilerHookの方が出る情報量が多いのでそちらを使うこと
    (縮約されたグラフはこちらでしかでない？)
    """

    def __init__(self,
                 output_dir):
        """

        :param output_dir: 出力先ディレクトリ
        """
        self._output_dir = output_dir
        self._completed_output = False

    def begin(self):
        self._summary_writer = tf.summary.FileWriterCache.get(self._output_dir)

    def before_run(self, run_context):
        if not self._completed_output:
            global_step = tf.train.get_global_step()
            requests = {"global_step": global_step}
            opts = tf.RunOptions(output_partition_graphs=True)

            return tf.train.SessionRunArgs(requests, options=opts)

        return None

    def after_run(self, run_context, run_values):
        if not self._completed_output:
            self._completed_output = True
            run_metadata = run_values.run_metadata

            for cur_graph in run_metadata.partition_graphs:
                self._summary_writer.add_graph(graph=cur_graph)

            self._summary_writer.flush()


def construct_sylph_evaluator_hook(batch_size, dataset, network, dir_out, every_n_iter=1000, max_eval_iter=None, use_keras=False):
    """sylph標準のdatasetとnetworkから評価用のhookを作成する

    :param int batch_size: 評価時のbatch_size
    :param sylph.dataset.dataset.DatasetBase dataset: 入力にとるデータセット
    :param sylph.network.network.ClassificationNetwork network: 推論用グラフ
    :param str dir_out: Evlaluatorの出力先
    :param int every_n_iter: 何回のiterationごとに評価するか
    :param int max_eval_iter: 評価でのiteration回数の上限
    :param bool use_keras: Kerasを利用しているかどうか (learning modeをfeet_dictで与えるかどうか)
    :return:
    """
    eval_dataset = dataset

    iterator, dataset_init_fn = eval_dataset.get_dataset_iterator(
        batch_size=batch_size, repeat_and_shuffle=False, distortion=False
    )

    labels, images = iterator.get_next()
    logits = network.inference(images)
    detection = tf.argmax(logits, axis=1)
    is_corrects = tf.equal(detection, labels)

    eval_loss = losses.classification_loss_per_data(logits, labels)
    eval_corrects  = is_corrects

    return EvaluatorHook(ts_eval_loss=eval_loss, ts_corrects=eval_corrects, fn_reset=dataset_init_fn,
                         output_dir=dir_out, every_n_iter=every_n_iter, max_eval_iter=max_eval_iter, use_keras=use_keras)



class CheckpointableSaverHook(tf.train.SessionRunHook):
    """
    CheckpointSaverHookのようにCheckpointを周期的に保存する
    """

    def __init__(self,
                 checkpoint,
                 output_prefix,
                 every_n_steps=100,
                 every_n_secs=None):
        """

        :param tf.train.Checkpoint checkpoint:
        :param output_prefix:
        :param every_n_steps:
        :param every_n_secs:
        """

        if (every_n_steps is None) == (every_n_secs is None):
            raise ValueError("exactly one of every_n_steps and every_n_secs should be provided.")

        self._timer = tf.train.SecondOrStepTimer(every_steps=every_n_steps,
                                                 every_secs=every_n_secs)

        self._checkpoint = checkpoint
        self._output_prefix = output_prefix
        self._last_global_step = None
        self._global_step_check_count = 0
        self._steps_per_run = 1

    def construct_graph(self):
        """MonitoredSession等のfinalizedしたGraphに対して書き込めないことへの対策コード

        Graphがfinalizeされる前に呼ぶ

        :return:
        """
        tmp_dir = tempfile.gettempdir()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self._checkpoint.save(tmp_dir + "/ckpt")

    def _set_steps_per_run(self, steps_per_run):
        self._steps_per_run = steps_per_run

    def _save(self, global_step, session):
        self._checkpoint.save(self._output_prefix, session=session)

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()

        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use CheckpointableSaverHook.")

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results
        if self._timer.should_trigger_for_step(
                stale_global_step + self._steps_per_run):
            # get the real value after train op.
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)

                self._save(global_step, run_context.session)

    def end(self, session):
        global_step = session.run(self._global_step_tensor)
        self._save(global_step, session)
