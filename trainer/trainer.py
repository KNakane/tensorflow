import sys
sys.path.append('./utility')
sys.path.append('./network')
sys.path.append('./dataset')
import tensorflow as tf
from utils import Utils
from collections import OrderedDict
from hooks import SavedModelBuilderHook, MyLoggerHook, OptunaHook, AEHook

class BasedTrainer():
    def __init__(self,
                 FLAGS,
                 message,
                 data,
                 model,
                 name):
        self.checkpoints_to_keep = FLAGS.checkpoints_to_keep
        self.keep_checkpoint_every_n_hours = FLAGS.keep_checkpoint_every_n_hours
        self.max_steps = FLAGS.n_epoch
        self.save_checkpoint_steps = self.max_steps / 10 if FLAGS.save_checkpoint_steps is None else FLAGS.save_checkpoint_steps
        self.batch_size = FLAGS.batch_size
        if hasattr(FLAGS, 'aug') and FLAGS.aug != "None":
            self.aug = FLAGS.aug
        else:
            self.aug = None
        self.name = name
        self.message = message
        self.data = data
        self.global_step = tf.train.get_or_create_global_step()
        self.model = model
        self.restore_dir = FLAGS.init_model
        self.util = Utils(prefix=self.name)

    def load(self):
        # Load Dataset
        dataset = self.data.load(self.data.x_train, self.data.y_train, batch_size=self.batch_size, is_training=True, augmentation=self.aug)
        valid = self.data.load(self.data.x_test, self.data.y_test, batch_size=self.batch_size*3, is_training=False)
        self.iterator = dataset.make_initializable_iterator()
        self.valid_iter = valid.make_initializable_iterator()
        inputs, labels = self.iterator.get_next()
        valid_inputs, valid_labels = self.valid_iter.get_next()
        return inputs, labels, valid_inputs, valid_labels

    def build_logits(self, train_data, train_ans, valid_data, valid_ans):
        raise Exception('Set logits functuin')

    def hook_append(self, metrics, signature_def_map=None):
        """
        hooksをまとめる関数
        """
        hooks = []
        hooks.append(tf.train.NanTensorHook(self.train_loss))
        return hooks

    def summary(self, train_inputs, valid_inputs):
        """
        tensorboardに表示するデータをまとめる関数
        """
        # tensorboard
        tf.summary.scalar('train/loss', self.train_loss)
        tf.summary.scalar('train/accuracy', self.train_accuracy)
        tf.summary.scalar('train/Learning_rate', self.model.optimizer.lr)
        tf.summary.scalar('test/loss', self.test_loss)
        tf.summary.scalar('test/accuracy', self.test_accuracy)
        if len(train_inputs.shape) == 4:
            tf.summary.image('train/image', train_inputs)
            tf.summary.image('test/image', valid_inputs)
        if self.name == 'AutoEncoder' or self.name == 'VAE' or self.name == 'CVAE':
            tf.summary.image('train/encode_image', self.train_logits)
            tf.summary.image('test/encode_image', self.test_logits)
        return

    def before_train(self):

        def init_fn(scaffold, session):
            session.run([self.iterator.initializer,self.valid_iter.initializer],
                        feed_dict={self.data.features_placeholder: self.data.x_train,
                                   self.data.labels_placeholder: self.data.y_train,
                                   self.data.valid_placeholder: self.data.x_test,
                                   self.data.valid_labels_placeholder: self.data.y_test})
        
        # create saver
        saver = tf.train.Saver(
                max_to_keep=self.checkpoints_to_keep,
                keep_checkpoint_every_n_hours=self.keep_checkpoint_every_n_hours)

        scaffold = tf.train.Scaffold(
            init_fn=init_fn,
            saver=saver)

        tf.logging.set_verbosity(tf.logging.INFO)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

        # saved model
        signature_def_map = {
                        'predict': tf.saved_model.signature_def_utils.build_signature_def(
                            inputs={'inputs': tf.saved_model.utils.build_tensor_info(self.data.features_placeholder)},
                            outputs={'predict': tf.saved_model.utils.build_tensor_info(self.predict)},
                            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,)
                        }

        metrics = OrderedDict({
            "global_step": self.global_step,
            "train loss": self.train_loss,
            "train accuracy":self.train_accuracy,
            "test loss": self.test_loss,
            "test accuracy":self.test_accuracy})

        hooks = self.hook_append(metrics, signature_def_map)

        session = tf.train.MonitoredTrainingSession(
            config=config,
            checkpoint_dir=self.util.model_path,
            hooks=hooks,
            scaffold=scaffold,
            save_summaries_steps=1,
            save_checkpoint_steps=self.save_checkpoint_steps,
            summary_dir=self.util.tf_board)
        
        return session

    def train(self):
        raise Exception('Set train functuin')


class Train(BasedTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_logits(self, train_data, train_ans, valid_data, valid_ans):
        # train
        self.train_logits = self.model.inference(train_data, train_ans) if self.name == 'CVAE' else self.model.inference(train_data)
        self.train_loss = self.model.loss(self.train_logits, train_data) if self.name == 'AutoEncoder' or self.name == 'VAE' or self.name == 'CVAE' else self.model.loss(self.train_logits, train_ans)
        self.predict = self.model.predict(train_data)
        opt_op = self.model.optimize(self.train_loss, self.global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op = tf.group([opt_op] + update_ops)
        self.train_accuracy = self.model.evaluate(self.train_logits, train_data) if self.name == 'AutoEncoder' or self.name == 'VAE' or self.name == 'CVAE' else self.model.evaluate(self.train_logits, train_ans)

        # test
        self.test_logits = self.model.test_inference(valid_data, valid_ans, True) if self.name == 'CVAE' else self.model.test_inference(valid_data, reuse=True)
        self.test_loss = self.model.loss(self.test_logits, valid_data) if self.name == 'AutoEncoder' or self.name == 'VAE' or self.name == 'CVAE' else self.model.loss(self.test_logits, valid_ans)
        self.test_accuracy = self.model.evaluate(self.test_logits, valid_data) if self.name == 'AutoEncoder' or self.name == 'VAE' or self.name == 'CVAE' else self.model.evaluate(self.test_logits, valid_ans)

        return 

    def hook_append(self, metrics, signature_def_map=None):
        """
        hooksをまとめる関数
        """
        hooks = super().hook_append(metrics=metrics, signature_def_map=signature_def_map)
        hooks.append(MyLoggerHook(self.message, self.util.log_dir, metrics, every_n_iter=100))
        hooks.append(SavedModelBuilderHook(self.util.saved_model_path, signature_def_map))
        if self.max_steps:
            hooks.append(tf.train.StopAtStepHook(last_step=self.max_steps))
        return hooks

    def train(self):
        self.util.conf_log()
        inputs, corrects, valid_inputs, valid_labels = self.load()
        self.build_logits(inputs, corrects, valid_inputs, valid_labels)
        self.summary(inputs, valid_inputs)
        session = self.before_train()

        with session:
            if self.restore_dir is not None:
                ckpt = tf.train.get_checkpoint_state(self.restore_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    saver.restore(session, ckpt.model_checkpoint_path)
            while not session.should_stop():
                session.run([self.train_op])
        return 


class AETrainer(BasedTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_logits(self, train_data, train_ans, valid_data, valid_ans):
        # train
        self.train_logits = self.model.inference(train_data, train_ans) if self.name == 'CVAE' else self.model.inference(train_data)
        self.train_loss = self.model.loss(self.train_logits, train_data)
        self.predict = self.model.predict(train_data)
        opt_op = self.model.optimize(self.train_loss, self.global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op = tf.group([opt_op] + update_ops)
        self.train_accuracy = self.model.evaluate(self.train_logits, train_data)

        # test
        self.test_logits = self.model.test_inference(valid_data, valid_ans, True) if self.name == 'CVAE' else self.model.test_inference(valid_data, reuse=True)
        self.test_loss = self.model.loss(self.test_logits, valid_data)
        self.test_accuracy = self.model.evaluate(self.test_logits, valid_data)

        return 

    def hook_append(self, metrics, signature_def_map=None):
        """
        hooksをまとめる関数
        """
        hooks = super().hook_append(metrics=metrics, signature_def_map=signature_def_map)
        hooks.append(MyLoggerHook(self.message, self.util.log_dir, metrics, every_n_iter=100))
        hooks.append(SavedModelBuilderHook(self.util.saved_model_path, signature_def_map))
        hooks.append(AEHook(self.test_logits, self.util.log_dir, every_n_iter=100))
        if self.max_steps:
            hooks.append(tf.train.StopAtStepHook(last_step=self.max_steps))
        return hooks

    def train(self):
        self.util.conf_log()
        inputs, corrects, valid_inputs, valid_labels = self.load()
        self.build_logits(inputs, corrects, valid_inputs, valid_labels)
        self.summary(inputs, valid_inputs)
        session = self.before_train()

        with session:
            if self.restore_dir is not None:
                ckpt = tf.train.get_checkpoint_state(self.restore_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    saver.restore(session, ckpt.model_checkpoint_path)
            while not session.should_stop():
                _, test_input, test_output = session.run([self.train_op, valid_inputs, self.test_logits])
        
        self.util.construct_figure(test_input, test_output)
        return


class OptunaTrain(BasedTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_logits(self, train_data, train_ans, valid_data, valid_ans):
        # train
        self.train_logits = self.model.inference(train_data, train_ans) if self.name == 'CVAE' else self.model.inference(train_data)
        self.train_loss = self.model.loss(self.train_logits, train_data) if self.name == 'AutoEncoder' or self.name == 'VAE' or self.name == 'CVAE' else self.model.loss(self.train_logits, train_ans)
        self.predict = self.model.predict(train_data)
        opt_op = self.model.optimize(self.train_loss, self.global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op = tf.group([opt_op] + update_ops)
        self.train_accuracy = self.model.evaluate(self.train_logits, train_data) if self.name == 'AutoEncoder' or self.name == 'VAE' or self.name == 'CVAE' else self.model.evaluate(self.train_logits, train_ans)

        # test
        self.test_logits = self.model.test_inference(valid_data, valid_ans, True) if self.name == 'CVAE' else self.model.test_inference(valid_data, reuse=True)
        self.test_loss = self.model.loss(self.test_logits, valid_data) if self.name == 'AutoEncoder' or self.name == 'VAE' or self.name == 'CVAE' else self.model.loss(self.test_logits, valid_ans)
        self.test_accuracy = self.model.evaluate(self.test_logits, valid_data) if self.name == 'AutoEncoder' or self.name == 'VAE' or self.name == 'CVAE' else self.model.evaluate(self.test_logits, valid_ans)

        return

        
    def hook_append(self, metrics, signature_def_map=None):
        """
        hooksをまとめる関数
        """
        hooks = super().hook_append(metrics=metrics, signature_def_map=signature_def_map)
        hooks.append(OptunaHook(metrics))
        return hooks

    
    def train(self):
        """
        optunaを用いた場合のtrainer
        """
        inputs, corrects, valid_inputs, valid_labels = self.load()
        iteration = self.data.x_test.shape[0] // self.batch_size
        self.build_logits(inputs, corrects, valid_inputs, valid_labels)

        def init_fn(scaffold, session):
            session.run([self.iterator.initializer,self.valid_iter.initializer],
                        feed_dict={self.data.features_placeholder: self.data.x_train,
                                   self.data.labels_placeholder: self.data.y_train,
                                   self.data.valid_placeholder: self.data.x_test,
                                   self.data.valid_labels_placeholder: self.data.y_test})

        scaffold = tf.train.Scaffold(init_fn=init_fn)

        tf.logging.set_verbosity(tf.logging.INFO)
        metrics = OrderedDict({
            "global_step": self.global_step,
            "train loss": self.train_loss,
            "train accuracy":self.train_accuracy,
            "test loss": self.test_loss,
            "test accuracy":self.test_accuracy})

        hooks = self.hook_append(metrics)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7))

        session = tf.train.MonitoredTrainingSession(
                    config=config,
                    hooks=hooks,
                    scaffold=scaffold)

        with session:
            for _ in range(self.max_steps):
                session.run([self.train_op])
                avg_accuracy = 0
            for __ in range(iteration):
                test_accuracy = session.run(self.test_accuracy)
                avg_accuracy += test_accuracy

        return avg_accuracy / iteration