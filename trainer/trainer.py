import os, sys
import time
import tensorflow as tf
from utility.utils import Utils
from dataset.load import Load
from collections import OrderedDict

class Trainer():
    def __init__(self,
                 FLAGS,
                 message,
                 data,
                 model,
                 name):
        self.name = name
        self.message = message
        self.data = data
        self.model = model
        self.n_epoch = FLAGS.n_epoch
        self.batch_size = FLAGS.batch_size
        self.restore_dir = FLAGS.init_model
        self.util = Utils(prefix=self.name)
        self.util.initial()
        if hasattr(FLAGS, 'aug'):
            self.aug = FLAGS.aug

    def load(self):
        train_dataset = self.data.load(self.data.x_train, self.data.y_train, batch_size=self.batch_size, is_training=True, augmentation=self.aug)
        test_dataset = self.data.load(self.data.x_test, self.data.y_test, batch_size=self.batch_size*3, is_training=False)
        return train_dataset, test_dataset

    def begin_train(self):
        self.util.write_configuration(self.message, True)
        self.util.save_init(self.model)
        return tf.summary.create_file_writer(self.util.tf_board)

    @tf.function
    def _train_body(self, images, labels):
        with tf.GradientTape() as tape:
            y_pre = self.model.inference(images)
            loss = self.model.loss(y_pre, labels)
        self.model.optimize(loss, tape)
        acc = self.model.accuracy(y_pre, labels)
        return loss, acc

    @tf.function
    def _test_body(self, images, labels):
        y_pre = self.model.inference(images)
        loss = self.model.loss(y_pre, labels)
        acc = self.model.accuracy(y_pre, labels)
        return loss, acc

    def epoch_end(self, metrics, other=None):
        try:
            learning_rate = self.model.optimizer.lr(metrics['epoch']).numpy()
        except:
            learning_rate = self.model.optimizer.lr
        tf.summary.experimental.set_step(metrics['epoch'])
        tf.summary.scalar('detail/epoch', metrics['epoch'])
        tf.summary.scalar('detail/time_per_step', metrics['time/step'])
        tf.summary.scalar('detail/learning_rate', learning_rate)
        tf.summary.scalar('train/loss', metrics['train_loss'])
        tf.summary.scalar('train/accuracy', metrics['train_accuracy'])
        tf.summary.scalar('test/loss', metrics['test_loss'])
        tf.summary.scalar('test/accuracy', metrics['test_accuracy'])
        if 'train_image' in other and len(other['train_image'].shape) == 4:
            tf.summary.image('train/image', other['train_image'])
            tf.summary.image('test/image', other['test_image'])


        print("epoch: %d  train_loss: %.4f  train_accuracy: %.3f test_loss: %.4f  test_accuracy: %.3f  time/step: %0.3fms" 
                                %(metrics['epoch'], metrics['train_loss'], metrics['train_accuracy'], metrics['test_loss'], metrics['test_accuracy'], metrics['time/step']))
        self.util.write_log(message=metrics)
        if metrics['epoch'] % 50 == 0:
            self.util.save_model(global_step=metrics['epoch'])
        return

    def train(self):
        board_writer = self.begin_train()
        board_writer.set_as_default()
        if self.restore_dir is not None:
            self.util.restore_agent(self.model ,self.restore_dir)
        
        train_dataset, test_dataset = self.load()

        # set mean loss
        train_loss_fn = tf.keras.metrics.Mean(name='train_loss')
        test_loss_fn = tf.keras.metrics.Mean(name='test_loss')

        for i in range(1, self.n_epoch+1):
            start_time = time.time()
            for (_, (train_images, train_labels)) in enumerate(train_dataset.take(self.batch_size)):
                loss, train_accuracy = self._train_body(train_images, train_labels)
                train_loss = train_loss_fn(loss)
            time_per_episode = time.time() - start_time
            for (_, (test_images, test_labels)) in enumerate(test_dataset.take(self.batch_size)):
                loss, test_accuracy = self._test_body(test_images, test_labels)
                test_loss = test_loss_fn(loss)

            # Training results
            metrics = OrderedDict({
                "epoch": i,
                "train_loss": train_loss.numpy(),
                "train_accuracy":train_accuracy.numpy(),
                "test_loss": test_loss.numpy(),
                "test_accuracy" : test_accuracy.numpy(),
                "time/step": time_per_episode
            })

            #
            other_metrics = OrderedDict({
                "train_image" : train_images[:3],
                "test_image" : test_images[:3]
            })
            self.epoch_end(metrics, other_metrics)
        return


class AE_Trainer(Trainer):
    def __init__(self,
                 FLAGS,
                 message,
                 data,
                 model,
                 name):
        super().__init__()