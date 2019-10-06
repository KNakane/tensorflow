import os, sys
import time
import tensorflow as tf
from utils import Utils
from eager_load import Load
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
        self.global_step = tf.train.get_or_create_global_step()
        self.util = Utils(prefix=self.name)
        self.util.initial() 
        if hasattr(FLAGS, 'aug'):
            self.aug = FLAGS.aug

    def load(self):
        dataset = self.data.load(self.data.x_train, self.data.y_train, batch_size=self.batch_size, is_training=True, augmentation=self.aug)
        return dataset

    def train(self):
        board_writer = self.begin_train()
        board_writer.set_as_default()
        if self.restore_dir is not None:
            self.util.restore_agent(self.model ,self.restore_dir)
        
        #inputs, corrects, valid_inputs, valid_labels = self.load()
        dataset = self.load()

        for i in range(self.n_epoch):
            running_loss = 0
            running_acc = 0
            start_time = time.time()
            for (batch, (images, labels)) in enumerate(dataset.take(self.batch_size)):
                loss, acc = self._train_body(self.model, images, labels, self.global_step)
                running_loss += tf.reduce_mean(loss)
                running_acc += tf.reduce_mean(acc)
            time_per_episode = time.time() - start_time
            self.epoch_end(i, running_loss, running_acc, time_per_episode)
        return

    def begin_train(self):
        self.util.write_configuration(self.message, True)
        self.util.save_init(self.model)
        return tf.contrib.summary.create_file_writer(self.util.tf_board)

    @tf.contrib.eager.defun
    def _train_body(self, model, images, labels, global_steps):
        with tf.GradientTape() as tape:
            y_pre = model.inference(images)
            acc = model.accuracy(y_pre, labels)
            loss = model.loss(y_pre, labels)
        model.optimize(loss, global_steps, tape)
        return loss, acc

    def epoch_end(self, epoch, loss, accuracy, time):
        loss = loss.numpy()/self.batch_size
        accuracy = 100*accuracy.numpy()/self.batch_size
        tf.contrib.summary.scalar('epoch', epoch)
        tf.contrib.summary.scalar('train/loss', loss)
        tf.contrib.summary.scalar('train/accuracy', accuracy)
        tf.contrib.summary.scalar('time_per_step', time)
        print("epoch: %d  train_loss: %.4f  train_accuracy: %.3f  time/step: %0.3fms" %(epoch+1, loss, accuracy, time))
        metrics = OrderedDict({
            "epoch": epoch,
            "train_loss": loss,
            "train_accuracy":accuracy,
            "time/step": time})
        self.util.write_log(message=metrics)
        if epoch % 50:
            self.util.save_model()
        return