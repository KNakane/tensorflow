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
        self.save_checkpoint_steps = self.n_epoch / 10 if FLAGS.save_checkpoint_steps is None else FLAGS.save_checkpoint_steps
        self.checkpoints_to_keep = FLAGS.checkpoints_to_keep
        self.keep_checkpoint_every_n_hours = FLAGS.keep_checkpoint_every_n_hours
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
        self.util.save_init(self.model, keep=self.checkpoints_to_keep, n_hour=self.keep_checkpoint_every_n_hours)
        return tf.summary.create_file_writer(self.util.tf_board)

    @tf.function
    def _train_body(self, images, labels):
        with tf.GradientTape() as tape:
            with tf.name_scope('train_logits'):
                y_pre = self.model.inference(images)
            with tf.name_scope('train_loss'):
                loss = self.model.loss(y_pre, labels)
        self.model.optimize(loss, tape)
        with tf.name_scope('train_accuracy'):
            acc = self.model.accuracy(y_pre, labels)
        return y_pre, loss, acc

    @tf.function
    def _test_body(self, images, labels):
        with tf.name_scope('test_logits'):
            y_pre = self.model.test_inference(images)
        with tf.name_scope('test_loss'):
            loss = self.model.loss(y_pre, labels)
        with tf.name_scope('test_accuracy'):
            acc = self.model.accuracy(y_pre, labels)
        return y_pre, loss, acc

    def epoch_end(self, metrics, other=None):
        learning_rate = self.model.optimizer.lr(metrics['epoch']).numpy() if type(self.model.optimizer.lr) is tf.optimizers.schedules.ExponentialDecay else self.model.optimizer.lr
        tf.summary.experimental.set_step(metrics['epoch'])
        tf.summary.scalar('detail/epoch', metrics['epoch'])
        tf.summary.scalar('detail/time_per_step', metrics['time/epoch'])
        tf.summary.scalar('detail/learning_rate', learning_rate)
        tf.summary.scalar('train/loss', metrics['train_loss'])
        tf.summary.scalar('train/accuracy', metrics['train_accuracy'])
        tf.summary.scalar('test/loss', metrics['test_loss'])
        tf.summary.scalar('test/accuracy', metrics['test_accuracy'])
        if 'train_image' in other and len(other['train_image'].shape) == 4:
            tf.summary.image('train/image', other['train_image'])
            tf.summary.image('test/image', other['test_image'])
        if 'Decode_train_image' in other:
            tf.summary.image('train/decode_image', other['Decode_train_image'])
            tf.summary.image('test/decode_image', other['Decode_test_image'])


        print("epoch: %d  train_loss: %.4f  train_accuracy: %.3f test_loss: %.4f  test_accuracy: %.3f  time/epoch: %0.3fms" 
                                %(metrics['epoch'], metrics['train_loss'], metrics['train_accuracy'], metrics['test_loss'], metrics['test_accuracy'], metrics['time/epoch']))
        self.util.write_log(message=metrics)
        if metrics['epoch'] % self.save_checkpoint_steps == 0:
            self.util.save_model(global_step=metrics['epoch'])
        return

    def train(self):
        board_writer = self.begin_train()

        if self.restore_dir is not None:
            self.util.restore_agent(self.model ,self.restore_dir)
        
        train_dataset, test_dataset = self.load()

        # set mean loss
        train_loss_fn = tf.keras.metrics.Mean(name='train_loss')
        test_loss_fn = tf.keras.metrics.Mean(name='test_loss')

        # Graph for tensorboard
        tf.summary.trace_on(graph=True, profiler=True)

        with board_writer.as_default():
            for i in range(1, self.n_epoch+1):
                start_time = time.time()
                for (_, (train_images, train_labels)) in enumerate(train_dataset.take(self.batch_size)):
                    _, loss, train_accuracy = self._train_body(train_images, train_labels)
                    train_loss = train_loss_fn(loss)
                if i == 1:
                    tf.summary.trace_export("summary", step=1, profiler_outdir=self.util.tf_board)
                    tf.summary.trace_off()

                time_per_episode = time.time() - start_time
                for (_, (test_images, test_labels)) in enumerate(test_dataset.take(self.batch_size)):
                    _, loss, test_accuracy = self._test_body(test_images, test_labels)
                    test_loss = test_loss_fn(loss)

                # Training results
                metrics = OrderedDict({
                    "epoch": i,
                    "train_loss": train_loss.numpy(),
                    "train_accuracy":train_accuracy.numpy(),
                    "test_loss": test_loss.numpy(),
                    "test_accuracy" : test_accuracy.numpy(),
                    "time/epoch": time_per_episode
                })

                #
                other_metrics = OrderedDict({
                    "train_image" : train_images[:3],
                    "test_image" : test_images[:3]
                })
                self.epoch_end(metrics, other_metrics)
        
        
        tf.summary.trace_off()
        return


class AE_Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @tf.function
    def _train_body(self, images, correct_image, labels):
        with tf.GradientTape() as tape:
            with tf.name_scope('train_logits'):
                y_pre = self.model.inference(images, labels) if self.name == 'CVAE' else self.model.inference(images)
            with tf.name_scope('train_loss'):
                loss = self.model.loss(y_pre, correct_image)
        self.model.optimize(loss, tape)
        with tf.name_scope('train_accuracy'):
            acc = self.model.accuracy(y_pre, correct_image)
        return y_pre, loss, acc


    @tf.function
    def _test_body(self, images, correct_image, labels):
        with tf.name_scope('test_logits'):
            y_pre = self.model.test_inference(images, labels) if self.name == 'CVAE' else self.model.test_inference(images)
        with tf.name_scope('test_loss'):
            loss = self.model.loss(y_pre, correct_image)
        with tf.name_scope('test_accuracy'):
            acc = self.model.accuracy(y_pre, correct_image)
        with tf.name_scope('Prediction'):
            predict = self.model.predict(images)

        return y_pre, loss, acc, predict

    def train(self):
        board_writer = self.begin_train()
        board_writer.set_as_default()
        if self.restore_dir is not None:
            self.util.restore_agent(self.model ,self.restore_dir)
        
        train_dataset, test_dataset = self.load()

        # Graph for tensorboard
        tf.summary.trace_on(graph=True, profiler=True)
        with board_writer.as_default():
            for i in range(1, self.n_epoch+1):
                start_time = time.time()
                for (_, (train_images, train_labels)) in enumerate(train_dataset.take(self.batch_size)):
                    train_pre, train_loss, train_accuracy = self._train_body(train_images, train_images, train_labels)
                time_per_episode = time.time() - start_time
                
                if i == 1:
                    tf.summary.trace_export("summary", step=1, profiler_outdir=self.util.tf_board)
                    tf.summary.trace_off()

                for (_, (test_images, test_labels)) in enumerate(test_dataset.take(self.batch_size)):
                    test_pre, test_loss, test_accuracy, predict_image = self._test_body(test_images, test_images, test_labels)

                if i == 1 or i % 50 == 0:
                    if self.name == 'AutoEncoder':
                        self.util.construct_figure(test_images.numpy(), test_pre.numpy(), i)
            
                    elif self.name == 'VAE' or self.name == 'CVAE':
                        self.util.reconstruct_image(predict_image.numpy(), i)

                # Training results
                metrics = OrderedDict({
                    "epoch": i,
                    "train_loss": train_loss.numpy(),
                    "train_accuracy":train_accuracy,
                    "test_loss": test_loss.numpy(),
                    "test_accuracy" : test_accuracy.numpy(),
                    "time/epoch": time_per_episode
                })

                #
                other_metrics = OrderedDict({
                    "train_image" : train_images[:3],
                    "test_image" : test_images[:3],
                    "Decode_train_image" : train_pre,
                    "Decode_test_image" : test_pre
                })
                self.epoch_end(metrics, other_metrics)

        tf.summary.trace_off()
        return


class GAN_Trainer(Trainer):
    def __init__(self,
                 FLAGS,
                 message,
                 data,
                 model,
                 name):
        super().__init__(FLAGS, message, data, model, name)
        self.z_dim = FLAGS.z_dim
        self.n_disc_update = FLAGS.n_disc_update
        self.condtional = FLAGS.conditional

    @tf.function
    def _train_body(self, images, labels):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_logit, real_logit, fake_image = self.model.inference(images, self.batch_size)
            d_loss, g_loss = self.model.loss(fake_logit, real_logit)
        self.model.discriminator_optimize(d_loss, disc_tape, self.n_disc_update)
        self.model.generator_optimize(g_loss, gen_tape)
        acc = self.model.accuracy(real_logit, fake_logit)
        return fake_image, d_loss, g_loss, acc

    @tf.function
    def _test_body(self, images):
        fake_logit = self.model.test_inference(images, self.batch_size*3)
        return fake_logit

    def epoch_end(self, metrics, other=None):
        Discriminator_learning_rate = self.model.d_optimizer.lr(metrics['epoch']).numpy() if type(self.model.d_optimizer.lr) is tf.optimizers.schedules.ExponentialDecay else self.model.d_optimizer.lr
        Generator_learning_rate = self.model.g_optimizer.lr(metrics['epoch']).numpy() if type(self.model.g_optimizer.lr) is tf.optimizers.schedules.ExponentialDecay else self.model.g_optimizer.lr
        tf.summary.experimental.set_step(metrics['epoch'])
        tf.summary.scalar('detail/epoch', metrics['epoch'])
        tf.summary.scalar('detail/time_per_step', metrics['time/epoch'])
        tf.summary.scalar('detail/Discriminator_learning_rate', Discriminator_learning_rate)
        tf.summary.scalar('detail/Generator_learning_rate', Generator_learning_rate)
        tf.summary.scalar('train/Discriminator_loss', metrics['Discriminator_loss'])
        tf.summary.scalar('train/Generator_loss', metrics['Generator_loss'])
        if 'train_image' in other and len(other['train_image'].shape) == 4:
            tf.summary.image('train/image', other['train_image'])
        if 'Decode_train_image' in other:
            tf.summary.image('train/decode_image', other['Decode_train_image'])
            tf.summary.image('test/decode_image', other['Decode_test_image'])
        
        print("epoch: %d  Discriminator_loss: %.4f  Generator_loss: %.4f  train_accuracy: %.3f time/epoch: %0.3fms" 
                                %(metrics['epoch'], metrics['Discriminator_loss'], metrics['Generator_loss'], metrics['train_accuracy'], metrics['time/epoch']))
        self.util.write_log(message=metrics)
        if metrics['epoch'] % self.save_checkpoint_steps == 0:
            self.util.save_model(global_step=metrics['epoch'])
        return

    def train(self):
        board_writer = self.begin_train()
        board_writer.set_as_default()

        if self.restore_dir is not None:
            self.util.restore_agent(self.model ,self.restore_dir)
        
        train_dataset, _ = self.load()
        test_inputs = tf.random.uniform([self.batch_size*3, self.z_dim], dtype=tf.float32)

        for i in range(1, self.n_epoch+1):
            start_time = time.time()
            for (j, (train_images, _)) in enumerate(train_dataset.take(self.batch_size)):
                train_pre, dis_loss, gene_loss, train_accuracy = self._train_body(train_images, None)
            time_per_episode = time.time() - start_time
            test_pre = self._test_body(test_inputs)

            if i == 1 or i % 50 == 0:
                self.util.plot_figure(test_pre.numpy(), i)

            # Training results
            metrics = OrderedDict({
                "epoch": i,
                "Discriminator_loss": dis_loss.numpy(),
                "Generator_loss": gene_loss.numpy(),
                "train_accuracy":train_accuracy,
                "time/epoch": time_per_episode
            })

            #
            other_metrics = OrderedDict({
                "train_image" : train_images[:3],
                "Decode_train_image" : train_pre,
                "Decode_test_image" : test_pre
            })
            self.epoch_end(metrics, other_metrics)
        return
