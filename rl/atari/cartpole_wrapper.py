# -*- coding: utf-8 -*-
#tensorboard --logdir ./logs
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../practice/program'))
import tensorflow as tf
from optimizer import *
from model import CNNFunction,NNFunction
import numpy as np
import gym
import gym.spaces
from trainer import Trainer
from dqn import DQN,DDQN


def main(argv):
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    agent = DDQN(model=NNFunction(output_dim=env.action_space.n),
                 n_actions=env.action_space.n,
                 n_features=env.observation_space.shape[0],
                 learning_rate=0.01, e_greedy=0.9,
                 replace_target_iter=100,
                 e_greedy_increment=0.001,
                 optimizer=FLAGS.opt
                 )

    trainer = Trainer(agent=agent, 
                      env=env, 
                      n_episode=FLAGS.n_episode, 
                      max_step=FLAGS.step, 
                      replay_size=FLAGS.batch_size, 
                      data_size=10**6,
                      n_warmup=FLAGS.n_warmup,
                      render=FLAGS.render)

    print()
    print("---Start Learning------")
    print("data : {}".format(env))
    print("epoch : {}".format(FLAGS.n_episode))
    print("step : {}".format(FLAGS.step))
    print("batch_size : {}".format(FLAGS.batch_size))
    print("learning rate : {}".format(FLAGS.lr))
    print("Optimizer : {}".format(FLAGS.opt))
    print("n_warmup : {}".format(FLAGS.n_warmup))
    print("model_update : {}".format(FLAGS.model_update))
    print("-----------------------")
    trainer.train_cartpole()


if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('n_episode', '100000', 'Input max episode')
    flags.DEFINE_integer('step', '10000', 'Input max steps')
    flags.DEFINE_integer('batch_size', '32', 'Input batch size')
    flags.DEFINE_integer('n_warmup', '1000', 'n_warmup value')
    flags.DEFINE_integer('model_update', '1000', 'target_model_update_freq')
    flags.DEFINE_boolean('render', 'False', 'render')
    flags.DEFINE_float('lr', '1e-4', 'Input learning rate')
    flags.DEFINE_string('opt','RMSProp','Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]')
    tf.app.run()