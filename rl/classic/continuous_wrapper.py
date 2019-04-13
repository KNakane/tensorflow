# -*- coding: utf-8 -*-
#tensorboard --logdir ./logs
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../env'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../agents'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utility'))
import gym
import gym.spaces
import numpy as np
import tensorflow as tf
from optimizer import *
from ddpg import DDPG, TD3
from rl_trainer import Trainer
from utils import set_output_dim
from pendulum_env import WrappedPendulumEnv


def set_model(outdim):
    actor = [['fc', 32, tf.nn.relu],
             ['fc', 32, tf.nn.relu],
             ['fc', outdim, tf.nn.tanh]]

    critic = [['fc', 32, tf.nn.relu],
             ['fc', 32, tf.nn.relu],
             ['fc', outdim, None]]
    # Don't forget tf.nn.tanh in activation function
    return actor, critic


def main(argv):
    env = gym.make(FLAGS.env)
    env = env.unwrapped
    if FLAGS.env == 'Pendulum-v0':
        env = WrappedPendulumEnv(env)
        FLAGS.step = 200

    out_dim = set_output_dim(FLAGS.agent, FLAGS.category, env.action_space.shape[0])
    
    agent = eval(FLAGS.agent)(model=set_model(outdim=out_dim),
                              n_actions=env.action_space.shape[0],
                              n_features=env.observation_space.shape[0],
                              learning_rate=FLAGS.lr,
                              batch_size=FLAGS.batch_size, 
                              e_greedy=0.9,
                              replace_target_iter=1,
                              e_greedy_increment=0.01,
                              optimizer=FLAGS.opt,
                              is_categorical=FLAGS.category,
                              max_action=env.action_space.high[0]
                              )

    trainer = Trainer(agent=agent, 
                      env=env, 
                      n_episode=FLAGS.n_episode, 
                      max_step=FLAGS.step, 
                      replay_size=FLAGS.batch_size, 
                      data_size=10**6,
                      n_warmup=FLAGS.n_warmup,
                      priority=FLAGS.priority,
                      multi_step=FLAGS.multi_step,
                      render=FLAGS.render,
                      test_episode=2,
                      test_interval=50,
                      test_frame=FLAGS.rec,
                      test_render=FLAGS.test_render,
                      init_model_dir=FLAGS.init_model)

    print()
    print("---Start Learning------")
    print("data : {}".format(FLAGS.env))
    print("agent : {}".format(FLAGS.agent))
    print("epoch : {}".format(FLAGS.n_episode))
    print("step : {}".format(FLAGS.step))
    print("batch_size : {}".format(FLAGS.batch_size))
    print("learning rate : {}".format(FLAGS.lr))
    print("Optimizer : {}".format(FLAGS.opt))
    print("priority : {}".format(FLAGS.priority))
    print("multi_step : {}".format(FLAGS.multi_step))
    print("categorical : {}".format(FLAGS.category))
    print("n_warmup : {}".format(FLAGS.n_warmup))
    print("model_update : {}".format(FLAGS.model_update))
    if FLAGS.init_model != 'None':
        print("init_model : {}".format(FLAGS.init_model))
    print("-----------------------")

    if FLAGS.init_model != 'None':
        trainer.test(episode=5)
    else:
        trainer.train()


if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('agent', 'DDPG', 'Choise Agents -> [DDPG, TD3]')
    flags.DEFINE_string('env', 'Pendulum-v0', 'Choice environment -> [Pendulum-v0,MountainCarContinuous-v0,LunarLanderContinuous-v2,InvertedPendulum-v2,InvertedDoublePendulum-v2]')
    flags.DEFINE_float('lr', '1e-3', 'Input learning rate')
    flags.DEFINE_string('opt','RMSProp','Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]')
    flags.DEFINE_integer('n_episode', '100000', 'Input max episode')
    flags.DEFINE_integer('step', '400', 'Input max steps')
    flags.DEFINE_integer('batch_size', '32', 'Input batch size')
    flags.DEFINE_integer('multi_step', '1', 'how many multi_step')
    flags.DEFINE_integer('n_warmup', '400', 'n_warmup value')
    flags.DEFINE_integer('model_update', '1', 'target_model_update_freq')
    flags.DEFINE_boolean('render', 'False', 'render')
    flags.DEFINE_boolean('priority', 'False', 'prioritized Experience Replay')
    flags.DEFINE_boolean('category', 'False', 'Categorical DQN')
    flags.DEFINE_boolean('noise', 'False', 'Noisy Net')
    flags.DEFINE_string('init_model','None','Choice the initial model directory')
    flags.DEFINE_boolean('rec', 'False', 'Create test frame -> True/False')
    flags.DEFINE_boolean('test_render', 'False', 'test render -> True/False')
    tf.app.run()