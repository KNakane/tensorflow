# -*- coding: utf-8 -*-
#tensorboard --logdir ./logs
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../agents'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utility'))
import gym
import gym.spaces
import numpy as np
import tensorflow as tf
from optimizer import *
from dqn import DQN,DDQN,Rainbow
from policy_gradient import PolicyGradient
from rl_trainer import Trainer
from utils import set_output_dim


def set_model(outdim):
    model_set = [['fc', 10, tf.nn.relu],
                 ['fc', 10, tf.nn.relu],
                 ['fc', outdim, None]]
    return model_set


def main(argv):
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    if FLAGS.agent == 'Rainbow':
        FLAGS.network = 'Dueling_Net'
        FLAGS.priority = True
        FLAGS.multi_step = 3
        FLAGS.category = True
        FLAGS.noise = True
        FLAGS.opt = 'Adam'
        FLAGS.lr = 0.00025 / 4
    elif FLAGS.agent == 'PolicyGradient':
        FLAGS.multi_step = 30
    
    out_dim = set_output_dim(FLAGS.network, FLAGS.category, env.action_space.n)

    agent = eval(FLAGS.agent)(model=set_model(outdim=out_dim),
                              n_actions=env.action_space.n,
                              n_features=env.observation_space.shape[0],
                              learning_rate=0.01,
                              batch_size=FLAGS.batch_size, 
                              e_greedy=0.9,
                              replace_target_iter=30,
                              e_greedy_increment=0.001,
                              optimizer=FLAGS.opt,
                              network='Dueling_Net' if FLAGS.agent == 'Rainbow' else FLAGS.network,
                              is_categorical=FLAGS.category,
                              is_noise=FLAGS.noise
                              )

    trainer = Trainer(agent=agent, 
                      env=env, 
                      n_episode=FLAGS.n_episode, 
                      max_step=FLAGS.step, 
                      replay_size=FLAGS.batch_size, 
                      data_size=10**6,
                      n_warmup=FLAGS.n_warmup,
                      priority=True if FLAGS.agent == 'Rainbow' else FLAGS.priority,
                      multi_step=3 if FLAGS.agent == 'Rainbow' else FLAGS.multi_step,
                      render=FLAGS.render,
                      test_episode=2,
                      test_interval=50,
                      test_frame=FLAGS.rec,
                      test_render=FLAGS.test_render,
                      init_model_dir=FLAGS.init_model)

    print()
    print("---Start Learning------")
    print("data : {}".format(env))
    print("Agent : {}".format(FLAGS.agent))
    print("epoch : {}".format(FLAGS.n_episode))
    print("Network : {}".format(FLAGS.network))
    print("step : {}".format(FLAGS.step))
    print("batch_size : {}".format(FLAGS.batch_size))
    print("learning rate : {}".format(FLAGS.lr))
    print("Optimizer : {}".format(FLAGS.opt))
    print("priority : {}".format(FLAGS.priority))
    print("multi_step : {}".format(FLAGS.multi_step))
    print("categorical : {}".format(FLAGS.category))
    print("noisy : {}".format(FLAGS.noise))
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
    flags.DEFINE_string('agent', 'DQN', 'Choise Agents -> [DQN, DDQN, Rainbow, PolicyGradient]')
    flags.DEFINE_integer('n_episode', '100000', 'Input max episode')
    flags.DEFINE_string('network', 'EagerNN', 'Choise Network -> [EagerNN, Dueling_Net]')
    flags.DEFINE_float('lr', '1e-4', 'Input learning rate')
    flags.DEFINE_string('opt','RMSProp','Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]')
    flags.DEFINE_integer('step', '1000', 'Input max steps')
    flags.DEFINE_integer('batch_size', '32', 'Input batch size')
    flags.DEFINE_integer('multi_step', '1', 'how many multi_step')
    flags.DEFINE_integer('n_warmup', '1000', 'n_warmup value')
    flags.DEFINE_integer('model_update', '1000', 'target_model_update_freq')
    flags.DEFINE_boolean('render', 'False', 'render')
    flags.DEFINE_boolean('priority', 'False', 'prioritized Experience Replay')
    flags.DEFINE_boolean('category', 'False', 'Categorical DQN')
    flags.DEFINE_boolean('noise', 'False', 'Noisy Net')
    flags.DEFINE_string('init_model','None','Choice the initial model directory')
    flags.DEFINE_boolean('rec', 'False', 'Create test frame -> True/False')
    flags.DEFINE_boolean('test_render', 'False', 'test render -> True/False')
    tf.app.run()