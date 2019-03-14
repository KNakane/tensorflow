# -*- coding: utf-8 -*-
#tensorboard --logdir ./logs
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../env'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../agents'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utility'))
import gym
import gym.spaces
import gym_ple
import numpy as np
import tensorflow as tf
from optimizer import *
from dqn import DQN,DDQN,Rainbow
from rl_trainer import Trainer
from pygame_env import PygameObserver


def set_model(outdim):
    model_set = [['conv', 8, 32, 4, tf.nn.relu],
                 ['conv', 4, 64, 2, tf.nn.relu],
                 ['conv', 3, 64, 1, tf.nn.relu],
                 ['flat'],
                 ['fc', 256, tf.nn.relu],
                 ['fc', outdim, None]]
    return model_set


def main(argv):
    env = gym.make(FLAGS.env)
    env = PygameObserver(env, 84, 84, 4)
    if FLAGS.agent == 'Rainbow':
        FLAGS.network = 'Dueling_Net'
        FLAGS.priority = True
        FLAGS.multi_step = 3
    agent = eval(FLAGS.agent)(model=set_model(outdim=env.action_space.n),
                              n_actions=env.action_space.n,
                              n_features=env.observation_space.shape[0],
                              learning_rate=FLAGS.lr,
                              batch_size=FLAGS.batch_size, 
                              e_greedy=0.9,
                              reward_decay=0.9,
                              replace_target_iter=100,
                              e_greedy_increment=0.0005,
                              optimizer=FLAGS.opt,
                              network=FLAGS.network,
                              is_categorical=FLAGS.category
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
                      test_interval=50)

    print()
    print("---Start Learning------")
    print("data : {}".format(FLAGS.env))
    print("Agent : {}".format(FLAGS.agent))
    print("Network : {}".format(FLAGS.network))
    print("epoch : {}".format(FLAGS.n_episode))
    print("step : {}".format(FLAGS.step))
    print("batch_size : {}".format(FLAGS.batch_size))
    print("learning rate : {}".format(FLAGS.lr))
    print("Optimizer : {}".format(FLAGS.opt))
    print("priority : {}".format(FLAGS.priority))
    print("multi_step : {}".format(FLAGS.multi_step))
    print("n_warmup : {}".format(FLAGS.n_warmup))
    print("model_update : {}".format(FLAGS.model_update))
    print("-----------------------")
    
    if FLAGS.model is not 'None':
        trainer.test()
    else:
        trainer.train()


if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('env', 'Catcher-v0', 'Choise Agents -> [Catcher-v0, FlappyBird-v0, Pong-v0, PixelCopter-v0, MonsterKong-v0, PuckWorld-v0, RaycastMaze-v0, Snake-v0, WaterWorld-v0]')
    flags.DEFINE_string('agent', 'DQN', 'Choise Agents -> [DQN, DDQN]')
    flags.DEFINE_integer('n_episode', '100000', 'Input max episode')
    flags.DEFINE_string('network', 'EagerNN', 'Choise Network -> [EagerNN, Dueling_Net]')
    flags.DEFINE_integer('step', '3000', 'Input max steps')
    flags.DEFINE_integer('batch_size', '32', 'Input batch size')
    flags.DEFINE_integer('multi_step', '1', 'how many multi_step')
    flags.DEFINE_integer('n_warmup', '1000', 'n_warmup value')
    flags.DEFINE_integer('model_update', '1000', 'target_model_update_freq')
    flags.DEFINE_boolean('render', 'False', 'render')
    flags.DEFINE_boolean('priority', 'False', 'prioritized Experience Replay')
    flags.DEFINE_boolean('category', 'False', 'Categorical DQN')
    flags.DEFINE_boolean('noise', 'False', 'Noisy Net')
    flags.DEFINE_float('lr', '1e-4', 'Input learning rate')
    flags.DEFINE_string('model','None','Choice the initial model directory')
    flags.DEFINE_string('opt','RMSProp','Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]')
    tf.app.run()