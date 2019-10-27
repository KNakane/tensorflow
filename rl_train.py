import os, sys, re
from absl import app
from absl import flags
import tensorflow as tf
from rl.setting.cartpole_setting import Cartpole_setting
from rl.setting.pendulum_setting import Pendulum_setting


def main(args):
    if re.search('CartPole', FLAGS.env):
        setting = Cartpole_setting(FLAGS)
    elif re.search('Pendulum', FLAGS.env):
        setting = Pendulum_setting(FLAGS)
    else:
        raise NotImplementedError()

    setting.train()
    return


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_string('agent', 'DQN', 'Choise Agents -> [DQN, DDQN, DDPG, TD3, Rainbow, PolicyGradient, A3C]')
    flags.DEFINE_string('env', 'CartPole-v0','Choice environment, please check env list')
    flags.DEFINE_boolean('duel', 'False', 'dueling network or not')
    flags.DEFINE_integer('n_episode', '100000', 'Input max episode')
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
    flags.DEFINE_integer('n_workers', '1', 'Distributed workers')
    flags.DEFINE_string('init_model','None','Choice the initial model directory')
    flags.DEFINE_boolean('rec', 'False', 'Create test frame -> True/False')
    flags.DEFINE_boolean('test_render', 'False', 'test render -> True/False')
    app.run(main)