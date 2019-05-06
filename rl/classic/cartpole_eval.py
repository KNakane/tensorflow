import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../agents'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utility'))
import gym
from gym import spaces
import tensorflow as tf
from rl_trainer import Trainer, PolicyTrainer, DistributedTrainer
from dqn import DQN,DDQN,Rainbow
from policy_gradient import PolicyGradient
from actor_critic import A3C
from utils import set_output_dim
from cartpole_wrapper import set_model
from collections import OrderedDict


def main(argv):
    env = gym.make('CartPole-v0')
    env = env.unwrapped

    if FLAGS.agent == 'Rainbow':
        FLAGS.network = 'Dueling_Net'
        FLAGS.multi_step = 3
        FLAGS.category = True
        FLAGS.noise = True

    message = OrderedDict({
        "Env": env,
        "Agent": FLAGS.agent,
        "Network": FLAGS.network,
        "Episode": FLAGS.n_episode,
        "Max_Step":FLAGS.step,
        "Categorical": FLAGS.category,
        "init_model": FLAGS.model})

    out_dim = set_output_dim(FLAGS, env.action_space.n)

    agent = eval(FLAGS.agent)(model=set_model(outdim=out_dim),
                n_actions=env.action_space.n,
                n_features=env.observation_space.shape,
                learning_rate=0,
                e_greedy=0,
                reward_decay=0,
                replace_target_iter=0,
                e_greedy_increment=0,
                optimizer=None,
                network=FLAGS.network,
                trainable=False,
                is_categorical=FLAGS.category,
                is_noise=FLAGS.noise
                )

    if FLAGS.agent == 'PolicyGradient':
        trainer = PolicyTrainer(agent=agent, 
                          env=env, 
                          n_episode=FLAGS.n_episode, 
                          max_step=FLAGS.step, 
                          replay_size=0, 
                          data_size=0,
                          n_warmup=0,
                          priority=None,
                          multi_step=0,
                          render=FLAGS.render,
                          test_episode=5,
                          test_interval=0,
                          test_frame=FLAGS.rec,
                          test_render=FLAGS.test_render,
                          metrics=message,
                          init_model_dir=FLAGS.model)
    
    elif FLAGS.agent == 'A3C' or FLAGS.agent == 'Ape_X':
        trainer = DistributedTrainer(agent=agent,
                                     n_workers=0,
                                     env=env, 
                                     n_episode=FLAGS.n_episode, 
                                     max_step=FLAGS.step, 
                                     replay_size=0, 
                                     data_size=0,
                                     n_warmup=0,
                                     priority=None,
                                     multi_step=0,
                                     render=False,
                                     test_episode=5,
                                     test_interval=0,
                                     test_frame=FLAGS.rec,
                                     test_render=FLAGS.test_render,
                                     metrics=message,
                                     init_model_dir=FLAGS.model)
    
    else:
        trainer = Trainer(agent=agent, 
                          env=env, 
                          n_episode=FLAGS.n_episode, 
                          max_step=FLAGS.step, 
                          replay_size=0, 
                          data_size=0,
                          n_warmup=0,
                          priority=None,
                          multi_step=0,
                          render=FLAGS.render,
                          test_episode=5,
                          test_interval=0,
                          test_frame=FLAGS.rec,
                          test_render=FLAGS.test_render,
                          metrics=message,
                          init_model_dir=FLAGS.model)

    trainer.test()


    return


if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('agent', 'DQN', 'Choise Agents -> [DQN, DDQN, Rainbow, A3C]')
    flags.DEFINE_string('network', 'EagerNN', 'Choise Network -> [EagerNN, Dueling_Net]')
    flags.DEFINE_integer('n_episode', '100000', 'Input max episode')
    flags.DEFINE_integer('step', '1000000', 'Input max steps')
    flags.DEFINE_boolean('render', 'False', 'render')
    flags.DEFINE_boolean('category', 'False', 'Categorical DQN')
    flags.DEFINE_boolean('noise', 'False', 'Noisy Net')
    flags.DEFINE_string('model','None','Choice the model directory')
    flags.DEFINE_boolean('rec', 'False', 'Create test frame -> True/False')
    flags.DEFINE_boolean('test_render', 'False', 'test render -> True/False')
    tf.app.run()