import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../agents'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utility'))
import gym
from gym import spaces
import tensorflow as tf
from rl_trainer import Trainer
from dqn import DQN,DDQN,Rainbow
from utils import set_output_dim
from pygame_wrapper import set_model
from pygame_env import PygameObserver


def main(argv):
    env = gym.make(FLAGS.env)
    env = PygameObserver(env, 84, 84, 4)

    if FLAGS.agent == 'Rainbow':
        FLAGS.network = 'Dueling_Net'
        FLAGS.category = True
        FLAGS.noise = True

    out_dim = set_output_dim(FLAGS.network, FLAGS.category, env.action_space.n)

    agent = eval(FLAGS.agent)(model=set_model(outdim=out_dim),
                n_actions=env.action_space.n,
                n_features=env.observation_space.shape,
                learning_rate=0,
                e_greedy=0,
                reward_decay=0,
                replace_target_iter=0,
                e_greedy_increment=0,
                trainable=False,
                optimizer=None,
                network=FLAGS.network,
                is_categorical=FLAGS.category,
                is_noise=FLAGS.noise
                )

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
                      init_model_dir=FLAGS.model)

    print()
    print("---Start Learning------")
    print("data : {}".format(FLAGS.env))
    print("Agent : {}".format(FLAGS.agent))
    print("Network : {}".format(FLAGS.network))
    print("epoch : {}".format(FLAGS.n_episode))
    print("categorical : {}".format(FLAGS.category))
    print("model : {}".format(FLAGS.model))
    print("-----------------------")

    trainer.test()


    return


if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('agent', 'DQN', 'Choise Agents -> [DQN, DDQN, Rainbow]')
    flags.DEFINE_string('env', 'Catcher-v0', 'Choise Agents -> [Catcher-v0, FlappyBird-v0, Pong-v0, PixelCopter-v0, MonsterKong-v0, PuckWorld-v0, RaycastMaze-v0, Snake-v0, WaterWorld-v0]')
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