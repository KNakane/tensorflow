import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../env'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../agents'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../trainer'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utility'))
import gym, roboschool
import tensorflow as tf
from rl_trainer import Trainer
from ddpg import DDPG, TD3
from utils import set_output_dim
from roboschool_wrapper import set_model,find_gpu


def main(argv):
    env = gym.make(FLAGS.env)
    env = env.unwrapped

    out_dim = set_output_dim(FLAGS, env.action_space.shape[0])

    agent = eval(FLAGS.agent)(model=set_model(outdim=out_dim),
                              n_actions=env.action_space.shape[0],
                              n_features=env.observation_space.shape[0],
                              learning_rate=0,
                              batch_size=0, 
                              e_greedy=0,
                              replace_target_iter=0,
                              e_greedy_increment=0,
                              optimizer=None,
                              is_categorical=FLAGS.category,
                              trainable=False,
                              max_action=env.action_space.high[0],
                              gpu=find_gpu())

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
    print("epoch : {}".format(FLAGS.n_episode))
    print("categorical : {}".format(FLAGS.category))
    print("init_model : {}".format(FLAGS.model))
    print("-----------------------")

    trainer.test()


    return


if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('agent', 'DDPG', 'Choise Agents -> [DDPG, TD3]')
    flags.DEFINE_string('env', 'RoboschoolAnt-v1', 'Choice environment -> [RoboschoolAnt-v1, RoboschoolReacher-v1, RoboschoolHopper-v1, RoboschoolWalker2d-v1, RoboschoolHalfCheetah-v1, RoboschoolHumanoid-v1, RoboschoolHumanoidFlagrun-v1]')
    flags.DEFINE_integer('n_episode', '100000', 'Input max episode')
    flags.DEFINE_integer('step', '1000000', 'Input max steps')
    flags.DEFINE_boolean('render', 'False', 'render')
    flags.DEFINE_boolean('category', 'False', 'Categorical DQN')
    flags.DEFINE_boolean('noise', 'False', 'Noisy Net')
    flags.DEFINE_string('model','None','Choice the model directory')
    flags.DEFINE_boolean('rec', 'False', 'Create test frame -> True/False')
    flags.DEFINE_boolean('test_render', 'False', 'test render -> True/False')
    tf.app.run()