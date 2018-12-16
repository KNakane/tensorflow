import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../practice/program'))
import numpy as np
import pandas as pd
import tensorflow as tf
import gym
from dqn import DQN
from optimizer import *
from model import CNNFunction,NNFunction
from replay_memory import ReplayMemory
from model import CNNFunction,NNFunction
from writer import Writer
from atari_wrapper import make_atari,wrap_deepmind

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            model,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            batch_size=32,
            e_greedy_increment=None,
            optimizer='RMSProp',
    ):
        self.model = model
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.batch_size = batch_size
        self._optimizer = optimizer
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self._iteration = 0

        # consist of [target_net, evaluate_net]
        self._build_net()
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net/Q_net')
        assert len(e_params) > 0
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net/target_net')
        assert len(t_params) > 0

        with tf.variable_scope('replace_op'):
            self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.writer = Writer(self.sess)

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        try:
            self.s = tf.placeholder(tf.float32, [None] + list(self.n_features), name='s')  # input
        except:
            self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            self.q_eval = self.model(inputs=self.s, name='Q_net', trainable=True)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            self.loss_summary = tf.summary.scalar("loss", self.loss)
        
        with tf.variable_scope('train'):
            opt = eval(self._optimizer)(self.lr)
            self._train_op = opt.optimize(self.loss)

        # ------------------ build target_net ------------------
        try:
            self.s_ = tf.placeholder(tf.float32, [None] + list(self.n_features), name='s_')    # input
        except:
            self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            self.q_next = self.model(inputs=self.s_, name='target_net', trainable=False)

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def update_q_net(self, replay_data):
        # check to replace target parameters
        if self._iteration % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
        bs, ba, done, bs_, br = replay_data

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: bs_,  # fixed params
                self.s: bs,  # newest params
            })

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = ba
        reward = br
        done = done

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1) * (1. - done)

        self.merged = tf.summary.merge_all()
        
        # train eval network
        merged, _, self.cost = self.sess.run([self.merged, self._train_op, self.loss],
                                     feed_dict={self.s: bs,
                                                self.q_target: q_target})

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        self._iteration += 1
        self.writer.add(merged, self._iteration)

class Trainer():
    def __init__(self, 
                 agent, 
                 env, 
                 n_episode, 
                 max_step, 
                 replay_size=32, 
                 data_size=10**6,
                 n_warmup=5*10**4,
                 test=False,
                 test_episode=5,
                 render=False):
        self.agent = agent
        self.env = env
        self.n_episode = n_episode
        self.max_steps = max_step
        self.render = render
        self.data_size = data_size
        self.n_warmup = n_warmup
        self.replay_size = replay_size  # batch_size
        self.test = test
        self.test_episode = test_episode
        self.replay_buf = ReplayMemory(self.data_size)

    def train(self):

        for episode in range(self.n_episode):    
            state = self.env.reset()
            total_reward = 0
            for step in range(self.max_steps):
                self.env.render()

                action = self.agent.choose_action(state)

                state_, reward, done, info = self.env.step(action)

                
                # the smaller theta and closer to center the better
                x, x_dot, theta, theta_dot = state_
                r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
                r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                reward = r1 + r2
                

                #RL.store_transition(state, action, reward, state_)
                self.replay_buf.push(state, action, done, state_, reward)

                total_reward += reward
                if len(self.replay_buf) > self.replay_size and len(self.replay_buf) > self.n_warmup:
                    transitions = self.replay_buf.sample(self.agent.batch_size)
                    train_data = map(np.array, zip(*transitions))
                    self.agent.update_q_net(train_data)

                if done or step == self.max_steps - 1:
                    record_dict = dict(step = step,
                                       total_reward = total_reward,
                                       average_reward = total_reward / step)
                    self.agent.writer.add_list(record_dict, episode)
                    print("episode: %d  total_steps: %d  total_reward: %0.2f"%(episode, step, total_reward))
                    break

                state = state_
            pass

        self.agent.writer.save_model(episode)

        if self.test:
            self.agent.writer.restore_model()
            frames = []
            for episode in range(self.test_episode):
                self.env.reset()
                for step in range(self.max_steps):
                    frames.append(self.env.render(mode='rgb_array'))
                    action = self.agent.get_action(state, episode)
                    next_state, reward, done, _ = self.env.step(action)
                    total_reward += reward

                if done:
                    record_dict = dict(step = step,
                                       total_reward = total_reward,
                                       average_reward = total_reward / step)
                    self.agent.writer.add_list(record_dict, episode, True)
                    print("episode: %d  total_steps: %d  total_reward: %0.2f"%(episode, step, total_reward))
                    #display_frames_as_gif(frames,"gif_image", './')

        self.env.close()


def main(args):
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    #env = make_atari(FLAGS.env)
    #env = wrap_deepmind(env, frame_stack=True)
    agent = DeepQNetwork(
                  model=NNFunction(output_dim=env.action_space.n),
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

    trainer.train()

if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('env', 'BreakoutNoFrameskip-v4', 'Choice the environment')
    flags.DEFINE_integer('n_episode', '100000', 'Input max episode')
    flags.DEFINE_integer('step', '10000', 'Input max steps')
    flags.DEFINE_integer('batch_size', '32', 'Input batch size')
    flags.DEFINE_integer('n_warmup', '1000', 'n_warmup value')
    flags.DEFINE_integer('model_update', '1000', 'target_model_update_freq')
    flags.DEFINE_boolean('render', 'False', 'render')
    flags.DEFINE_float('lr', '1e-4', 'Input learning rate')
    flags.DEFINE_string('opt','RMSProp','Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]')
    tf.app.run()