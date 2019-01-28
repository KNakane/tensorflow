# -*- coding: utf-8 -*-
#tensorboard --logdir ./logs
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utility'))
import random
import threading
import numpy as np
import tensorflow as tf
from collections import deque
from utils import Utils
from display_as_gif import display_frames_as_gif
from replay_memory import ReplayBuffer,PrioritizeReplayBuffer


class Trainer():
    def __init__(self, 
                 agent, 
                 env, 
                 n_episode, 
                 max_step, 
                 replay_size=32, 
                 data_size=10**6,
                 n_warmup=5*10**4,
                 priority=False,
                 multi_step=1,
                 render=False,
                 test_episode=5,
                 test_interval=1000):
        self.agent = agent
        self.env = env
        self.n_episode = n_episode
        self.max_steps = max_step
        self.render = render
        self.data_size = data_size
        self.n_warmup = n_warmup
        self.replay_size = replay_size  # batch_size
        self.multi_step = multi_step
        self.test_episode = test_episode
        self.test_interval = test_interval if test_interval is not None else 10000
        self.util = Utils(prefix=self.agent.__class__.__name__)
        self.util.conf_log() 
        self.replay_buf = PrioritizeReplayBuffer(self.data_size) if priority else ReplayBuffer(self.data_size)
        self.global_step = tf.train.get_or_create_global_step()
        self.state_deque = deque(maxlen=self.multi_step)
        self.reward_deque = deque(maxlen=self.multi_step)
        self.action_deque = deque(maxlen=self.multi_step)
    
    def train(self):
        writer = tf.contrib.summary.create_file_writer(self.util.tf_board)
        writer.set_as_default()
        total_steps = 0
        learning_flag = 0
        for episode in range(self.n_episode):
            self.global_step.assign_add(1)
            with tf.contrib.summary.always_record_summaries():
                state = self.env.reset()
                total_reward = 0
                for step in range(self.max_steps):
                    if self.render:
                        self.env.render()

                    action = self.agent.choose_action(state)
                    state_, reward, done, _ = self.env.step(action)

                    # Multi-step learning
                    self.state_deque.append(state)
                    self.reward_deque.append(reward)
                    self.action_deque.append(action)

                    # the smaller theta and closer to center the better
                    if self.env.__class__.__name__ == 'CartPoleEnv':
                        x, x_dot, theta, theta_dot = state_
                        r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
                        r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                        reward = r1 + r2

                    if len(self.state_deque) == self.multi_step or done:
                        t_reward, reward, p_index = self.multi_step_reward(self.reward_deque, self.agent.gamma)
                        state = self.state_deque[0]
                        action = self.action_deque[0]
                        self.replay_buf.push(state, action, done, state_, t_reward, p_index)

                    total_reward += reward
                    if len(self.replay_buf) > self.replay_size and len(self.replay_buf) > self.n_warmup:
                        indexes, transitions, _ = self.replay_buf.sample(self.agent.batch_size, episode/self.n_episode)
                        train_data = map(np.array, zip(*transitions))
                        self.agent.update_q_net(train_data)
                        learning_flag = 1
                        if len(self.agent.bs[0].shape) == 4:
                            tf.contrib.summary.image('train/input_img', tf.expand_dims(self.agent.bs[:,:,:,0], 3))
                        tf.contrib.summary.scalar('train/loss', self.agent.loss)
                        tf.contrib.summary.scalar('train/e_greedy', self.agent.epsilon)

                        if (indexes != None):
                            for i, value in enumerate(self.agent.td_error):
                                td_error = value
                                self.replay_buf.update(indexes[i], td_error)

                    if done or step == self.max_steps - 1:
                        total_steps += step
                        tf.contrib.summary.scalar('train/total_steps', total_steps)
                        tf.contrib.summary.scalar('train/steps_per_episode', step)
                        tf.contrib.summary.scalar('train/total_reward', total_reward)
                        tf.contrib.summary.scalar('train/average_reward', total_reward / step)
                        print("episode: %d total_steps: %d  steps/episode: %d  total_reward: %0.2f"%(episode, total_steps, step, total_reward))
                        self.state_deque.clear()
                        self.action_deque.clear()
                        self.reward_deque.clear()
                        break

                    state = state_
            # test
            if episode % self.test_interval == 0 and learning_flag:
                frames = []
                test_total_steps = 0
                test_total_reward = 0
                print('-------------------- test -------------------------------------')
                for test_episode in range(self.test_episode):
                    state = self.env.reset()
                    for test_step in range(self.max_steps):
                        #img = self.env.render(mode='rgb_array')
                        #if img is not None:
                        #    frames.append(img)
                        action = self.agent.choose_action(state)
                        next_state, reward, done, _ = self.env.step(action)
                        test_total_reward += reward
                        
                        if done or step == self.max_steps - 1:
                            test_total_steps += test_step
                            display_frames_as_gif(frames, "test_{}_{}".format(episode, test_episode), self.util.res_dir)
                            tf.contrib.summary.scalar('test/total_steps_{}'.format(test_episode), test_total_steps)
                            tf.contrib.summary.scalar('test/steps_per_episode_{}'.format(test_episode), test_step)
                            tf.contrib.summary.scalar('test/total_reward_{}'.format(test_episode), test_total_reward)
                            tf.contrib.summary.scalar('test/average_reward_{}'.format(test_episode), test_total_reward / test_step)
                            print("test_episode: %d total_steps: %d  steps/episode: %d  total_reward: %0.2f"%(test_episode, test_total_steps, test_step, test_total_reward))
                            break
                        state = next_state                     
                print('---------------------------------------------------------------')

        self.env.close()

    def multi_step_reward(self, rewards, gamma):
        ret = 0.
        t_ret = 0.
        for idx, reward in enumerate(rewards):
            ret += reward * (gamma ** idx)
            t_ret += reward
        return ret, t_ret/(idx+1), idx